from typing import NamedTuple, Tuple
import networkx as nx
from tqdm import trange, tqdm
import cupy as np
from ..nodes.node import SynapticSignal
from ..nodes.optimization import OptimizableFunction, glue_optimizations
from ..nodes.pipeline import PipelineNode
from ..datasets import dataset


class NetworkException(Exception):
    pass


NodeChannelPair = NamedTuple("NodeChannelPair", [("node", int), ("channel", int)])
NodeChannelPair.standardize = lambda x: (
    x if isinstance(x, NodeChannelPair) else NodeChannelPair(x, 0)
)


class DirectedGraphModel(OptimizableFunction):
    def __init__(self):
        self._graph = nx.DiGraph()
        self._input_nodes = []
        self._output_nodes = []
        self.last_outputs = {}
        self._channel_adj = {}
        self._learning_rate = 0.01

    def add_node(self, node: PipelineNode, input_channel=None, output_channel=None):
        node_idx = len(self._graph)
        self._graph.add_node(node_idx, inner=node, input={}, output={})
        if input_channel is not None:
            self._input_nodes.append(NodeChannelPair(node_idx, input_channel))
        if output_channel is not None:
            self._output_nodes.append(NodeChannelPair(node_idx, output_channel))

    def add_edge(
        self, source: int, target: int, source_channel=None, target_channel=None
    ):
        self._graph.add_edge(source, target)
        try:
            a = nx.topological_sort(self._graph)
            next(a)
        except nx.NetworkXUnfeasible:
            # TODO: add recurrent connections
            self._graph.remove_edge(source, target)
            raise NetworkException("Cycle detected in graph. Cannot add edge.")
        target_channel = target_channel if target_channel is not None else 0
        source_channel = source_channel if source_channel is not None else 0
        target_pair = NodeChannelPair(target, target_channel)
        if source not in self._channel_adj:
            self._channel_adj[source] = {}
        if source_channel not in self._channel_adj[source]:
            self._channel_adj[source][source_channel] = []
        self._channel_adj[source][source_channel].append(target_pair)

    # TODO: assert that input_nodes and output_nodes are in the graph
    def assign_io_nodes(
        self, input_nodes: list | None = None, output_nodes: list | None = None
    ):
        if input_nodes is None:
            self._input_nodes = self._find_input_nodes()
        else:
            self._input_nodes = [NodeChannelPair.standardize(n) for n in input_nodes]
        if output_nodes is None:
            self._output_nodes = self._find_output_nodes()
        else:
            self._output_nodes = [NodeChannelPair.standardize(n) for n in output_nodes]

    def remove_node(self, node: int):
        self._graph.remove_node(node)

    def remove_edge(self, source: int | NodeChannelPair, target: int | NodeChannelPair):
        source = NodeChannelPair.standardize(source)
        target = NodeChannelPair.standardize(target)
        self._graph.remove_edge(source, target)

    # TODO: add channel standardization
    def add_adjacency_matrix(self, adjacency_table: dict):
        self._graph = nx.from_dict_of_lists(adjacency_table)

    # TODO: fix as graph-only method; nodes should not need to standardize inputs
    def _standardize_input(self, inputs, backwards=False):
        node_list = self._output_nodes if backwards else self._input_nodes
        if isinstance(inputs, dict):
            if not isinstance(inputs.keys()[0], NodeChannelPair):
                return {k: {0: v} for k, v in inputs.items()}
            return inputs
        elif isinstance(inputs, list) and len(inputs) == len(node_list):
            return {n.node: {n.channel: i} for n, i in zip(node_list, inputs)}
        elif isinstance(inputs, list):
            return {node_list[0].node: {node_list[0].channel: inputs}}
        return {node_list[0].node: {node_list[0].channel: [inputs]}}

    def _find_input_nodes(self):
        return [
            NodeChannelPair.standardize(n)
            for n in self._graph.nodes
            if not list(self._graph.predecessors(n))
        ]

    def _find_output_nodes(self):
        return [
            NodeChannelPair.standardize(n)
            for n in self._graph.nodes
            if not list(self._graph.successors(n))
        ]

    # finds the inputs that should be fed to a node at node_idx
    # TODO: cache feed addrs so we don't have to recalculate them
    def _get_feed(self, node_idx, suppl_feed: SynapticSignal | None) -> SynapticSignal:
        try:
            input_addrs = []
            for p in self._graph.predecessors(node_idx):
                for outchannel, targets in self._channel_adj[p].items():
                    for t in targets:
                        if t.node == node_idx:
                            input_addrs.append(
                                (NodeChannelPair(p, outchannel), t.channel)
                            )
        except KeyError as e:
            raise NetworkException(f"Node {node_idx} has missing predecessor! {e}")
        feed = suppl_feed if suppl_feed is not None else {}
        feed = feed if isinstance(feed, dict) else {0: feed}
        for (source_idx, source_channel), inchannel in input_addrs:
            if inchannel not in feed:
                feed[inchannel] = []
            feed[inchannel].append(
                self._graph.nodes[source_idx]["output"][source_channel]
            )
        return feed

    def _run_node(self, node_idx, inputs=None):
        feed = self._get_feed(node_idx, inputs)
        try:
            self._graph.nodes[node_idx]["input"] = feed
            self._graph.nodes[node_idx]["output"] = self._graph.nodes[node_idx][
                "inner"
            ].forward(feed)
        except ValueError as e:
            raise NetworkException(f"Axis mismatch on forward node {node_idx}: {e}")
        except AttributeError as e:
            raise NetworkException(
                f"Node {node_idx} has no input, or input is of incorrect form: {e}"
            )

    def _get_backfeed(
        self, node_idx, suppl_feed: SynapticSignal | None
    ) -> SynapticSignal:
        feed = {
            outchannel: [
                self._graph.nodes[source.node]["gradients"][source.channel]
                for source in sources
            ]
            for outchannel, sources in self._channel_adj.get(node_idx, {}).items()
        }
        if suppl_feed is not None:
            for k, v in suppl_feed.items():
                if k not in feed:
                    feed[k] = []
                feed[k] = v + feed[k]
        # NOTE: it seems like mere summation should result in exploding gradients... but it doesn't?
        return {
            outchannel: np.stack(grads).sum(axis=0)
            for outchannel, grads in feed.items()
        }

    def _run_node_backward(self, node_idx, error_gradients):
        backfeed = self._get_backfeed(node_idx, error_gradients)
        try:
            (
                updates,
                self._graph.nodes[node_idx]["gradients"],
            ) = self._graph.nodes[
                node_idx
            ]["inner"].backward(backfeed, self._graph.nodes[node_idx]["input"])
        except ValueError as e:
            raise NetworkException(f"Axis mismatch on backward node {node_idx}: {e}")
        except AttributeError as e:
            raise NetworkException(
                f"Node {node_idx} has no input, or input is of incorrect form: {e}"
            )
        return updates

    def discover_input_and_output_nodes(self):
        self._input_nodes = self._input_nodes or self._find_input_nodes()
        self._output_nodes = self._output_nodes or self._find_output_nodes()
        if not self._input_nodes:
            raise NetworkException("No input nodes found in graph.")
        if not self._output_nodes:
            raise NetworkException("No output nodes found in graph.")

    def forward(self, inputs):
        self.discover_input_and_output_nodes()
        # TODO: remove inter-node standardization: all inter-node coms should be dict[list[ndarray]]
        inputs = self._standardize_input(inputs)
        for n in nx.topological_sort(self._graph):
            outside_inputs = inputs.get(n, None)
            self._run_node(n, outside_inputs)

        self.last_outputs = {
            n: self._graph.nodes[n.node]["output"][n.channel]
            for n in self._output_nodes
        }
        return self.last_outputs

    def backward(self, error_gradients):
        error_gradients = {
            "cost": self._standardize_input(error_gradients, backwards=True)
        }
        updates = {}
        for node_idx in reversed(list(nx.topological_sort(self._graph))):
            cost_feed = error_gradients["cost"].get(node_idx, None)
            updates[node_idx] = self._run_node_backward(node_idx, cost_feed)

        return updates, {
            idx.node: self._graph.nodes[idx.node]["gradients"]
            for idx in self._input_nodes
        }

    def update(self, updates):
        for idx, update in updates.items():
            self._graph.nodes[idx]["inner"].update(
                update, learning_rate=self._learning_rate
            )

    # TODO: update for channels
    # TODO: handle multiple inputs to nodes
    def optimize(
        self,
        rep_idx: int = 0,
        prefix="__model",
        freeze_inits=False,
        freeze_params=False,
    ) -> Tuple[dict, dict]:
        self.discover_input_and_output_nodes()
        my_prefix = f"{prefix}{rep_idx}_node"
        first_input_name = f"self.{prefix}{rep_idx}_first_in"

        my_desc = {
            "__init__": {
                "args": [],
                "body": [],
                "return": [],
            },
            "forward": {
                "args": ["inputs"],
                "body": [f"{first_input_name} = inputs"],
                "return": [],
            },
            "backward": {
                "args": [],
                "body": [],
                "return": [],
            },
        }

        node_characteristics = {}
        var_replaces = {}
        for n in nx.topological_sort(self._graph):
            # TODO: update for channels
            if n in self._input_nodes:
                var_replaces["inputs"] = first_input_name
                var_replaces["last_recorded_input"] = first_input_name
            node_characteristics[n] = self._graph.nodes[n]["inner"].optimize(
                var_replaces, n, my_prefix, freeze_inits, freeze_params
            )
            my_desc = glue_optimizations(
                my_desc, node_characteristics[n], var_replaces, n, my_prefix
            )
        updates = []
        gradients = []
        for line in my_desc["backward"]["body"]:
            if line.strip().startswith("__"):
                varname = line.split(" = ")[0].strip()
                if varname.endswith("_update"):
                    updates.append(varname)
                if varname.endswith("_gradient"):
                    gradient_node_index = varname.split(my_prefix)[1].split(
                        "_gradient"
                    )[0]
                    try:
                        gradient_node_index = int(gradient_node_index)
                        if gradient_node_index in self._input_nodes:
                            gradients.append(varname)
                    except ValueError:
                        pass
        my_desc["backward"]["return"] = [
            "[" + ", ".join(reversed(updates)) + "]",
            "[" + ", ".join(reversed(gradients)) + "]",
        ]
        return my_desc

    def compile_optimized(self, freeze_inits=False, freeze_params=False) -> str:
        desc = self.optimize(freeze_inits=freeze_inits, freeze_params=freeze_params)
        tab = "    "
        new_module_code = ""
        if "__import__" in desc:
            new_module_code += "\n".join(desc["__import__"]) + "\n"
        else:
            # TODO: low priority: handle numpy as option
            new_module_code += "import cupy as np\n"
        new_module_code += "class DGM:\n"
        for k, v in desc.items():
            if k == "__import__":
                continue
            new_module_code += (
                f"{tab}def {k}(self, " + ", ".join(v["args"]) + f"):\n{tab}{tab}"
            )
            new_module_code += f"\n{tab}{tab}".join(v["body"]) + "\n"
            if v["return"]:
                new_module_code += f"{tab}{tab}return {', '.join(v['return'])}\n"
        return new_module_code

    # TODO: gradient weighting
    # TODO: learning rate
    # TODO: handle multiple output nodes
    def train(self, inputs, targets, cost_func: OptimizableFunction, epochs):
        cost_history = {k: [] for k in self._output_nodes}
        for _ in trange(epochs, desc="training..."):
            outputs = self.forward(inputs)
            cost_history = {
                k: cost_history[k] + [cost_func(v, targets[k])]
                for k, v in outputs.items()
            }
            error_gradients = {
                k: cost_func.backward(v, targets[k]) for k, v in outputs.items()
            }
            updates, _ = self.backward(error_gradients)
            self.update(updates)
        return cost_history

    def train_alternate(
        self,
        training_data: dataset.Dataset,
        cost_func: OptimizableFunction,
        epochs,
        batch_size,
    ):
        random_subset_size = 100
        cost_history = {k: [] for k in self._output_nodes}
        batched_epoch_size = len(training_data) // batch_size
        for _ in tqdm(range(epochs), desc="training..."):
            training_data.shuffle()
            subdata = training_data.random_subset(random_subset_size)
            cost_history = {
                k: cost_history[k] + [self.test_on(subdata, cost_func)]
                for k in self._output_nodes
            }
            for batch in tqdm(
                training_data.batch(batch_size),
                total=batched_epoch_size,
                desc="training on batches...",
            ):
                outputs = self.forward(batch._inputs)
                error_gradients = {
                    k: cost_func.backward(outputs, batch._labels)
                    for k in self._output_nodes
                }
                updates, _ = self.backward(error_gradients)
                self.update(updates)
        subdata = training_data.random_subset(random_subset_size)
        cost_history = {
            k: cost_history[k] + [self.test_on(subdata, cost_func)]
            for k in self._output_nodes
        }
        return cost_history

    def train_on(
        self, dataset: dataset.Dataset, cost_func: OptimizableFunction, epochs
    ):
        return self.train(dataset._inputs, dataset._labels, cost_func, epochs)

    # TODO: decide how to handle multiple output nodes vs single output node
    def test(self, inputs, targets, cost_func: OptimizableFunction):
        outputs = self.forward(inputs)
        return cost_func(outputs, targets)

    def test_on(self, dataset: dataset.Dataset, cost_func: OptimizableFunction):
        return self.test(dataset._inputs, dataset._labels, cost_func)
