from typing import Tuple
import networkx as nx
from tqdm import trange, tqdm
import cupy as np
from ..nodes.optimization import OptimizableFunction, glue_optimizations
from ..nodes.pipeline import PipelineNode
from ..datasets import dataset


class NetworkException(Exception):
    pass


class DirectedGraphModel(OptimizableFunction):
    def __init__(self):
        self._graph = nx.DiGraph()
        self._input_nodes = []
        self._output_nodes = []
        self.last_outputs = {}
        self._site_adj = {}
        self._learning_rate = 0.01

    def add_node(self, node: PipelineNode, is_input=False, is_output=False):
        node_idx = len(self._graph)
        self._graph.add_node(node_idx, inner=node)
        if is_input:
            self._input_nodes.append(node_idx)
        if is_output:
            self._output_nodes.append(node_idx)

    def add_edge(self, source: int, target: int, input_site=None, output_site=None):
        self._graph.add_edge(source, target)
        try:
            a = nx.topological_sort(self._graph)
            next(a)
        except nx.NetworkXUnfeasible:
            self._graph.remove_edge(source, target)
            raise NetworkException(
                "Cycle detected in graph. Cannot add edge.")
        output_site = output_site if output_site is not None else 0
        input_site = input_site if input_site is not None else 0
        self._site_adj[source] = (output_site, input_site)

    def assign_io_nodes(self, input_nodes: list, output_nodes: list):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes

    def remove_node(self, node: int):
        self._graph.remove_node(node)

    def remove_edge(self, source: int, target: int):
        self._graph.remove_edge(source, target)

    def add_adjacency_matrix(self, adjacency_table: dict):
        self._graph = nx.from_dict_of_lists(adjacency_table)

    def _standardize_input(self, inputs, backwards=False):
        node_list = self._output_nodes if backwards else self._input_nodes
        if isinstance(inputs, dict):
            return inputs
        elif isinstance(inputs, list) and len(inputs) == len(node_list):
            return dict(zip(node_list, inputs))
        return {node_list[0]: inputs}

    def _find_input_nodes(self):
        return [n for n in self._graph.nodes if not list(self._graph.predecessors(n))]

    def _find_output_nodes(self):
        return [next(reversed(list(nx.topological_sort(self._graph))))]

    def discover_input_and_output_nodes(self):
        self._input_nodes = self._find_input_nodes()
        self._output_nodes = self._find_output_nodes()

    def forward(self, inputs):
        self._output_nodes = self._output_nodes or self._find_output_nodes()
        self._input_nodes = self._input_nodes or self._find_input_nodes()
        if not self._input_nodes:
            raise NetworkException("No input nodes found in graph.")
        if not self._output_nodes:
            raise NetworkException("No output nodes found in graph.")
        inputs = self._standardize_input(inputs)
        for n in nx.topological_sort(self._graph):
            staged_input = [
                self._graph.nodes[p]['inner'].last_output for p in self._graph.predecessors(n)] or inputs.get(n, [])
            staged_input = staged_input if len(
                staged_input) > 1 else staged_input[0]
            try:
                self._graph.nodes[n]['inner'].forward(staged_input)
            except ValueError:
                staged_shape = staged_input.shape if isinstance(
                    staged_input, np.ndarray) else [n.shape for n in staged_input]
                raise NetworkException(f"Axis mismatch on forward node {n} (input from nodes: {list(self._graph.predecessors(n))}) which expects {
                                       self._graph.nodes[n]['inner'].input_shape} but got {staged_shape} instead.")
            except AttributeError as e:
                raise NetworkException(
                    f"Node {n} has no input, or input is of incorrect form: {e}")
        self._output_nodes = self._output_nodes or []
        self.last_outputs = {
            n: self._graph.nodes[n]['inner'].last_output for n in self._output_nodes}
        return self.last_outputs if len(self.last_outputs) > 1 else self.last_outputs[self._output_nodes[0]]

    def backward(self, error_gradients):
        graph_rev = reversed(list(nx.topological_sort(self._graph)))
        error_gradients = {"cost": self._standardize_input(
            error_gradients, backwards=True)}
        updates = {}
        for node_idx in graph_rev:
            gradients = [error_gradients[p]
                         for p in self._graph.successors(node_idx)]
            if node_idx in self._output_nodes:
                g = error_gradients["cost"][node_idx]
                if isinstance(g, list):
                    gradients.extend(g)
                else:
                    gradients.append(g)
            total_grad = sum(gradients)
            try:
                update, gradient = self._graph.nodes[node_idx]['inner'].backward(
                    total_grad)
                updates[node_idx] = update
                error_gradients[node_idx] = gradient
            except ValueError:
                raise NetworkException(f"Axis mismatch on backward node {node_idx} (gradients frome nodes: {list(self._graph.successors(node_idx))}) which expects {
                                       self._graph.nodes[node_idx]['inner'].input_shape} but got {total_grad.shape} instead.")
        return updates, {idx: error_gradients[idx] for idx in self._input_nodes}

    def update(self, updates):
        for idx, update in updates.items():
            self._graph.nodes[idx]['inner'].update(
                update, learning_rate=self._learning_rate)

    # TODO: track input & output sites
    # TODO: handle multiple inputs to nodes
    def optimize(self, rep_idx: int = 0, prefix="__model", freeze_inits=False, freeze_params=False) -> Tuple[dict, dict]:
        self._output_nodes = self._output_nodes or self._find_output_nodes()
        self._input_nodes = self._input_nodes or self._find_input_nodes()
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
            if n in self._input_nodes:
                var_replaces["inputs"] = first_input_name
                var_replaces["last_recorded_input"] = first_input_name
            node_characteristics[n] = self._graph.nodes[n]['inner'].optimize(
                var_replaces, n, my_prefix, freeze_inits, freeze_params)
            my_desc = glue_optimizations(
                my_desc, node_characteristics[n], var_replaces, n, my_prefix)
        updates = []
        gradients = []
        for line in my_desc["backward"]["body"]:
            if line.strip().startswith("__"):
                varname = line.split(" = ")[0].strip()
                if varname.endswith("_update"):
                    updates.append(varname)
                if varname.endswith("_gradient"):
                    gradient_node_index = varname.split(my_prefix)[1].split("_gradient")[0]
                    try:
                        gradient_node_index = int(gradient_node_index)
                        if gradient_node_index in self._input_nodes:
                            gradients.append(varname)
                    except ValueError:
                        pass
        my_desc["backward"]["return"] = ['[' + ', '.join(reversed(updates)) + ']', '[' + ', '.join(reversed(gradients)) + ']']
        return my_desc

    def compile_optimized(self, freeze_inits=False, freeze_params=False) -> str:
        desc = self.optimize(freeze_inits=freeze_inits, freeze_params=freeze_params)
        tab = "    "
        new_module_code = ""
        if "__import__" in desc:
            new_module_code += '\n'.join(desc["__import__"]) + '\n'
        else:
            # TODO: low priority: handle numpy as option
            new_module_code += "import cupy as np\n"
        new_module_code += "class DGM:\n"
        for k, v in desc.items():
            if k == "__import__":
                continue
            new_module_code += f"{tab}def {k}(self, " + ', '.join(v["args"]) + f"):\n{tab}{tab}"
            new_module_code += f'\n{tab}{tab}'.join(v["body"]) + '\n'
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
            cost_history = {k: cost_history[k] + [cost_func(
                v, targets[k])] for k, v in outputs.items()}
            error_gradients = {k: cost_func.backward(
                v, targets[k]) for k, v in outputs.items()}
            updates, _ = self.backward(error_gradients)
            self.update(updates)
        return cost_history

    def train_alternate(self, training_data: dataset.Dataset, cost_func: OptimizableFunction, epochs, batch_size):
        random_subset_size = 100
        cost_history = {k: [] for k in self._output_nodes}
        batched_epoch_size = len(training_data) // batch_size
        for _ in tqdm(range(epochs), desc="training..."):
            training_data.shuffle()
            subdata = training_data.random_subset(random_subset_size)
            cost_history = {
                k: cost_history[k] + [self.test_on(subdata, cost_func)] for k in self._output_nodes}
            for batch in tqdm(training_data.batch(batch_size), total=batched_epoch_size, desc="training on batches..."):
                outputs = self.forward(batch._inputs)
                error_gradients = {k: cost_func.backward(
                    outputs, batch._labels) for k in self._output_nodes}
                updates, _ = self.backward(error_gradients)
                self.update(updates)
        subdata = training_data.random_subset(random_subset_size)
        cost_history = {
            k: cost_history[k] + [self.test_on(subdata, cost_func)] for k in self._output_nodes}
        return cost_history

    def train_on(self, dataset: dataset.Dataset, cost_func: OptimizableFunction, epochs):
        return self.train(dataset._inputs, dataset._labels, cost_func, epochs)

    # TODO: decide how to handle multiple output nodes vs single output node
    def test(self, inputs, targets, cost_func: OptimizableFunction):
        outputs = self.forward(inputs)
        return cost_func(outputs, targets)

    def test_on(self, dataset: dataset.Dataset, cost_func: OptimizableFunction):
        return self.test(dataset._inputs, dataset._labels, cost_func)
