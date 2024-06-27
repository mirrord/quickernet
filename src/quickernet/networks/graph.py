import networkx as nx

from ..nodes import node, utils


class NetworkException(Exception):
    pass


class DirectedGraphModel(utils.OptimizableFunction):
    def __init__(self):
        self._graph = nx.DiGraph()
        self._next_label = 0
        self._input_nodes = []
        self._output_nodes = []
        self.last_outputs = {}

    def add_node(self, node: node.PipelineNode, is_input=False, is_output=False):
        self._graph.add_node(self._next_label, inner=node)
        self._next_label += 1

    # TODO: check for cycles
    def add_edge(self, source: int, target: int):
        self._graph.add_edge(source, target)

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

    def forward(self, inputs):
        graph_walker = nx.topological_sort(self._graph)
        self._output_nodes = self._output_nodes or [
            next(reversed(list(nx.topological_sort(self._graph))))]
        self._input_nodes = self._input_nodes or [next(graph_walker)]
        if not self._input_nodes:
            raise NetworkException("No input nodes found in graph.")
        inputs = self._standardize_input(inputs)
        for input_idx in self._input_nodes:
            try:
                self._graph.nodes[input_idx]['inner'].forward(
                    inputs[input_idx])
            except ValueError:
                raise NetworkException(f"Axis mismatch on network input (input node {input_idx}) which expects {
                                       self._graph.nodes[input_idx]['inner'].input_shape} but got {inputs[input_idx].shape} instead.")
        for n in graph_walker:
            staged_input = [
                self._graph.nodes[p]['inner'].last_output for p in self._graph.predecessors(n)]
            try:
                self._graph.nodes[n]['inner'].forward(staged_input)
            except ValueError:
                raise NetworkException(f"Axis mismatch on forward node {n} (input frome nodes: {list(self._graph.predecessors(n))}) which expects {
                                       self._graph.nodes[n]['inner'].input_shape} but got {[n.shape for n in staged_input]} instead.")
        self._output_nodes = self._output_nodes or []
        self.last_outputs = {
            n: self._graph.nodes[n]['inner'].last_output for n in self._output_nodes}
        return self.last_outputs

    def backward(self, error_gradients):
        graph_rev = reversed(list(nx.topological_sort(self._graph)))
        error_gradients = {"cost": self._standardize_input(
            error_gradients, backwards=True)}
        updates = {}
        for node_idx in graph_rev:
            gradients = [error_gradients[p]
                         for p in self._graph.successors(node_idx)]
            if node_idx in self._output_nodes:
                gradients.extend(error_gradients["cost"][node_idx])
            avg_grad = sum(gradients) / len(gradients)
            try:
                update, gradient = self._graph.nodes[node_idx]['inner'].backward(
                    avg_grad)
                updates[node_idx] = update
                error_gradients[node_idx] = gradient
            except ValueError:
                raise NetworkException(f"Axis mismatch on backward node {node_idx} (gradients frome nodes: {list(self._graph.successors(node_idx))}) which expects {
                                       self._graph.nodes[node_idx]['inner'].input_shape} but got {avg_grad.shape} instead.")
        return updates, {idx: error_gradients[idx] for idx in self._input_nodes}

    def update(self, updates):
        for idx, update in updates.items():
            self._graph.nodes[idx]['inner'].update(update)

    def optimize(self):
        pass  # TODO: implement this

    # TODO: batching
    # TODO: gradient weighting
    # TODO: learning rate
    def train(self, inputs, targets, cost_func: utils.OptimizableFunction, epochs):
        cost_history = {k: [] for k in self._output_nodes}
        for _ in range(epochs):
            outputs = self.forward(inputs)
            cost_history = {k: cost_history[k] + cost_func(
                v, targets[k]) for k, v in outputs.items()}
            error_gradients = {k: cost_func.backward(
                v, targets[k]) for k, v in outputs.items()}
            updates, _ = self.backward(error_gradients)
            self.update(updates)
        return cost_history
