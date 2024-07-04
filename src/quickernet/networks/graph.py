import networkx as nx
from tqdm import trange, tqdm
from ..nodes import node, utils
from ..datasets import dataset


class NetworkException(Exception):
    pass


class DirectedGraphModel(utils.OptimizableFunction):
    def __init__(self):
        self._graph = nx.DiGraph()
        self._next_label = 0
        self._input_nodes = []
        self._output_nodes = []
        self.last_outputs = {}
        self._learning_rate = 0.01

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

    def _find_input_nodes(self):
        return [n for n in self._graph.nodes if not list(self._graph.predecessors(n))]

    def _find_output_nodes(self):
        return [next(reversed(list(nx.topological_sort(self._graph))))]

    def discover_input_and_output_nodes(self):
        self._input_nodes = self._find_input_nodes()
        self._output_nodes = self._find_output_nodes()

    def forward(self, inputs):
        graph_walker = nx.topological_sort(self._graph)
        self._output_nodes = self._output_nodes or self._find_output_nodes()
        self._input_nodes = self._input_nodes or self._find_input_nodes()
        if not self._input_nodes:
            raise NetworkException("No input nodes found in graph.")
        if not self._output_nodes:
            raise NetworkException("No output nodes found in graph.")
        inputs = self._standardize_input(inputs)
        for n in graph_walker:
            staged_input = [
                self._graph.nodes[p]['inner'].last_output for p in self._graph.predecessors(n)] or inputs.get(n, [])
            staged_input = staged_input if len(
                staged_input) > 1 else staged_input[0]
            try:
                self._graph.nodes[n]['inner'].forward(staged_input)
            except ValueError:
                raise NetworkException(f"Axis mismatch on forward node {n} (input frome nodes: {list(self._graph.predecessors(n))}) which expects {
                                       self._graph.nodes[n]['inner'].input_shape} but got {[n.shape for n in staged_input]} instead.")
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
            self._graph.nodes[idx]['inner'].update(
                update, learning_rate=self._learning_rate)

    def optimize(self):
        pass  # TODO: implement this

    # TODO: gradient weighting
    # TODO: learning rate
    # TODO: handle multiple output nodes
    def train(self, inputs, targets, cost_func: utils.OptimizableFunction, epochs):
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

    def train_alternate(self, training_data: dataset.Dataset, cost_func: utils.OptimizableFunction, epochs, batch_size):
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

    def train_on(self, dataset: dataset.Dataset, cost_func: utils.OptimizableFunction, epochs):
        return self.train(dataset._inputs, dataset._labels, cost_func, epochs)

    # TODO: decide how to handle multiple output nodes vs single output node
    def test(self, inputs, targets, cost_func: utils.OptimizableFunction):
        outputs = self.forward(inputs)
        return cost_func(outputs, targets)

    def test_on(self, dataset: dataset.Dataset, cost_func: utils.OptimizableFunction):
        return self.test(dataset._inputs, dataset._labels, cost_func)

    # TODO: this is not the right place for this function - move to utils
    def get_accuracy(self, dataset: dataset.Dataset):
        def multiclass_precision_and_recall(correct_preds, all_predictions, target_class, labels):
            # failed because all_predictions is a single-element array
            predictions_in_class = all_predictions == target_class
            true_pos = correct_preds[predictions_in_class].sum().item()
            false_pos = (sum(predictions_in_class) - true_pos).item()
            false_neg = sum((~predictions_in_class)[
                            labels == target_class]).item()
            prec_denom = true_pos + false_pos
            rec_denom = true_pos + false_neg
            precision = 0 if prec_denom == 0 else true_pos / prec_denom
            recall = 0 if rec_denom == 0 else true_pos / rec_denom
            return precision, recall

        dataset.debinarize()
        num_classes = dataset.num_classes()
        expected_outputs = dataset._labels
        # result is a single-element array!!!
        forward_output = self.forward(dataset._inputs)
        outputs = utils.debinarize(forward_output)
        correct_preds = outputs == expected_outputs
        accuracies = {'all': (sum(correct_preds) / len(outputs)).item()}
        accuracies.update({f"class_{i}": (multiclass_precision_and_recall(
            correct_preds, outputs, i, dataset._labels)) for i in range(num_classes)})
        dataset.binarize(num_classes)
        return accuracies
