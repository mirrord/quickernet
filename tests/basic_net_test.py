import cupy as np
from quickernet.nodes.linear import Linear
from quickernet.nodes.activations import Sigmoid
from quickernet.nodes.node import PipelineNode
from quickernet.nodes.costs import QuadraticCost
from quickernet.networks.graph import DirectedGraphModel

sigmoid = np.ElementwiseKernel(
    "float64 x", "float64 y", "y = 1 / (1 + exp(-x))", "expit"
)


def dsigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


def diff_squares(y, y_true):
    return np.sum(np.square(y - y_true)).item()


def ddiff_squares(y, y_true):
    return 2 * (y - y_true)


def classic_net_predict(weights, biases, input):
    for b, w in zip(biases, weights):
        input = sigmoid(np.dot(input, w) + b)
    return input


def classic_single_node_backprop(weights, biases, input, err_grad):
    zs = np.dot(input, weights) + biases

    delta = err_grad * dsigmoid(zs)

    nabla_b = delta
    nabla_w = np.dot(input.T, delta)
    return (nabla_b, nabla_w), np.dot(delta, weights.T)


def classic_net_backprop(weights, biases, input, exp_out):
    activations = [input]  # list to store all the activations, layer by layer
    zs = []  # list to store all the z vectors, layer by layer
    for b, w in zip(biases, weights):
        zs.append(np.dot(activations[-1], w) + b)
        activations.append(sigmoid(zs[-1]))

    # error calc
    delta = ddiff_squares(activations[-1], exp_out)

    # gradient calc
    delta = delta * dsigmoid(zs[-1])
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(activations[-2].T, delta)
    gradients = [delta]
    for layer_idx in range(2, len(weights) + 1):
        nabla_b[-layer_idx] = np.dot(
            nabla_b[-layer_idx + 1], weights[-layer_idx + 1].T
        ) * dsigmoid(zs[-layer_idx])
        nabla_w[-layer_idx] = np.dot(activations[-layer_idx -
                                     1].T, nabla_b[-layer_idx])
        gradients.insert(0, nabla_b[-layer_idx])
    return (nabla_b, nabla_w), gradients


def test_pipeline():
    # test the linear node
    input_dim = 5
    output_dim = 3
    num_samples = 4
    lin = Linear(input_dim, output_dim)
    pipenode = PipelineNode([lin, Sigmoid()])
    graph = DirectedGraphModel()
    graph.add_node(pipenode)
    input_nodes = graph._find_input_nodes()
    assert (input_nodes == [0])
    output_nodes = graph._find_output_nodes()
    assert (output_nodes == [0])
    x = np.random.randn(num_samples, input_dim)
    dgn_output = graph.forward(x)
    assert (dgn_output.shape == (num_samples, output_dim))
    classic_out = classic_net_predict(
        [lin.weight], [lin.bias], x.copy())
    assert (np.allclose(dgn_output, classic_out))

    # test the backpropagation
    expected_out = np.random.randn(num_samples, output_dim)
    error_gradient = QuadraticCost().backward(dgn_output, expected_out)
    classic_gradient = ddiff_squares(dgn_output, expected_out)
    assert (np.allclose(error_gradient, classic_gradient))

    dgn_node_updates, dgn_gradient = pipenode.backward(error_gradient)
    classic_node_updates, classic_gradient = classic_single_node_backprop(
        lin.weight, lin.bias, x, error_gradient)
    assert (np.allclose(dgn_gradient, classic_gradient))
    assert (np.allclose(dgn_node_updates[0][0], classic_node_updates[0]))
    assert (np.allclose(dgn_node_updates[0][1], classic_node_updates[1]))

    dgn_updates, dgn_gradients = graph.backward(error_gradient)
    classic_updates, classic_gradients = classic_net_backprop(
        [lin.weight], [lin.bias], x.copy(), expected_out)
    assert (np.allclose(dgn_updates[0][0][0], classic_updates[0][0]))
    assert (np.allclose(dgn_updates[0][0][1], classic_updates[1][0]))

    # now two pipeline nodes
    lin2 = Linear(output_dim, 2)
    pipenode2 = PipelineNode([lin2, Sigmoid()])
    graph.add_node(pipenode2)
    graph.add_edge(0, 1)
    input_nodes = graph._find_input_nodes()
    assert (input_nodes == [0])
    output_nodes = graph._find_output_nodes()
    assert (output_nodes == [1])
    graph.discover_input_and_output_nodes()
    x = np.random.randn(num_samples, input_dim)
    dgn_output = graph.forward(x)
    classic_out = classic_net_predict(
        [lin.weight, lin2.weight], [lin.bias, lin2.bias], x.copy())
    assert (np.allclose(dgn_output, classic_out))

    # test the backpropagation
    expected_out = np.random.randn(num_samples, 2)
    error_gradient = QuadraticCost().backward(dgn_output, expected_out)
    classic_gradient = ddiff_squares(dgn_output, expected_out)
    assert (np.allclose(error_gradient, classic_gradient))

    dgn_updates, dgn_gradients = graph.backward(error_gradient)
    classic_updates, classic_gradients = classic_net_backprop(
        [lin.weight, lin2.weight], [lin.bias, lin2.bias], x, expected_out)
    # first node bias
    assert (np.allclose(dgn_updates[0][0][0], classic_updates[0][0]))
    # first node weight
    assert (np.allclose(dgn_updates[0][0][1], classic_updates[1][0]))
    # second node bias
    assert (np.allclose(dgn_updates[1][0][0], classic_updates[0][1]))
    # second node weight
    assert (np.allclose(dgn_updates[1][0][1], classic_updates[1][1]))
