import cupy as np
from quickernet.nodes.linear import Linear
from quickernet.nodes.activations import Sigmoid
from quickernet.nodes.pipeline import PipelineNode
from quickernet.nodes.costs import DiffSquaresCost
from quickernet.networks.graph import DirectedGraphModel, NodeChannelPair
from tests.utils import (
    classic_net_backprop,
    classic_net_predict,
    classic_single_node_backprop,
    ddiff_squares,
)


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
    assert input_nodes == [NodeChannelPair(0, 0)]
    output_nodes = graph._find_output_nodes()
    assert output_nodes == [NodeChannelPair(0, 0)]
    x = np.random.randn(num_samples, input_dim)
    dgn_output = graph.forward(x)[NodeChannelPair(0, 0)]
    assert dgn_output.shape == (num_samples, output_dim)
    classic_out = classic_net_predict([lin.weight], [lin.bias], x.copy())
    assert np.allclose(dgn_output, classic_out)

    # test the backpropagation
    expected_out = np.random.randn(num_samples, output_dim)
    error_gradient = DiffSquaresCost().backward(dgn_output, expected_out)
    classic_gradient = ddiff_squares(dgn_output, expected_out)
    assert np.allclose(error_gradient, classic_gradient)

    x_d = {0: [x]}
    error_d = {0: error_gradient}
    dgn_node_updates, dgn_gradient = pipenode.backward(error_d, x_d)
    classic_node_updates, classic_gradient = classic_single_node_backprop(
        lin.weight, lin.bias, x, error_gradient
    )
    assert np.allclose(dgn_gradient[0], classic_gradient)
    assert np.allclose(dgn_node_updates[0][0], classic_node_updates[0])
    assert np.allclose(dgn_node_updates[0][1], classic_node_updates[1])

    dgn_updates, dgn_gradients = graph.backward(error_gradient)
    classic_updates, classic_gradients = classic_net_backprop(
        [lin.weight], [lin.bias], x.copy(), expected_out
    )
    assert np.allclose(dgn_updates[0][0][0], classic_updates[0][0])
    assert np.allclose(dgn_updates[0][0][1], classic_updates[1][0])

    # now two pipeline nodes
    lin2 = Linear(output_dim, 2)
    pipenode2 = PipelineNode([lin2, Sigmoid()])
    graph.add_node(pipenode2)
    graph.add_edge(0, 1)
    graph.assign_io_nodes()
    input_nodes = graph._find_input_nodes()
    assert input_nodes == [NodeChannelPair(0, 0)]
    output_nodes = graph._find_output_nodes()
    assert output_nodes == [NodeChannelPair(1, 0)]
    graph.discover_input_and_output_nodes()
    x = np.random.randn(num_samples, input_dim)
    dgn_output = graph.forward(x)[NodeChannelPair(1, 0)]
    classic_out = classic_net_predict(
        [lin.weight, lin2.weight], [lin.bias, lin2.bias], x.copy()
    )
    assert np.allclose(dgn_output, classic_out)

    # test the backpropagation
    expected_out = np.random.randn(num_samples, 2)
    error_gradient = DiffSquaresCost().backward(dgn_output, expected_out)
    classic_gradient = ddiff_squares(dgn_output, expected_out)
    assert np.allclose(error_gradient, classic_gradient)

    dgn_updates, dgn_gradients = graph.backward(error_gradient)
    classic_updates, classic_gradients = classic_net_backprop(
        [lin.weight, lin2.weight], [lin.bias, lin2.bias], x, expected_out
    )
    # first node bias
    assert np.allclose(dgn_updates[0][0][0], classic_updates[0][0])
    # first node weight
    assert np.allclose(dgn_updates[0][0][1], classic_updates[1][0])
    # second node bias
    assert np.allclose(dgn_updates[1][0][0], classic_updates[0][1])
    # second node weight
    assert np.allclose(dgn_updates[1][0][1], classic_updates[1][1])
