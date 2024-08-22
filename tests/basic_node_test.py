import cupy as np
from quickernet.nodes.linear import Linear
from quickernet.nodes.activations import Sigmoid
from quickernet.nodes.pipeline import PipelineNode
from tests.utils import classic_net_predict, classic_single_node_backprop


def test_linear():
    input_dim = 5
    output_dim = 3
    lin = Linear(input_dim, output_dim)
    test_input = np.array([1, 2, 3, 4, 5], ndmin=2)
    forward_output = lin.forward(test_input)
    classic_output = np.dot(test_input, lin.weight) + lin.bias
    assert np.allclose(forward_output, classic_output)
    assert forward_output.shape == (1, output_dim)

    test_grad = np.array([1, 2, 3], ndmin=2)
    updates, gradient = lin.backward(test_grad, test_input)
    classic_gradient = np.dot(test_grad, lin.weight.T)
    classic_weight_gradient = np.dot(test_input.T, test_grad)
    assert np.allclose(gradient, classic_gradient)

    classic_update = (test_grad, classic_weight_gradient)
    assert np.allclose(updates[0], classic_update[0])
    assert np.allclose(updates[1], classic_update[1])

    classic_updated_weight = lin.weight - 0.1 * classic_weight_gradient
    classic_updated_bias = lin.bias - 0.1 * test_grad
    lin.update(updates, 0.1)
    assert np.allclose(lin.weight, classic_updated_weight)
    assert np.allclose(lin.bias, classic_updated_bias)


def test_pipeline_trivial():
    # test the linear node
    input_dim = 5
    output_dim = 3
    num_samples = 4
    lin = Linear(input_dim, output_dim)
    pipenode = PipelineNode([lin])
    x = np.random.randn(num_samples, input_dim)
    x_d = {0: [x]}
    dgn_output = pipenode.forward(x_d)[0]
    assert dgn_output.shape == (num_samples, output_dim)
    classic_out = np.dot(x, lin.weight) + lin.bias
    assert np.allclose(dgn_output, classic_out)

    # test the backpropagation
    expected_out = np.random.randn(num_samples, output_dim)
    error_gradient = dgn_output - expected_out
    classic_gradient = np.dot(error_gradient, lin.weight.T)
    dgn_node_updates, dgn_gradient = pipenode.backward({0: error_gradient}, x_d)
    dgn_gradient = dgn_gradient[0]
    dgn_node_updates = dgn_node_updates[0]
    classic_node_updates = (error_gradient, np.dot(x.T, error_gradient))
    assert np.allclose(dgn_gradient, classic_gradient)
    assert np.allclose(dgn_node_updates[0], classic_node_updates[0])
    assert np.allclose(dgn_node_updates[1], classic_node_updates[1])


def test_pipeline_simple():
    # test the linear node
    input_dim = 5
    output_dim = 3
    num_samples = 4
    lin = Linear(input_dim, output_dim)
    sig = Sigmoid()
    pipenode = PipelineNode([lin, sig])
    x = np.random.randn(num_samples, input_dim)
    x_d = {0: [x]}
    dgn_output = pipenode.forward(x_d)[0]
    assert dgn_output.shape == (num_samples, output_dim)
    classic_out = classic_net_predict([lin.weight], [lin.bias], x.copy())
    assert np.allclose(dgn_output, classic_out)
    test_grad = np.random.randn(num_samples, output_dim)
    updates, gradient = pipenode.backward({0: test_grad}, x_d)
    gradient = gradient[0]
    classic_updates, classic_gradient = classic_single_node_backprop(
        lin.weight, lin.bias, x.copy(), test_grad
    )
    assert np.allclose(gradient, classic_gradient)
    assert np.allclose(updates[0][0], classic_updates[0])
    assert np.allclose(updates[0][1], classic_updates[1])
