import cupy as np
from quickernet.nodes.synapses import SynapseSum
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
        nabla_w[-layer_idx] = np.dot(activations[-layer_idx - 1].T, nabla_b[-layer_idx])
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
    assert input_nodes == [0]
    output_nodes = graph._find_output_nodes()
    assert output_nodes == [0]
    x = np.random.randn(num_samples, input_dim)
    dgn_output = graph.forward(x)
    assert dgn_output.shape == (num_samples, output_dim)
    classic_out = classic_net_predict([lin.weight], [lin.bias], x.copy())
    assert np.allclose(dgn_output, classic_out)

    # test the backpropagation
    expected_out = np.random.randn(num_samples, output_dim)
    error_gradient = QuadraticCost().backward(dgn_output, expected_out)
    classic_gradient = ddiff_squares(dgn_output, expected_out)
    assert np.allclose(error_gradient, classic_gradient)

    dgn_node_updates, dgn_gradient = pipenode.backward(error_gradient)
    classic_node_updates, classic_gradient = classic_single_node_backprop(
        lin.weight, lin.bias, x, error_gradient
    )
    assert np.allclose(dgn_gradient, classic_gradient)
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
    input_nodes = graph._find_input_nodes()
    assert input_nodes == [0]
    output_nodes = graph._find_output_nodes()
    assert output_nodes == [1]
    graph.discover_input_and_output_nodes()
    x = np.random.randn(num_samples, input_dim)
    dgn_output = graph.forward(x)
    classic_out = classic_net_predict(
        [lin.weight, lin2.weight], [lin.bias, lin2.bias], x.copy()
    )
    assert np.allclose(dgn_output, classic_out)

    # test the backpropagation
    expected_out = np.random.randn(num_samples, 2)
    error_gradient = QuadraticCost().backward(dgn_output, expected_out)
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


def test_graph_optimize_simple_pipeline():
    # whole graph output should be (single str) code for it's optimized version
    input_dim = 5
    output_dim = 3
    lin = Linear(input_dim, output_dim)
    pipenode = PipelineNode([lin, Sigmoid()])
    graph = DirectedGraphModel()
    graph.add_node(pipenode)
    code_dict = graph.optimize()
    target = {
        "__imports__": [],
        "__init__": {
            "args": ["__model0_node0_step0_INPUT_DIM", "__model0_node0_step0_OUTPUT_DIM"],
            "body": ["self.__model0_node0_step0_bias = np.random.randn(1, __model0_node0_step0_OUTPUT_DIM)",
                     "self.__model0_node0_step0_weight = np.random.randn(__model0_node0_step0_INPUT_DIM, __model0_node0_step0_OUTPUT_DIM) * np.sqrt( 1 / (__model0_node0_step0_INPUT_DIM + __model0_node0_step0_OUTPUT_DIM) )",
                     "self.__model0_node0_step0_input_shape = ('BATCH_N', __model0_node0_step0_INPUT_DIM)"],
            "return": [],
        },
        "forward": {
            "args": ["inputs"],
            "body": ["self.__model0_first_in = inputs",
                     "self.__model0_node0_step0_out0 = np.dot(self.__model0_first_in, self.__model0_node0_step0_weight) + self.__model0_node0_step0_bias"],
            "return": ["1 / (1 + np.exp(-self.__model0_node0_step0_out0))"],
        },
        "backward": {
            "args": ["error_gradient"],
            "body": ["forward_output = 1 / (1 + np.exp(-self.__model0_node0_step0_out0))",
                     "__model0_node0_step1_gradient = error_gradient * forward_output * (1 - forward_output)",
                     "bias_gradient = __model0_node0_step1_gradient",
                     "weight_gradient = np.dot(self.__model0_first_in.T, bias_gradient)",
                     "__model0_node0_update = (bias_gradient, weight_gradient)",
                     "__model0_node0_gradient = np.dot(bias_gradient, self.__model0_node0_step0_weight.T)"],
            "return": ["[__model0_node0_update]", "[__model0_node0_gradient]"],
        },
    }
    assert code_dict["__init__"] == target["__init__"]
    assert code_dict["forward"] == target["forward"]
    assert code_dict["backward"] == target["backward"]

    code = graph.compile_optimized()
    assert code == '''import cupy as np
class DGM:
    def __init__(self, __model0_node0_step0_INPUT_DIM, __model0_node0_step0_OUTPUT_DIM):
        self.__model0_node0_step0_bias = np.random.randn(1, __model0_node0_step0_OUTPUT_DIM)
        self.__model0_node0_step0_weight = np.random.randn(__model0_node0_step0_INPUT_DIM, __model0_node0_step0_OUTPUT_DIM) * np.sqrt( 1 / (__model0_node0_step0_INPUT_DIM + __model0_node0_step0_OUTPUT_DIM) )
        self.__model0_node0_step0_input_shape = ('BATCH_N', __model0_node0_step0_INPUT_DIM)
    def forward(self, inputs):
        self.__model0_first_in = inputs
        self.__model0_node0_step0_out0 = np.dot(self.__model0_first_in, self.__model0_node0_step0_weight) + self.__model0_node0_step0_bias
        return 1 / (1 + np.exp(-self.__model0_node0_step0_out0))
    def backward(self, error_gradient):
        forward_output = 1 / (1 + np.exp(-self.__model0_node0_step0_out0))
        __model0_node0_step1_gradient = error_gradient * forward_output * (1 - forward_output)
        bias_gradient = __model0_node0_step1_gradient
        weight_gradient = np.dot(self.__model0_first_in.T, bias_gradient)
        __model0_node0_update = (bias_gradient, weight_gradient)
        __model0_node0_gradient = np.dot(bias_gradient, self.__model0_node0_step0_weight.T)
        return [__model0_node0_update], [__model0_node0_gradient]
'''

    # ensure that forward & backward results are the same
    compiled_net = compile(code, "__autogenerated__", "exec", dont_inherit=1)
    exec(compiled_net, globals())

    x = np.random.randn(4, 5)
    dgm = DGM(input_dim, output_dim)
    graph._graph.nodes[0]['inner']._pipeline[0].bias = dgm._DGM__model0_node0_step0_bias
    graph._graph.nodes[0]['inner']._pipeline[0].weight = dgm._DGM__model0_node0_step0_weight
    dgn_output = graph.forward(x)
    opti_output = dgm.forward(x)
    assert np.allclose(dgn_output, opti_output)


def test_graph_optimize_long_pipeline():
    # whole graph output should be (single str) code for it's optimized version
    input_dim = 5
    output_dim = 4
    second_output_dim = 3
    syn = SynapseSum()
    lin = Linear(input_dim, output_dim)
    lin2 = Linear(output_dim, second_output_dim)
    pipenode = PipelineNode([syn, lin, lin2, Sigmoid()])
    graph = DirectedGraphModel()
    graph.add_node(pipenode)

    # run forward() so that it optimizes away the synapsesum
    x = np.random.randn(4, 5)
    dgn_output = graph.forward(x)

    code = graph.compile_optimized()
    assert code == '''import cupy as np
class DGM:
    def __init__(self, __model0_node0_step1_INPUT_DIM, __model0_node0_step1_OUTPUT_DIM, __model0_node0_step2_INPUT_DIM, __model0_node0_step2_OUTPUT_DIM):
        self.__model0_node0_step1_bias = np.random.randn(1, __model0_node0_step1_OUTPUT_DIM)
        self.__model0_node0_step1_weight = np.random.randn(__model0_node0_step1_INPUT_DIM, __model0_node0_step1_OUTPUT_DIM) * np.sqrt( 1 / (__model0_node0_step1_INPUT_DIM + __model0_node0_step1_OUTPUT_DIM) )
        self.__model0_node0_step1_input_shape = ('BATCH_N', __model0_node0_step1_INPUT_DIM)
        self.__model0_node0_step2_bias = np.random.randn(1, __model0_node0_step2_OUTPUT_DIM)
        self.__model0_node0_step2_weight = np.random.randn(__model0_node0_step2_INPUT_DIM, __model0_node0_step2_OUTPUT_DIM) * np.sqrt( 1 / (__model0_node0_step2_INPUT_DIM + __model0_node0_step2_OUTPUT_DIM) )
        self.__model0_node0_step2_input_shape = ('BATCH_N', __model0_node0_step2_INPUT_DIM)
    def forward(self, inputs):
        self.__model0_first_in = inputs
        self.__model0_node0_step0_out0 = self.__model0_first_in
        self.__model0_node0_step1_out0 = np.dot(self.__model0_node0_step0_out0, self.__model0_node0_step1_weight) + self.__model0_node0_step1_bias
        self.__model0_node0_step2_out0 = np.dot(self.__model0_node0_step1_out0, self.__model0_node0_step2_weight) + self.__model0_node0_step2_bias
        return 1 / (1 + np.exp(-self.__model0_node0_step2_out0))
    def backward(self, error_gradient):
        forward_output = 1 / (1 + np.exp(-self.__model0_node0_step2_out0))
        __model0_node0_step3_gradient = error_gradient * forward_output * (1 - forward_output)
        bias_gradient = __model0_node0_step3_gradient
        weight_gradient = np.dot(self.__model0_node0_step1_out0.T, bias_gradient)
        __model0_node0_step2_update = (bias_gradient, weight_gradient)
        __model0_node0_step2_gradient = np.dot(bias_gradient, self.__model0_node0_step2_weight.T)
        bias_gradient = __model0_node0_step2_gradient
        weight_gradient = np.dot(self.__model0_node0_step0_out0.T, bias_gradient)
        __model0_node0_step1_update = (bias_gradient, weight_gradient)
        __model0_node0_step1_gradient = np.dot(bias_gradient, self.__model0_node0_step1_weight.T)
        __model0_node0_gradient = __model0_node0_step1_gradient
        return [__model0_node0_step1_update, __model0_node0_step2_update], [__model0_node0_gradient]
'''

    compiled_net = compile(code, "__autogenerated__", "exec", dont_inherit=1)
    exec(compiled_net, globals())

    # ensure that forward & backward results are the same
    dgm = DGM(input_dim, output_dim, output_dim, second_output_dim)
    dgm._DGM__model0_node0_step1_bias = graph._graph.nodes[0]['inner']._pipeline[1].bias
    dgm._DGM__model0_node0_step1_weight = graph._graph.nodes[0]['inner']._pipeline[1].weight
    dgm._DGM__model0_node0_step2_bias = graph._graph.nodes[0]['inner']._pipeline[2].bias
    dgm._DGM__model0_node0_step2_weight = graph._graph.nodes[0]['inner']._pipeline[2].weight
    opti_output = dgm.forward(x)
    assert np.allclose(dgn_output, opti_output)


def test_graph_optimize_2_pipelines():
    # whole graph output should be (single str) code for it's optimized version
    input_dim = 5
    output_dim = 4
    second_output_dim = 3
    lin = Linear(input_dim, output_dim)
    lin2 = Linear(output_dim, second_output_dim)
    pipenode = PipelineNode([SynapseSum(), lin, Sigmoid()])
    pipenode2 = PipelineNode([SynapseSum(), lin2, Sigmoid()])
    graph = DirectedGraphModel()
    graph.add_node(pipenode, True, False)
    graph.add_node(pipenode2, False, True)
    graph.add_edge(0, 1)

    # run forward() so that it optimizes away the synapsesum
    x = np.random.randn(4, 5)
    dgn_output = graph.forward(x)

    code = graph.compile_optimized()
    assert code == '''import cupy as np
class DGM:
    def __init__(self, __model0_node0_step1_INPUT_DIM, __model0_node0_step1_OUTPUT_DIM, __model0_node0_step2_INPUT_DIM, __model0_node0_step2_OUTPUT_DIM):
        self.__model0_node0_step1_bias = np.random.randn(1, __model0_node0_step1_OUTPUT_DIM)
        self.__model0_node0_step1_weight = np.random.randn(__model0_node0_step1_INPUT_DIM, __model0_node0_step1_OUTPUT_DIM) * np.sqrt( 1 / (__model0_node0_step1_INPUT_DIM + __model0_node0_step1_OUTPUT_DIM) )
        self.__model0_node0_step1_input_shape = ('BATCH_N', __model0_node0_step1_INPUT_DIM)
        self.__model0_node0_step2_bias = np.random.randn(1, __model0_node0_step2_OUTPUT_DIM)
        self.__model0_node0_step2_weight = np.random.randn(__model0_node0_step2_INPUT_DIM, __model0_node0_step2_OUTPUT_DIM) * np.sqrt( 1 / (__model0_node0_step2_INPUT_DIM + __model0_node0_step2_OUTPUT_DIM) )
        self.__model0_node0_step2_input_shape = ('BATCH_N', __model0_node0_step2_INPUT_DIM)
    def forward(self, inputs):
        self.__model0_first_in = inputs
        self.__model0_node0_step0_out0 = self.__model0_first_in
        self.__model0_node0_step1_out0 = np.dot(self.__model0_node0_step0_out0, self.__model0_node0_step1_weight) + self.__model0_node0_step1_bias
        self.__model0_node0_step2_out0 = np.dot(self.__model0_node0_step1_out0, self.__model0_node0_step2_weight) + self.__model0_node0_step2_bias
        return 1 / (1 + np.exp(-self.__model0_node0_step2_out0))
    def backward(self, error_gradient):
        forward_output = 1 / (1 + np.exp(-self.__model0_node0_step2_out0))
        __model0_node0_step3_gradient = error_gradient * forward_output * (1 - forward_output)
        bias_gradient = __model0_node0_step3_gradient
        weight_gradient = np.dot(self.__model0_node0_step1_out0.T, bias_gradient)
        __model0_node0_step2_update = (bias_gradient, weight_gradient)
        __model0_node0_step2_gradient = np.dot(bias_gradient, self.__model0_node0_step2_weight.T)
        bias_gradient = __model0_node0_step2_gradient
        weight_gradient = np.dot(self.__model0_node0_step0_out0.T, bias_gradient)
        __model0_node0_step1_update = (bias_gradient, weight_gradient)
        __model0_node0_step1_gradient = np.dot(bias_gradient, self.__model0_node0_step1_weight.T)
        __model0_node0_gradient = __model0_node0_step1_gradient
        return [__model0_node0_step1_update, __model0_node0_step2_update], [__model0_node0_gradient]
'''

    compiled_net = compile(code, "__autogenerated__", "exec", dont_inherit=1)
    exec(compiled_net, globals())

    # ensure that forward & backward results are the same
    dgm = DGM(input_dim, output_dim, output_dim, second_output_dim)
    dgm._DGM__model0_node0_step1_bias = graph._graph.nodes[0]['inner']._pipeline[1].bias
    dgm._DGM__model0_node0_step1_weight = graph._graph.nodes[0]['inner']._pipeline[1].weight
    dgm._DGM__model0_node0_step2_bias = graph._graph.nodes[0]['inner']._pipeline[2].bias
    dgm._DGM__model0_node0_step2_weight = graph._graph.nodes[0]['inner']._pipeline[2].weight
    opti_output = dgm.forward(x)
    assert np.allclose(dgn_output, opti_output)
