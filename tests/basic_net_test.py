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


def test_linear_optimize():
    # test the linear node
    input_dim = 5
    output_dim = 3
    lin = Linear(input_dim, output_dim)
    code = lin.optimize({})
    target = {
        "__imports__": [],
        "__init__": {
            "args": ["input_dim", "output_dim"],
            "body": ["self.__node0_bias = np.random.randn(1, output_dim)", "self.__node0_weight = np.random.randn(input_dim, output_dim) * np.sqrt(", "1 / (input_dim + output_dim)", ")", "self.__node0_input_shape = ('BATCH_N', input_dim)"],
            "return": [],
        },
        "forward": {
            "args": ["inputs"],
            "body": [],
            "return": ["np.dot(inputs, self.__node0_weight) + self.__node0_bias"],
        },
        "backward": {
            "args": ["error_gradient", "last_recorded_input"],
            "body": ["bias_gradient = error_gradient", "weight_gradient = np.dot(last_recorded_input.T, bias_gradient)"],
            "return": ["(bias_gradient, weight_gradient)", "np.dot(bias_gradient, self.__node0_weight.T)"],
        },
    }
    assert code["__init__"] == target["__init__"]
    assert code["forward"] == target["forward"]
    assert code["backward"] == target["backward"]


def test_sigmoid_optimize():
    # test activation function
    sig = Sigmoid()
    code = sig.optimize({})
    target = {
        "__imports__": [],
        "__init__": {},
        "forward": {
            "args": ["inputs"],
            "body": [],
            "return": ["1 / (1 + np.exp(-inputs))"],
        },
        "backward": {
            "args": ["error_gradient", "last_recorded_input"],
            "body": ["forward_output = 1 / (1 + np.exp(-last_recorded_input))"],
            "return": ["None", "error_gradient * forward_output * (1 - forward_output)"],
        },
    }
    assert code["forward"] == target["forward"]
    assert code["backward"] == target["backward"]


def test_synapse_optimize():
    syn = SynapseSum()
    code = syn.optimize({})
    target = {
        "forward": {
            "args": ["inputs"],
            "body": [],
            "return": ["sum(inputs)"],
        },
        "backward": {
            "args": ["error_gradient", "last_recorded_input"],
            "body": [],
            "return": ["None", "error_gradient"],
        },
    }
    assert code["forward"] == target["forward"]
    assert code["backward"] == target["backward"]


def test_pipeline_optimize():
    input_dim = 5
    output_dim = 3
    syn = SynapseSum()
    lin = Linear(input_dim, output_dim)
    lin2 = Linear(input_dim, output_dim)
    pipenode = PipelineNode([syn, lin, lin2, Sigmoid()])
    code, _ = pipenode.optimize({})
    target = {
        "__imports__": [],
        "__init__": {
            "args": ["input_dim", "output_dim", "input_dim_1", "output_dim_1"],
            "body": ["self.__node0_step1_bias = np.random.randn(1, output_dim)",
                     "self.__node0_step1_weight = np.random.randn(input_dim, output_dim) * np.sqrt( 1 / (input_dim + output_dim) )",
                     "self.__node0_step1_input_shape = ('BATCH_N', input_dim)",

                     "self.__node0_step2_bias = np.random.randn(1, output_dim_1)",
                     "self.__node0_step2_weight = np.random.randn(input_dim_1, output_dim_1) * np.sqrt( 1 / (input_dim_1 + output_dim_1) )",
                     "self.__node0_step2_input_shape = ('BATCH_N', input_dim_1)"],
            "return": [],
        },
        "forward": {
            "args": ["inputs"],
            "body": ["self.__node0_step0_out0 = sum(inputs)",
                     "self.__node0_step1_out0 = np.dot(self.__node0_step0_out0, self.__node0_step1_weight) + self.__node0_step1_bias",
                     "self.__node0_step2_out0 = np.dot(self.__node0_step1_out0, self.__node0_step2_weight) + self.__node0_step2_bias"],
            "return": ["1 / (1 + np.exp(-self.__node0_step2_out0))"],
        },
        "backward": {
            "args": ["error_gradient"],
            "body": ["forward_output = 1 / (1 + np.exp(-self.__node0_step2_out0))",
                     "__node0_step3_gradient = error_gradient * forward_output * (1 - forward_output)",

                     "bias_gradient = __node0_step3_gradient",
                     "weight_gradient = np.dot(self.__node0_step1_out0.T, bias_gradient)",
                     "__node0_step2_update = (bias_gradient, weight_gradient)",
                     "__node0_step2_gradient = np.dot(bias_gradient, self.__node0_step2_weight.T)",

                     "bias_gradient = __node0_step2_gradient",
                     "weight_gradient = np.dot(self.__node0_step0_out0.T, bias_gradient)",
                     "__node0_step1_update = (bias_gradient, weight_gradient)",
                     "__node0_step1_gradient = np.dot(bias_gradient, self.__node0_step1_weight.T)"],
            "return": ["None", "__node0_step1_gradient"],
        },
    }
    assert code["__init__"] == target["__init__"]
    assert code["forward"] == target["forward"]
    assert code["backward"] == target["backward"]


def test_graph_optimize_simple_pipeline():
    # whole graph output should be (single str) code for it's optimized version
    input_dim = 5
    output_dim = 3
    lin = Linear(input_dim, output_dim)
    pipenode = PipelineNode([lin, Sigmoid()])
    graph = DirectedGraphModel()
    graph.add_node(pipenode)
    code_dict, _ = graph.optimize()
    target = {
        "__imports__": [],
        "__init__": {
            "args": ["input_dim", "output_dim"],
            "body": ["self.__model0_node0_step0_bias = np.random.randn(1, output_dim)",
                     "self.__model0_node0_step0_weight = np.random.randn(input_dim, output_dim) * np.sqrt( 1 / (input_dim + output_dim) )",
                     "self.__model0_node0_step0_input_shape = ('BATCH_N', input_dim)"],
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
    def __init__(self, input_dim, output_dim):
        self.__model0_node0_step0_bias = np.random.randn(1, output_dim)
        self.__model0_node0_step0_weight = np.random.randn(input_dim, output_dim) * np.sqrt( 1 / (input_dim + output_dim) )
        self.__model0_node0_step0_input_shape = ('BATCH_N', input_dim)
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
    output_dim = 3
    syn = SynapseSum()
    lin = Linear(input_dim, output_dim)
    lin2 = Linear(input_dim, output_dim)
    pipenode = PipelineNode([syn, lin, lin2, Sigmoid()])
    graph = DirectedGraphModel()
    graph.add_node(pipenode)
    code = graph.optimize()
    assert code == '''
import cupy as np
class DGM:
    def __init__(self, input_dim, output_dim):
        self.__node0_step0_bias = np.random.randn(1, output_dim)
        self.__node0_step0_weight = np.random.randn(input_dim, output_dim) * np.sqrt(
        1 / (input_dim + output_dim)
        )
        self.input_shape = ('BATCH_N', input_dim)
    def __call__(self, inputs):
        self.__node0_step0_out = np.dot(inputs, self.__node0_step0_weight) + self.__node0_step0_bias
        return 1 / (1 + np.exp(-self.__node0_step0_out))
    def backward(self, error_gradient, inputs):
        forward_output = 1 / (1 + np.exp(-inputs))
        __step0_gradient = error_gradient * forward_output * (1 - forward_output)
        bias_gradient = __step0_gradient
        weight_gradient = np.dot(self.__node0_step0_out.T, bias_gradient)
        return (bias_gradient, weight_gradient), np.dot(bias_gradient, self.__node0_step0_weight.T)
'''

    # ensure that forward & backward results are the same
    x = np.random.randn(4, 5)
    dgn_output = graph.forward(x)
    DGM = exec(code)
    dgm = DGM()
    opti_output = dgm(x)
    assert np.allclose(dgn_output, opti_output)
