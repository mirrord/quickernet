# import cupy as np
from quickernet.nodes.synapses import SynapseSum
from quickernet.nodes.linear import Linear
from quickernet.nodes.activations import Sigmoid
from quickernet.nodes.node import PipelineNode


def test_linear_optimize():
    # test the linear node
    input_dim = 5
    output_dim = 3
    lin = Linear(input_dim, output_dim)
    code = lin.optimize({})
    target = {
        "__imports__": [],
        "__init__": {
            "args": ["__node0_INPUT_DIM", "__node0_OUTPUT_DIM"],
            "body": ["self.__node0_bias = np.random.randn(1, __node0_OUTPUT_DIM)",
                     "self.__node0_weight = np.random.randn(__node0_INPUT_DIM, __node0_OUTPUT_DIM) * np.sqrt( 1 / (__node0_INPUT_DIM + __node0_OUTPUT_DIM) )",
                     "self.__node0_input_shape = ('BATCH_N', __node0_INPUT_DIM)"],
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
            "return": ["inputs"],
        },
        "backward": {
            "args": ["error_gradient", "last_recorded_input"],
            "body": [],
            "return": ["None", "error_gradient"],
        },
    }
    assert code["forward"] == target["forward"]
    assert code["backward"] == target["backward"]

    syn([1, 2, 3])
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
    code = pipenode.optimize({})
    target = {
        "__imports__": [],
        "__init__": {
            "args": ["__node0_step1_INPUT_DIM", "__node0_step1_OUTPUT_DIM", "__node0_step2_INPUT_DIM", "__node0_step2_OUTPUT_DIM"],
            "body": ["self.__node0_step1_bias = np.random.randn(1, __node0_step1_OUTPUT_DIM)",
                     "self.__node0_step1_weight = np.random.randn(__node0_step1_INPUT_DIM, __node0_step1_OUTPUT_DIM) * np.sqrt( 1 / (__node0_step1_INPUT_DIM + __node0_step1_OUTPUT_DIM) )",
                     "self.__node0_step1_input_shape = ('BATCH_N', __node0_step1_INPUT_DIM)",

                     "self.__node0_step2_bias = np.random.randn(1, __node0_step2_OUTPUT_DIM)",
                     "self.__node0_step2_weight = np.random.randn(__node0_step2_INPUT_DIM, __node0_step2_OUTPUT_DIM) * np.sqrt( 1 / (__node0_step2_INPUT_DIM + __node0_step2_OUTPUT_DIM) )",
                     "self.__node0_step2_input_shape = ('BATCH_N', __node0_step2_INPUT_DIM)"],
            "return": [],
        },
        "forward": {
            "args": ["inputs"],
            "body": ["self.__node0_step0_out0 = inputs",
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
