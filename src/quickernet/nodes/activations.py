
import cupy as np
from .utils import NodeFunction


class NoActivation(NodeFunction):
    pass


class Sigmoid(NodeFunction):
    def forward(self, inputs):
        return 1 / (1 + np.exp(-inputs))

    def backward(self, error_gradient, last_recorded_input):
        forward_output = 1 / (1 + np.exp(-last_recorded_input))
        return None, error_gradient * forward_output * (1 - forward_output)


class ReLU(NodeFunction):
    def forward(self, inputs):
        return np.maximum(0, inputs)

    def backward(self, error_gradient, last_recorded_input):
        return None, error_gradient * (last_recorded_input > 0)


class Softmax(NodeFunction):
    def forward(self, inputs):
        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def backward(self, error_gradient, last_recorded_input):
        exp = np.exp(last_recorded_input - np.max(last_recorded_input, axis=1, keepdims=True))
        forward_output = exp / np.sum(exp, axis=1, keepdims=True)
        return None, error_gradient * forward_output * (1 - forward_output)
