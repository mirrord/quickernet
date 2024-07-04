
import cupy as np
from .utils import NodeFunction


class NoActivation(NodeFunction):
    pass


class Sigmoid(NodeFunction):
    def forward(self, inputs):
        return 1 / (1 + np.exp(-inputs))

    def backward(self, error_gradient, inputs):
        forward_output = self(inputs)
        return None, error_gradient * forward_output * (1 - forward_output)


class ReLU(NodeFunction):
    def forward(self, inputs):
        return np.maximum(0, inputs)

    def backward(self, error_gradient, inputs):
        return None, error_gradient * (inputs > 0)


class Softmax(NodeFunction):
    def forward(self, inputs):
        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def backward(self, error_gradient, inputs):
        forward_output = self(inputs)
        return None, error_gradient * forward_output * (1 - forward_output)
