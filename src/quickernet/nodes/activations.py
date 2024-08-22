import cupy as np
from .node import NodeFunction


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
        # return np.greater(x, 0).astype('float64')


class LeakyReLU(NodeFunction):
    def forward(self, inputs):
        return np.maximum(0.01 * inputs, inputs)

    def backward(self, error_gradient, last_recorded_input):
        return None, error_gradient * np.where(last_recorded_input > 0, 1, 0.01)


class Softmax(NodeFunction):
    def forward(self, inputs):
        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def backward(self, error_gradient, last_recorded_input):
        exp = np.exp(
            last_recorded_input - np.max(last_recorded_input, axis=1, keepdims=True)
        )
        forward_output = exp / np.sum(exp, axis=1, keepdims=True)
        return None, error_gradient * forward_output * (1 - forward_output)


class Tanh(NodeFunction):
    def forward(self, inputs):
        return np.tanh(inputs)

    def backward(self, error_gradient, last_recorded_input):
        t = np.tanh(last_recorded_input)
        return None, error_gradient * (1 - (t * t))


class Swish(NodeFunction):
    def forward(self, inputs):
        return inputs * (1 / (1 + np.exp(-inputs)))

    def backward(self, error_gradient, last_recorded_input):
        sigx = 1 / (1 + np.exp(-last_recorded_input))
        swishx = last_recorded_input * sigx
        return None, error_gradient * (swishx + (sigx * (1 - swishx)))


class Scale(NodeFunction):
    def forward(self, inputs):
        return inputs / inputs.shape[1]

    def backward(self, error_gradient, last_recorded_input):
        return 1 / last_recorded_input.shape[1]


class Norm(NodeFunction):
    def forward(self, inputs):
        return inputs / np.max(inputs, axis=1, keepdims=True)

    def backward(self, error_gradient, last_recorded_input):
        return 1 / np.max(last_recorded_input, axis=1, keepdims=True)


class L2Norm(NodeFunction):
    def forward(self, inputs):
        return inputs / np.linalg.norm(inputs, axis=1, keepdims=True)

    def backward(self, error_gradient, last_recorded_input):
        return 1 / np.linalg.norm(last_recorded_input, axis=1, keepdims=True)
