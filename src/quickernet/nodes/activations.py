
import cupy as np
from .utils import NodeFunction


class NoActivation(NodeFunction):
    pass


class Sigmoid(NodeFunction):
    def forward(self, inputs):
        return 1 / (1 + np.exp(-inputs))

    def backwards(self, error_gradient, inputs):
        forward_output = self(inputs)
        return None, error_gradient * forward_output * (1 - forward_output)
