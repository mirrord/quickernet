
import cupy as np
from .utils import OptimizableFunction


class NoActivation(OptimizableFunction):
    def __call__(self, inputs):
        return inputs

    def backwards(self, inputs):
        return 1


class Sigmoid(OptimizableFunction):
    def __call__(self, inputs):
        return 1 / (1 + np.exp(-inputs))

    def backwards(self, error_gradient, inputs):
        forward_output = self(inputs)
        return None, error_gradient * forward_output * (1 - forward_output)
