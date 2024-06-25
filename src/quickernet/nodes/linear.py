import cupy as np
from .utils import OptimizableFunction


class Linear(OptimizableFunction):
    # TODO: implement other initialization methods (Xavier, He, etc.)
    def __init__(self, input_dim, output_dim):
        self.weight = np.random.randn(input_dim, output_dim)
        self.bias = np.random.randn(1, output_dim)

    def __call__(self, inputs):
        return np.dot(inputs, self.weight) + self.bias
    
    def __str__(self):
        print(self.weight, self.bias)
        return f"<{self.__class__.__name__}: ({self.weight.shape}, {self.bias.shape})>"

    def backwards(self, error_gradient, inputs):
        bias_gradient = error_gradient
        weight_gradient = np.dot(inputs.T, bias_gradient)
        return (bias_gradient, weight_gradient), np.dot(bias_gradient, self.weight.T)

    def update(self, updates):
        self.bias -= updates[0]
        self.weight -= updates[1]


def list_except(lst, idx):
    return lst[:idx] + lst[idx + 1:]


# NOTE: this function requires multiple inputs, i.e. inputs must be a list.
# As a result, it won't work with the current implementation of PipelineNode.
# Should unary and n-ary functions be treated differently?
class SelfLinear(OptimizableFunction):
    def __call__(self, inputs):
        return np.matmul(*inputs)

    def backwards(self, error_gradient, inputs):
        return None, [np.matmul(*list_except(inputs, idx)) * error_gradient for idx, _ in enumerate(inputs)]

    def update(self, updates):
        pass
