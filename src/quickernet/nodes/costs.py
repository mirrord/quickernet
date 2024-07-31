import cupy as np

from .optimization import OptimizableFunction


class CostFunction(OptimizableFunction):
    def __call__(self, inputs, expected_output):
        return self.forward(inputs, expected_output)

    def forward(self, inputs, expected_output):
        raise NotImplementedError

    def backward(self, error_gradient, inputs, expected_output):
        raise NotImplementedError


class QuadraticCost(CostFunction):
    def forward(self, inputs, expected_output):
        return (np.sum(np.square(inputs - expected_output))).item() / inputs.shape[0]

    def backward(self, inputs, expected_output):
        return (inputs - expected_output) * 2
