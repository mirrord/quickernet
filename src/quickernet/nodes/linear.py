import cupy as np
from .utils import OptimizableFunction


class MatrixMultiply(OptimizableFunction):
    def __call__(self, inputs):
        return np.matmul(*inputs)

    def backwards(self, inputs, delta_bias):
        return np.dot(inputs.T, delta_bias)
