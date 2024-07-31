# import numpy as np
# import cupy as np
from .optimization import OptimizableFunction


class NodeFeedException(Exception):
    pass


class NodeFunction(OptimizableFunction):
    input_shape = None

    def __init__(self):
        super().__init__()

    def num_sites(self):
        return (1, 1)

    # TODO: implement optimize method to account for standardize_input method
    def __call__(self, inputs):
        return self.forward(self.standardize_input(inputs))

    def standardize_input(self, inputs):
        return inputs

    def forward(self, inputs):
        return inputs

    def backward(self, error_gradient, last_recorded_input):
        return None, error_gradient

    def update(self, updates, learning_rate):
        pass
