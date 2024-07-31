# import numpy as np
# import cupy as np
from .node import OptimizableFunction


class NodeFeedException(Exception):
    pass


class NodeFunction(OptimizableFunction):
    input_shape = None

    def __init__(self):
        super().__init__()

    # TODO: implement optimize method to account for standardize_input method
    def __call__(self, inputs):
        return self.forward(self.standardize_input(inputs))

    def standardize_input(self, inputs):
        return inputs

    def update(self, updates, learning_rate):
        pass
