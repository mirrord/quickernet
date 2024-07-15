
from typing import List
from .utils import NodeFunction


class SynapseFunction(NodeFunction):
    input_shape = None

    def forward(self, inputs_l: List):
        return inputs_l

    def backward(self, error_gradient, last_recorded_input):
        return None, error_gradient


class SynapseSum(SynapseFunction):
    def standardize_input(self, inputs):
        return inputs if isinstance(inputs, list) else [inputs]

    def forward(self, inputs: List):
        return sum(inputs)

    def backward(self, error_gradient, last_recorded_input):
        return None, error_gradient
