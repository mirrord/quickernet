
from typing import List
from .utils import NodeFunction


class SynapseFunction(NodeFunction):
    input_shape = None

    def forward(self, inputs_l: List):
        return inputs_l

    def backwards(self, gradient, inputs):
        return None, gradient


class SynapseSum(SynapseFunction):
    def forward(self, inputs_l: List):
        return sum(inputs_l)

    def backwards(self, gradient, inputs):
        return None, gradient
