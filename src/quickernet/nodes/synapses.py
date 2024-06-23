
from typing import List
from .utils import OptimizableFunction


class SynapseSum(OptimizableFunction):
    def __call__(inputs_l: List):
        return sum(inputs_l)

    def backwards(self, inputs):
        return 1
