
from typing import List
from .utils import NodeFunction


class SynapseFunction(NodeFunction):
    input_shape = None

    def __init__(self):
        super().__init__()

    def forward(self, inputs_l: List):
        return inputs_l

    def backward(self, error_gradient, last_recorded_input):
        return None, error_gradient


class SynapseSum(SynapseFunction):
    def __init__(self):
        super().__init__()
        self.__used = False

    def standardize_input(self, inputs):
        if isinstance(inputs, list):
            self.__used = len(inputs) > 1
            return inputs
        self.__used = False
        return [inputs]

    def forward(self, inputs: List):
        return sum(inputs)

    def backward(self, error_gradient, last_recorded_input):
        return None, error_gradient

    def optimize(self, var_replaces: dict, rep_idx: int = 0, prefix="__node", freeze_inits=False, freeze_params=False):
        if not self.__used:
            return {"forward": {"args": ["inputs"], "body": [], "return": [var_replaces.get("inputs", "inputs")]},
                    "backward": {"args": ["error_gradient", "last_recorded_input"], "body": [], "return": ["None", "error_gradient"]}}
        return {k: v for k, v in super().optimize(var_replaces, rep_idx, prefix, freeze_inits, freeze_params).items() if k != "__init__"}
