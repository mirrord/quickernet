from typing import Dict
from .node import NodeFunction

# TODO: refactor to PipelineFunctions instead of NodeFunctions
# Node classes should (mostly) be aliases for PipelineNodes
# TODO: refactor synapse functions entirely


# Synapse functions take a Dict[list[ndarray]] and return a list[ndarray]


class SynapseFunction(NodeFunction):
    input_shape = None

    def __init__(self):
        super().__init__()

    def forward(self, inputs_d: Dict):
        return list(inputs_d.values())

    def backward(self, error_gradient, last_recorded_input):
        return None, error_gradient


class SynapseSelect(SynapseFunction):
    def forward(self, inputs: Dict):
        return inputs[0]


class SynapseSum(SynapseFunction):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: Dict):
        return [sum(inputs[k]) for k in inputs]

    def backward(self, error_gradient, last_recorded_input):
        return None, error_gradient

    def optimize(
        self,
        var_replaces: Dict,
        rep_idx: int = 0,
        prefix="__node",
        freeze_inits=False,
        freeze_params=False,
    ):
        return {
            k: v
            for k, v in super()
            .optimize(var_replaces, rep_idx, prefix, freeze_inits, freeze_params)
            .items()
            if k != "__init__"
        }
