# import numpy as np
# import cupy as np
from typing import Tuple, Dict, List
from cupy import ndarray
from .optimization import OptimizableFunction

ChannelSignal = List[ndarray]
SynapticSignal = Dict[int, ChannelSignal]
PipelineUpdateSignal = List[ndarray]
ErrorSignal = Dict[int, ndarray]


class NodeFeedException(Exception):
    pass


class NodeFunction(OptimizableFunction):
    def __init__(self):
        super().__init__()

    def num_channels(self):
        return (1, 1)

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs: SynapticSignal) -> SynapticSignal:
        return inputs

    def backward(
        self, error_gradient: ErrorSignal, last_recorded_input: SynapticSignal
    ) -> Tuple[dict, ErrorSignal]:
        return None, error_gradient

    def update(self, updates: dict, learning_rate: float):
        pass


class PipelineFunction(OptimizableFunction):
    def __init__(self):
        super().__init__()

    def num_channels(self):
        return (1, 1)

    def __call__(self, inputs: ndarray):
        return self.forward(inputs)

    def forward(self, inputs: ndarray) -> ndarray:
        return inputs

    def backward(
        self, error_gradient: ndarray, last_recorded_input: ndarray
    ) -> Tuple[list, ndarray]:
        return None, error_gradient

    def update(self, updates: list, learning_rate: float):
        pass
