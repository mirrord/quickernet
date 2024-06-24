# import numpy as np
# import cupy as np
from typing import List


class NodeFeedException(Exception):
    pass


class PipelineNode:
    def __init__(self, pipeline: List):
        self._pipeline = pipeline
        self._history = []

    def forward(self, inputs):
        self._history = [inputs]
        for func in self._pipeline:
            staged_output = func(inputs)
            self._history.append(staged_output)
        return self._history.pop()

    def backward(self, error_gradient):
        last_update, staged_error_gradient = self._pipeline[-1].backwards(
            error_gradient, self._history[-1])
        updates = [last_update]
        for idx, func in enumerate(reversed(self._pipeline[:-1])):
            staged_update, staged_error_gradient = func.backwards(
                staged_error_gradient, self._history[-idx - 2])
            updates.insert(0, staged_update)
        return updates, staged_error_gradient

    def update(self, updates: List):
        for idx, func in enumerate(self._pipeline):
            func.update(updates[idx])
