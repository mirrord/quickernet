# import numpy as np
# import cupy as np
from typing import List, Tuple
from .utils import OptimizableFunction, NodeFunction


class NodeFeedException(Exception):
    pass


class PipelineNode(NodeFunction):
    def __init__(self, pipeline: List[OptimizableFunction]):
        self._pipeline = pipeline
        self._history = []
        self.last_output = None
        self.input_shape = None
        for obj in pipeline:
            if obj.input_shape:
                self.input_shape = obj.input_shape
                break
        if self.input_shape is None:
            raise NodeFeedException("No input shape found in pipeline")

    def __str__(self):
        return f"<{self.__class__.__name__}: {[str(p) for p in self._pipeline]}>"

    def clear(self):
        self._history = []
        self.last_output = None

    def forward(self, inputs):
        self._history = [inputs]
        for func in self._pipeline:
            staged_output = func(self._history[-1])
            self._history.append(staged_output)
        self.last_output = self._history.pop()
        return self.last_output

    def backward(self, error_gradient):
        staged_error_gradient = error_gradient
        updates = []
        for idx, func in enumerate(reversed(self._pipeline)):
            staged_update, staged_error_gradient = func.backward(
                staged_error_gradient, self._history[-idx - 1])
            updates.insert(0, staged_update)
        return updates, staged_error_gradient

    def update(self, updates: List, learning_rate: float):
        for idx, func in enumerate(self._pipeline):
            func.update(updates[idx], learning_rate)

    def optimize(self) -> Tuple[list, str, list]:
        my_inputs, lines, staged_outputs = self._pipeline[0].optimize()
        opt_lines = [lines]
        for func in self._pipeline[1:]:
            params, lines, next_output = func.optimize()
            for inarg, outarg in zip(staged_outputs, params):
                opt_lines.append(f"\t{outarg} = {inarg}")
            staged_outputs = next_output
            opt_lines.append(lines)
        return my_inputs, ''.join(opt_lines), staged_outputs
