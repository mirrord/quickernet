# import numpy as np
# import cupy as np
from typing import List, Tuple
from .utils import OptimizableFunction, NodeFunction, glue_optimizations  # , dprint


class NodeFeedException(Exception):
    pass


class PipelineNode(NodeFunction):
    # TODO: check for output shape mismatch/other stuff
    def __init__(self, pipeline: List[OptimizableFunction]):
        self._pipeline = pipeline
        self._history = []
        self.last_output = None
        self.input_shape = None
        for obj in pipeline:
            self.input_shape = getattr(obj, "input_shape", None)
            if self.input_shape is not None:
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
        for fidx, func in enumerate(self._pipeline):
            try:
                staged_output = func(self._history[-1])
            except ValueError as e:
                if "Axis dimension mismatch" in str(e):
                    raise NodeFeedException(f"Error in {self.__class__.__name__}::{func.__class__.__name__}: step {fidx} expected shape {func.input_shape}, got {self._history[-1].shape}")
                raise NodeFeedException(f"Error in {func.__class__.__name__}: {e}")
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

    def optimize(self, var_replaces: dict, rep_idx: int = 0, prefix="__node", freeze_inits=False, freeze_params=False) -> Tuple[list, str, list]:
        my_prefix = f"{prefix}{rep_idx}_step"
        my_desc = {} if not getattr(self._pipeline[0], "__used", True) else self._pipeline[0].optimize(var_replaces, 0, my_prefix, freeze_inits=freeze_inits, freeze_params=freeze_params)
        # dprint(my_desc)
        for idx, func in enumerate(self._pipeline[1:], start=1):
            # TODO: fix for multiple inputs
            var_replaces["inputs"] = f"self.{my_prefix}{idx - 1}_out0"
            var_replaces["last_recorded_input"] = f"self.{my_prefix}{idx - 1}_out0"
            desc = func.optimize(var_replaces, idx, my_prefix, freeze_inits=freeze_inits, freeze_params=freeze_params)
            # print(f"glueing with: {func.__class__.__name__}")
            # dprint(desc)
            my_desc = glue_optimizations(my_desc, desc, var_replaces, idx, my_prefix)
            # dprint(my_desc)
        return my_desc
