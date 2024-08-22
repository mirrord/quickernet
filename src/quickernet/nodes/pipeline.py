# import numpy as np
from cupy import ndarray
import cupy as np
from typing import List, Tuple, Dict
from .optimization import OptimizableFunction, glue_optimizations  # , dprint
from .node import NodeFeedException, NodeFunction, SynapticSignal, ErrorSignal
from .synapses import SynapseSelect

# TODO: upgrade synapses to separate from the pipeline
# synapses should produce a list of ndarrays which should be processed in parallel.
# They'll also have to save inputs that way (or will the graph do that?)


class PipelineNode(NodeFunction):
    # TODO: check for output shape mismatch/other stuff
    def __init__(
        self,
        pipeline: List[OptimizableFunction],
        synapse: OptimizableFunction = SynapseSelect(),
    ):
        self._pipeline = pipeline
        self._synapse = synapse
        self._history = {}
        self._channels = (
            self._pipeline[0].num_channels()[0],
            self._pipeline[-1].num_channels()[1],
        )
        self._num_updates = []

    def __str__(self):
        return f"<{self.__class__.__name__}: {[str(p) for p in self._pipeline]}>"

    def clear(self):
        self._history = {}

    def _forward(self, inputs: ndarray) -> List[ndarray]:
        history = [inputs]
        for fidx, func in enumerate(self._pipeline):
            try:
                history.append(func(history[-1]))
            except ValueError as e:
                if "Axis dimension mismatch" in str(e):
                    raise NodeFeedException(
                        f"Error in {self.__class__.__name__}::{func.__class__.__name__}: step {fidx} expected shape {func.input_shape}, got {self._history[-1].shape}"
                    )
                raise NodeFeedException(f"Error in {func.__class__.__name__}: {e}")
        history.pop(0)  # forget the first one, the graph stores that
        return history

    def forward(self, inputs: SynapticSignal) -> SynapticSignal:
        self.clear()
        staged_input = self._synapse.forward(inputs)
        for idx, item in enumerate(staged_input):
            self._history[idx] = self._forward(item)
        self._channels = (len(staged_input), len(self._history))
        return {i: self._history[i].pop(-1) for i in range(len(self._history))}

    def _backward(
        self, error_gradient: ndarray, last_recorded_input: List[ndarray]
    ) -> Tuple[dict, ndarray]:
        staged_error_gradient = error_gradient
        updates = {}
        last_stage_idx = len(self._pipeline) - 1
        for nidx, func in enumerate(reversed(self._pipeline)):
            idx = last_stage_idx - nidx
            staged_update, staged_error_gradient = func.backward(
                staged_error_gradient, last_recorded_input[idx]
            )
            if staged_update is not None:
                updates[idx] = staged_update
        return updates, staged_error_gradient

    # list of runs (channels) [ list of stages [ list of updates ] ]
    def _merge_updates(
        self, update_dict: Dict[int, Dict[int, List[ndarray]]]
    ) -> Dict[int, List[ndarray]]:
        num_stages = len(self._pipeline)
        merged_updates = {}
        for stage_idx in range(num_stages):
            stage_updates = []
            for step in range(len(update_dict[0].get(stage_idx, []))):
                step_update = np.stack(
                    [
                        update_dict[channel][stage_idx][step]
                        for channel in update_dict.keys()
                    ]
                ).sum(axis=0)
                stage_updates.append(step_update)
            merged_updates[stage_idx] = stage_updates
        return merged_updates

    def backward(
        self, error_gradient: SynapticSignal, last_recorded_input: SynapticSignal
    ) -> Tuple[dict, ErrorSignal]:
        updates = {}
        gradients = {}
        first_stage_input_history = self._synapse.forward(last_recorded_input)
        # TODO: does this mutate the history? fix if so
        for channel, input_history in self._history.items():
            pad_history = [first_stage_input_history[channel]] + input_history
            updates[channel], error_gradient = self._backward(
                error_gradient[channel], pad_history
            )
            gradients[channel] = error_gradient
        return self._merge_updates(updates), gradients

    def update(self, updates: dict, learning_rate: float):
        for idx, func in enumerate(self._pipeline):
            func.update(updates[idx], learning_rate)

    def optimize(
        self,
        var_replaces: dict,
        rep_idx: int = 0,
        prefix="__node",
        freeze_inits=False,
        freeze_params=False,
    ) -> Tuple[list, str, list]:
        my_prefix = f"{prefix}{rep_idx}_step"
        my_desc = (
            {}
            if not getattr(self._pipeline[0], "__used", True)
            else self._pipeline[0].optimize(
                var_replaces,
                0,
                my_prefix,
                freeze_inits=freeze_inits,
                freeze_params=freeze_params,
            )
        )
        # dprint(my_desc)
        for idx, func in enumerate(self._pipeline[1:], start=1):
            # TODO: fix for multiple inputs
            var_replaces["inputs"] = f"self.{my_prefix}{idx - 1}_out0"
            var_replaces["last_recorded_input"] = f"self.{my_prefix}{idx - 1}_out0"
            desc = func.optimize(
                var_replaces,
                idx,
                my_prefix,
                freeze_inits=freeze_inits,
                freeze_params=freeze_params,
            )
            # print(f"glueing with: {func.__class__.__name__}")
            # dprint(desc)
            my_desc = glue_optimizations(my_desc, desc, var_replaces, idx, my_prefix)
            # dprint(my_desc)
        return my_desc
