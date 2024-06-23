# import numpy as np
import cupy as np
from typing import List

from . import synapses, activations, linear


class NodeFeedException(Exception):
    pass


class NeuralNode:
    """first draft neural node class"""
    _linear = None
    _activation = None
    _synapse = None
    _dims = None
    _normalize = None
    _weights = None
    _bias = None

    def __init__(self,
                 linear=linear.MatrixMultiply(),
                 activation=activations.NoActivation(),
                 synapse=synapses.SynapseSum(),
                 dims=None,
                 normalize=None,  # TODO: should this be a function?
                 ):
        self._linear = linear
        self._activation = activation
        self._synapse = synapse
        self._dims = dims
        self._normalize = normalize
        self.clear()

    def __str__(self):
        return self.__class__.__name__

    def set_dims(self, dims):
        self._dims = dims

    def shape(self):
        return (0,)

    def clear(self):
        """Clear the cache, including input, output, and function pointers."""
        self.output = None
        self.inputs = None  # []
        self.linear_output = None

    def get_output(self):
        """Retrieve the last-calculated output of this node, or None if never activated."""
        return self.output

    def does_update(self):
        return True

    def forward(self, inputs: List):
        # not sure yet which input I should keep here
        # self.inputs = inputs
        # merged_inputs = self._synapse(inputs)
        # if self._normalize:
        #     merged_inputs = self._normalize(merged_inputs)
        # self.linear_output = self._linear(merged_inputs)
        self.inputs = self._synapse(inputs)
        if self._normalize:
            self.inputs = self._normalize(self.inputs)
        self.linear_output = self._linear(self.inputs)
        self.output = self._activation(self.linear_output)
        return self.output

    def backward(self, de_dz_foward: np.array):
        delta_bias = de_dz_foward * self._activation.backwards(self.linear_output)
        delta_weight = self._linear.backwards(self.inputs, delta_bias)
        de_dz_next = np.dot(delta_bias, self._weights.T)  # why are we doing this again? Should this be another linear.backwards?
        return (
            delta_bias,
            delta_weight,
            de_dz_next,
        )  # last is de_dz for next layer down

    def update(self, delta_bias: np.array, delta_weight: np.array):
        """Update the weights and biases by subtraction."""
        self._bias -= delta_bias
        self._weights -= delta_weight

    def compile(self):
        return None
