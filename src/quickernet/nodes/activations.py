
from .utils import OptimizableFunction


class NoActivation(OptimizableFunction):
    def __call__(self, inputs):
        return inputs

    def backwards(self, inputs):
        return 1
