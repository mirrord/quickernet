
from typing import Any
from inspect import getsourcelines


class OptimizableFunction:
    def __call__(self, inputs: Any):
        return None

    # this is a good enough general case, but sometimes there are better ways
    # returns a tuple of (source, return statement)
    def optimize(self):
        source_lines = getsourcelines(self.__call__)[0]
        return (''.join(source_lines[:-1]), source_lines[-1])
