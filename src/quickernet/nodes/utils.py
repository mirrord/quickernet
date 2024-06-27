
from typing import Any, Tuple
from inspect import getsourcelines, signature


def params_only(return_statement: str) -> list:
    objs = []
    innercount = 0
    begindx = 0
    return_list_str = return_statement.split('return ')[1]
    for idx, c in enumerate(return_list_str):
        if c == '(':
            innercount += 1
        elif c == ')':
            innercount -= 1
        elif c == ',' and innercount == 0:
            objs.append(return_list_str[begindx:idx].strip())
            begindx = idx + 1
    objs.append(return_list_str[begindx:].strip())
    return objs


def list_except(lst, idx):
    return lst[:idx] + lst[idx + 1:]


class OptimizableFunction:
    def __call__(self, inputs: Any):
        return self.forward(inputs)

    def forward(self, inputs: Any):
        return inputs

    def backward(self, error_gradient, inputs: Any) -> Tuple[Any, Any]:
        return None, error_gradient

    # this is a good enough general case, but sometimes there are better ways
    # returns a tuple of (inputs, source, return statements)
    def optimize(self, backwards=False) -> Tuple[list, str, list]:
        f = self.backward if backwards else self.forward
        source_lines = getsourcelines(f)[0]
        return list(signature(self.__call__).parameters.keys()), \
            ''.join(source_lines[1:-1]), \
            params_only(source_lines[-1])


class NodeFunction(OptimizableFunction):
    input_shape = None
