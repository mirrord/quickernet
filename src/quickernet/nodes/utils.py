
from typing import Any, Tuple
from re import finditer
from inspect import getsourcelines, signature
import cupy as np


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


def replace_all(text: str, replacements: dict) -> str:
    for key, val in replacements.items():
        text = text.replace(key, val)
    return text


def dprint(d: dict, indent=1):
    brace_tabs = '\t' * (indent - 1)
    print(brace_tabs, "{")
    tabs = '\t' * indent
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{tabs}{k}:")
            dprint(v, indent + 1)
        elif isinstance(v, list):
            print(f"{tabs}{k}: [")
            for i in v:
                print(f"{tabs}\t{i}")
            print(f"{tabs}]")
        else:
            print(f"{tabs}{k}: {v}")
    print(brace_tabs, "}")


def get_self_tokens(code_line: str) -> list:
    return [substr.group() for substr in finditer(r"(self\.\w+)", code_line)]


def get_self_token_replacements(code_line: str, rep_idx: int, prefix: str) -> dict:
    replacements = {}
    self_prefix = f'self.{prefix}'
    my_prefix = f"{self_prefix}{rep_idx}_"
    for token in get_self_tokens(code_line):
        if not token.startswith("self.__"):
            new_token = token.replace('self.', my_prefix)
            replacements[token] = new_token
    return replacements


def list_except(lst, idx):
    return lst[:idx] + lst[idx + 1:]


# split function into parameters, body, and return statements
def characterize_function(f, replacements: dict = None, rep_idx: int = 0, prefix: str = "__") -> dict:
    # NOTE: this function does not preserve tabs
    replacements = replacements or {}
    source_lines = getsourcelines(f)[0]
    body_lines = []
    has_return = False
    in_block = False
    for idx, line in enumerate(source_lines[1:], start=1):
        stripped_line = line.strip()
        if stripped_line.startswith('return'):
            has_return = True
            break
        if stripped_line and not stripped_line.startswith('#'):
            replacements.update(get_self_token_replacements(stripped_line, rep_idx, prefix))
            stripped_line = replace_all(stripped_line, replacements)
            if stripped_line in ")]}":
                body_lines[-1] += ' ' + stripped_line
                in_block = False
            elif stripped_line[-1] in "([{":
                in_block = True
                body_lines.append(stripped_line)
            elif in_block:
                body_lines[-1] += ' ' + stripped_line
                if stripped_line[-1] in ")]}":
                    in_block = False
            else:
                body_lines.append(stripped_line)
    return_lines = []
    if has_return:
        full_return_line = ' '.join(source_lines[idx:]).strip().replace('\t', ' ')
        replacements.update(get_self_token_replacements(full_return_line, rep_idx, prefix))
        full_return_line = replace_all(full_return_line, replacements)
        return_lines = params_only(full_return_line)
    return {
        "args": list(signature(f).parameters.keys()),
        "body": body_lines,
        "return": return_lines
    }, replacements


# glue two function descriptions together
def glue_inits(fd1: dict, fd2: dict, variable_replacements: dict, rep_idx=0, prefix="node"):
    # glue parameters together
    params = fd1.get("args", [])
    for p in fd2.get("args", []):
        if p in params:
            variable_replacements[p] = p + "_" + str(len([a for a in params if a.startswith(p)]))
            p = variable_replacements[p]
        params.append(p)

    # glue bodies together by stitching outputs from one to inputs of the other
    body = fd1.get("body", [])
    for line in fd2.get("body", []):
        # find all member variables in the line and replace them with the new variable names
        self_token_reps = get_self_token_replacements(line, rep_idx, prefix)
        variable_replacements.update(self_token_reps)
        line = replace_all(line, variable_replacements)
        body.append(line)
    return {
        "args": params,
        "body": body,
        "return": []
    }, {k: v for k, v in variable_replacements.items() if k.startswith('self.')}


def glue_forwards(fd1: dict, fd2: dict, variable_replacements: dict, rep_idx=0, prefix="node"):
    # glue parameters together
    params = fd1.get("args", [])
    # TODO: coordinate variable name changes between functions

    # glue bodies together by stitching outputs from one to inputs of the other
    body = fd1.get("body", [])
    next_inputs = fd2.get("args", [])
    for idx, ret in enumerate(fd1.get("return", [])):
        out_name = f"self.{prefix}{rep_idx - 1}_out{idx}"
        body.append(f"{out_name} = {ret}")
        if len(next_inputs) > idx:
            variable_replacements[next_inputs[idx]] = out_name
        else:
            print(f"WARNING! Not enough inputs for output function ({out_name} not passed along)!")

    for line in fd2.get("body", []):
        variable_replacements.update(get_self_token_replacements(line, rep_idx, prefix))
        line = replace_all(line, variable_replacements)
        body.append(line)

    # glue return statements together
    ret = fd2.get("return", [])
    translated_ret = []
    for line in ret:
        line = replace_all(line, variable_replacements)
        translated_ret.append(line)
    return {
        "args": params,
        "body": body,
        "return": translated_ret
    }, {k: v for k, v in variable_replacements.items() if k.startswith('self.')}


def glue_backwards(fd1: dict, fd2: dict, variable_replacements: dict, rep_idx=0, prefix="node"):
    # glue parameters together
    param = "error_gradient"  # backward() always takes in the error gradient

    # glue bodies together by stitching outputs from one to inputs of the other
    body = fd2.get("body", [])
    # backward() always returns a tuple of (update, gradient)
    # but sometimes the update is None, and/or the gradient is just passed along
    # TODO: filter out the trivial cases
    output_update, output_gradient = fd2.get("return", ("None", param))
    # TODO: collate updates & return aggregate
    does_update = output_update != "None"
    if does_update:
        out_name = f"{prefix}{rep_idx}_update"
        output_update = replace_all(output_update, variable_replacements)
        body.append(f"{out_name} = {output_update}")

    if output_gradient != param or len(body) > 0:
        out_name = f"{prefix}{rep_idx}_gradient"
        output_gradient = replace_all(output_gradient, variable_replacements)
        body.append(f"{out_name} = {output_gradient}")
        variable_replacements[param] = out_name

    # take outputs from the *previous to the previous* node's forward() function
    if "inputs" not in variable_replacements:
        variable_replacements["inputs"] = f"self.{prefix}{rep_idx - 2}_out0"

    for line in fd1.get("body", []):
        line = replace_all(line, variable_replacements)
        body.append(line)

    # glue return statements together
    update_ret, gradient_ret = "None", param
    fd1ret = fd1.get("return", ["None", param])
    if fd1ret:
        update_ret, gradient_ret = fd1ret
    if update_ret != "None":
        update_ret = replace_all(update_ret, variable_replacements)
    gradient_ret = replace_all(gradient_ret, variable_replacements)
    return {
        "args": [param],
        "body": body,
        "return": [update_ret, gradient_ret]
    }, {k: v for k, v in variable_replacements.items() if k.startswith('self.')}


def glue_optimizations(fd1: dict, fd2: dict, var_replaces: dict, rep_idx=0, prefix="node"):
    # TODO: coordinate variable name changes between functions
    inits, var_replaces = glue_inits(fd1.get("__init__", {}), fd2.get("__init__", {}), var_replaces, rep_idx, prefix)
    forwards, var_replaces = glue_forwards(fd1.get("forward", {}), fd2.get("forward", {}), var_replaces, rep_idx, prefix)
    backwards, var_replaces = glue_backwards(fd1.get("backward", {}), fd2.get("backward", {}), var_replaces, rep_idx, prefix)
    return {
        "__init__": inits,
        "forward": forwards,
        "backward": backwards,
    }, var_replaces


# NOTE: these functions are tentative and will probably be moved to a more appropriate location later
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    a[:] = a[p]
    b[:] = b[p]


def binarize(y, num_classes):
    y = y.astype(int)
    targets = np.zeros((len(y), num_classes), np.float32)
    for i in range(targets.shape[0]):
        targets[i][y[i]] = 1
    return targets


def debinarize(y):
    return y.argmax(axis=1)
#####


class OptimizableFunction:
    def __call__(self, inputs: Any):
        return self.forward(inputs)

    def forward(self, inputs: Any):
        return inputs

    def backward(self, error_gradient, last_recorded_input: Any) -> Tuple[Any, Any]:
        return None, error_gradient

    # this is a good enough general case, but sometimes there are better ways
    # returns a dict describing each of the methods __init__, forward, and backward
    # TODO: implement freezing parameters & initializations (at the graph level?)
    def optimize(self, var_replaces, rep_idx: int = 0, prefix="__node", freeze_inits=False, freeze_params=False) -> dict:
        funcs = [self.__init__, self.forward, self.backward]
        desc = {}
        if "inputs" not in var_replaces and rep_idx > 0:
            var_replaces["inputs"] = f"self.{prefix}{rep_idx - 1}_out0"
        if "last_recorded_input" not in var_replaces and rep_idx > 0:
            var_replaces["last_recorded_input"] = f"self.{prefix}{rep_idx - 1}_out0"
        for f in funcs:
            if hasattr(f, '__code__'):
                desc[f.__name__], vr = characterize_function(f, var_replaces, rep_idx, prefix)
                var_replaces.update(vr)
        return desc


class NodeFunction(OptimizableFunction):
    input_shape = None

    # TODO: implement optimize method to account for standardize_input method
    def __call__(self, inputs):
        return self.forward(self.standardize_input(inputs))

    def standardize_input(self, inputs):
        return inputs

    def update(self, updates, learning_rate):
        pass
