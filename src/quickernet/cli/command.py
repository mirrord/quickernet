import argparse
import inspect
from functools import wraps


class CLIError(Exception):
    pass


def split_docstring(docstring):
    if not docstring:
        return {}
    docs = {}
    keep_going = False
    for line in docstring.split('\n'):
        sline = line.strip()
        if not sline:
            keep_going = False
        elif sline.startswith('@'):
            key, value = sline[1:].split(':', 1)
            docs[key.strip()] = value.strip()
            keep_going = True
        elif keep_going:
            docs[key] += " " + line
    return docs


def command(subs, parent_parser):
    def decorator(function):
        cmd = function.__name__.replace('_', '-')
        # print(f"adding command: {cmd}")
        command_docstring = function.__doc__ or ""
        command_docstring = command_docstring.split('\n')[0]
        subparser = subs.add_parser(
            cmd, help=command_docstring)
        docstrings = split_docstring(function.__doc__)
        for varname, vartype in function.__annotations__.items():
            defaults = {} if function.__kwdefaults__ is None else function.__kwdefaults__.items()
            docstring = docstrings.get(varname, str(vartype))
            if varname in defaults:
                subparser.add_argument(
                    f'--{varname}', type=vartype, default=function.__kwdefaults__['varname'], help=docstring)
            else:
                subparser.add_argument(
                    f'--{varname}', type=vartype, help=docstring)

        @wraps(function)
        def wrapper(args: argparse.Namespace):

            params = {}
            for param_name, param in inspect.signature(function).parameters.items():
                params[param_name] = args.__dict__.get(param_name, None)
                if params[param_name] is None:
                    if param.default == inspect.Parameter.empty:
                        raise CLIError(f"missing required argument: {param_name} of type {param.annotation.__name__}")
                    else:
                        params[param_name] = param.default
            result = function(**params)
            return result

        subparser.set_defaults(func=wrapper)
        # print(f"added subparser: {subparser}")
        return wrapper

    return decorator
