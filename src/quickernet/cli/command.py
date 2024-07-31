import argparse
import inspect
from functools import wraps


class CLIError(Exception):
    pass


def command(subs, parent_parser):
    def decorator(function):
        cmd = function.__name__.replace('_', '-')
        # print(f"adding command: {cmd}")
        subparser = subs.add_parser(
            cmd, help=function.__doc__, parents=[parent_parser], add_help=False)
        for varname, vartype in function.__annotations__.items():
            defaults = {} if function.__kwdefaults__ is None else function.__kwdefaults__.items()
            if varname in defaults:
                subparser.add_argument(
                    f'--{varname}', type=vartype, default=function.__kwdefaults__['varname'], help=vartype.__name__)
            else:
                subparser.add_argument(
                    f'--{varname}', type=vartype, help=vartype.__name__)

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
