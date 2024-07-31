

from . import cli
from .command import CLIError


def CLI(cli_args):
    args = cli.global_parser.parse_args(cli_args)
    if getattr(args, 'command') is None:
        try:
            args.func(args)
        except CLIError as e:
            print(f"Error: {e}")
            cli.global_parser.print_help()
            exit(1)
    else:
        cli.global_parser.print_help()
