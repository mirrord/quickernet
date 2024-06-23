

from quickernet.cli import cli


def CLI(cli_args):
    args = cli.global_parser.parse_args(cli_args)
    args.func(args)

