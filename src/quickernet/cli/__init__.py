

from quickernet.cli import cli


def CLI(cli_args):
    args = cli.global_parser.parse_args(cli_args)
    if getattr(args, 'command') is None:
        args.func(args)
    else:
        cli.global_parser.print_help()
    

