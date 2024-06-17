import argparse
import sys
from .cli.command import command

global_parser = argparse.ArgumentParser(
    description="Experiment with quicknet.")  # , add_help=False)
cmd_subparser = global_parser.add_subparsers(
    help="base command", dest="command")


@command(cmd_subparser, global_parser)
def run():
    '''just a test function'''
    print("hello world")


@command(cmd_subparser, global_parser)
def poop(a: int):
    '''poop lol'''
    return print(a + 10)


if __name__ == "__main__":
    args = global_parser.parse_args(sys.argv[1:])
    args.func(args)
