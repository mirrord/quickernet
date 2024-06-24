
import argparse

import cupy as np
from .command import command
from src.quickernet.nodes import node, linear, activations

# TODO: add support for making parameters necessary or optional
# TODO: add support for bool
# TODO: add configurable helpstrings for arguments (somehow)

global_parser = argparse.ArgumentParser(
    description="Experiment with quicknet.")  # , add_help=False)
cmd_subparser = global_parser.add_subparsers(
    help="base command", dest="command")


@command(cmd_subparser, global_parser)
def run():
    '''just a test function'''
    pipeline = [
        linear.Linear(2, 3),
        activations.Sigmoid()
    ]

    a = node.PipelineNode(pipeline)
    print(a.forward(np.array([[1, 2]])))


@command(cmd_subparser, global_parser)
def poop(a: int):
    '''poop lol'''
    return print(a + 10)

