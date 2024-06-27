
import argparse
import cupy as np

from .command import command
from src.quickernet.nodes import node, linear, activations, synapses
from src.quickernet.networks import graph

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
        synapses.SynapseSum(),
        linear.Linear(2, 3),
        activations.Sigmoid()
    ]

    pipeline2 = [
        synapses.SynapseSum(),
        linear.Linear(3, 3),
        activations.Sigmoid()
    ]

    a = node.PipelineNode(pipeline)
    b = node.PipelineNode(pipeline2)
    g = graph.DirectedGraphModel()
    g.add_node(a)
    g.add_node(b)
    g.add_edge(0, 1)
    print(g.forward(np.array([[1, 2]])))
    print(g.backward({k: v * 0.5 for k, v in g.last_outputs.items()}))


@command(cmd_subparser, global_parser)
def poop(a: int):
    '''poop lol'''
    return print(a + 10)
