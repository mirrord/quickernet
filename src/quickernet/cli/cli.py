
import argparse
import matplotlib.pyplot as plt
# import cupy as np

from .command import command
from src.quickernet.nodes import node, linear, activations, synapses, costs, utils
from src.quickernet.networks import graph
from src.quickernet.datasets import dataset

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
    p1 = node.PipelineNode([
        synapses.SynapseSum(),
        linear.Linear(28 * 28, 100),
        activations.Sigmoid()
    ])

    p2 = node.PipelineNode([
        synapses.SynapseSum(),
        linear.Linear(100, 100),
        activations.Sigmoid()
    ])

    p3 = node.PipelineNode([
        synapses.SynapseSum(),
        linear.Linear(100, 10),
        activations.Sigmoid()
    ])

    g = graph.DirectedGraphModel()
    g.add_node(p1)
    g.add_node(p2)
    g.add_node(p3)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(0, 2)

    dm = dataset.DatasetManager()
    training_data, test_data = dm.fetch_separate(
        "https://web.archive.org/web/20220331130319/https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "https://web.archive.org/web/20220331130319/https://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "https://web.archive.org/web/20220331130319/https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "https://web.archive.org/web/20220331130319/https://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
        (60000, 28 * 28), (60000, 1), (10000, 28 * 28), (10000, 1)
    )

    training_data.binarize(10)
    test_data.binarize(10)

    cost_func = costs.QuadraticCost()
    test_cost_before = g.test_on(test_data, cost_func)
    # cost_history = g.train_on(training_data, cost_func, 100)
    acc_b4 = g.get_accuracy(test_data)["all"]
    train_acc_b4 = g.get_accuracy(training_data)["all"]
    cost_history = g.train_alternate(training_data, cost_func, 5, 15)
    cost_after = g.test_on(test_data, cost_func)
    test_acc_after = g.get_accuracy(test_data)["all"]
    train_acc_after = g.get_accuracy(training_data)["all"]
    print(cost_history)
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(list(cost_history.values())[0])
    sample_input, sample_output = test_data.random_item()
    network_output = g.forward(sample_input)
    print("***********")
    print("output: ")
    print(network_output)
    print("expected: ")
    print(sample_output)
    print("***********")
    print(f"sample model output: {utils.debinarize(
        network_output)}, expected output: {utils.debinarize(sample_output.reshape(1, 10))}")
    print(f"cost before: {test_cost_before}, cost after: {cost_after}")
    print(f"accuracies before: train: {train_acc_b4} test: {acc_b4}")
    print(f"accuracies now: train: {train_acc_after} test: {test_acc_after}")
    plt.subplot(122)
    plt.imshow(sample_input.get().reshape(28, 28), cmap='gray')
    plt.show()


@command(cmd_subparser, global_parser)
def poop(a: int):
    '''poop lol'''
    return print(a + 10)
