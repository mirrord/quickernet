
import numpy as np
import cupy as cp
import pytest
# from time_trials.cython_funcs.mult import matmul_npc
# this is really just an example file for me to experiment with


def random_mats(cuda, dim):
    lib = cp if cuda else np

    def setup():
        return ((lib.random.uniform(low=0.1, high=8.0, size=(dim, dim)),
                lib.random.uniform(low=0.1, high=8.0, size=(dim, dim))), {})
    return setup


@pytest.mark.benchmark(group="matmul")
def test_npmult(benchmark):
    benchmark.pedantic(np.matmul, setup=random_mats(False, 500), rounds=100, warmup_rounds=10)


@pytest.mark.benchmark(group="matmul")
def test_cpmult(benchmark):
    benchmark.pedantic(cp.matmul, setup=random_mats(True, 500), rounds=100, warmup_rounds=10)

