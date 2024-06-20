
import numpy as np
import cupy as cp
import pytest
import random
from functools import wraps
from time_trials.cython_funcs.mult import matmul_npc, matmul_cpc
# this is really just an example file for me to experiment with

NUM_ROUNDS = 1000
MATRIX_SIZE_MIN = 1000
MATRIX_SIZE_MAX = 1000


def benchmark_setup(function):
    @wraps(function)
    def inner(*args, **kwargs):
        def setup():
            return function(*args, **kwargs)
        return setup
    return inner


@benchmark_setup
def random_mats_np(low_dim, high_dim, low_val, high_val):
    dim_x = random.randint(low_dim, high_dim)
    dim_y = random.randint(low_dim, high_dim)
    return ((np.random.uniform(low=low_val, high=high_val, size=(dim_x, dim_y)),
            np.random.uniform(low=low_val, high=high_val, size=(dim_y, dim_x))), {})


@benchmark_setup
def random_mats_cp(low_dim, high_dim, low_val, high_val):
    dim_x = random.randint(low_dim, high_dim)
    dim_y = random.randint(low_dim, high_dim)
    return ((cp.random.uniform(low=low_val, high=high_val, size=(dim_x, dim_y)),
            cp.random.uniform(low=low_val, high=high_val, size=(dim_y, dim_x))), {})


@pytest.mark.benchmark(group="matmul")
def test_npmult(benchmark):
    benchmark.pedantic(np.matmul, setup=random_mats_np(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0), rounds=NUM_ROUNDS, warmup_rounds=10)


@pytest.mark.benchmark(group="matmul_gpu")
def test_cpmult(benchmark):
    benchmark.pedantic(cp.matmul, setup=random_mats_cp(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0), rounds=NUM_ROUNDS, warmup_rounds=10)


@pytest.mark.benchmark(group="matmul")
def test_npmult_c(benchmark):
    benchmark.pedantic(matmul_npc, setup=random_mats_np(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0), rounds=NUM_ROUNDS, warmup_rounds=10)


@pytest.mark.benchmark(group="matmul_gpu")
def test_cpmult_c(benchmark):
    benchmark.pedantic(matmul_cpc, setup=random_mats_cp(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0), rounds=NUM_ROUNDS, warmup_rounds=10)
