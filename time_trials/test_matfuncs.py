
import numpy as np
import cupy as cp
import pytest
from time_trials.utils import random_mats_cp, random_mats_np
from time_trials.cython_funcs.mult import matmul_npc, matmul_cpc
# this is really just an example file for me to experiment with

NUM_ROUNDS = 1000
MATRIX_SIZE_MIN = 10000
MATRIX_SIZE_MAX = 10000


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
