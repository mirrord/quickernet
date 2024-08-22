import numpy as np
import cupy as cp
import pytest
from time_trials.utils import random_mats_cp, random_mats_np
from time_trials.cython_funcs.activate import (
    softmax as softmax_cy,
    sigmoid as sigmoid_cy,
    relu as relu_cy,
    leaky_relu as leaky_relu_cy,
    tanh as tanh_cy,
    swish as swish_cy,
    scale as scale_cy,
    norm as norm_cy,
    l2norm as l2norm_cy,
)


NUM_ROUNDS = 1000
MATRIX_SIZE_MIN = 10000
MATRIX_SIZE_MAX = 10000


def softmax_np(inputs):
    exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


def softmax_cp(inputs):
    exp = cp.exp(inputs - cp.max(inputs, axis=1, keepdims=True))
    return exp / cp.sum(exp, axis=1, keepdims=True)


@pytest.mark.benchmark(group="softmax")
def test_softmax_np(benchmark):
    benchmark.pedantic(
        softmax_np,
        random_mats_np(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


@pytest.mark.benchmark(group="softmax")
def test_softmax_cp(benchmark):
    benchmark.pedantic(
        softmax_cp,
        random_mats_cp(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


@pytest.mark.benchmark(group="softmax")
def test_softmax_cy(benchmark):
    benchmark.pedantic(
        softmax_cy,
        random_mats_cp(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


def sigmoid_np(inputs):
    1 / (1 + np.exp(-inputs))


def sigmoid_cp(inputs):
    1 / (1 + cp.exp(-inputs))


sigmoid_cpkern = cp.ElementwiseKernel(
    "float64 x", "float64 y", "y = 1 / (1 + exp(-x))", "expit"
)


@pytest.mark.benchmark(group="sigmoid")
def test_sigmoid_np(benchmark):
    benchmark.pedantic(
        sigmoid_np,
        random_mats_np(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


@pytest.mark.benchmark(group="sigmoid")
def test_sigmoid_cp(benchmark):
    benchmark.pedantic(
        sigmoid_cp,
        random_mats_cp(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


@pytest.mark.benchmark(group="sigmoid")
def test_sigmoid_cpkern(benchmark):
    benchmark.pedantic(
        sigmoid_cpkern,
        random_mats_cp(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


@pytest.mark.benchmark(group="sigmoid")
def test_sigmoid_cy(benchmark):
    benchmark.pedantic(
        sigmoid_cy,
        random_mats_cp(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


def relu_np(inputs):
    return np.maximum(0, inputs)


def relu_cp(inputs):
    return np.maximum(0, inputs)


relu_cpkern = cp.ElementwiseKernel("float64 x", "float64 y", "y = x * (x > 0)", "relu")


@pytest.mark.benchmark(group="relu")
def test_relu_np(benchmark):
    benchmark.pedantic(
        relu_np,
        random_mats_np(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


@pytest.mark.benchmark(group="relu")
def test_relu_cp(benchmark):
    benchmark.pedantic(
        relu_cp,
        random_mats_cp(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


@pytest.mark.benchmark(group="relu")
def test_relu_cpkern(benchmark):
    benchmark.pedantic(
        relu_cpkern,
        random_mats_cp(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


@pytest.mark.benchmark(group="relu")
def test_relu_cy(benchmark):
    benchmark.pedantic(
        relu_cy,
        random_mats_cp(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


def leaky_relu_np(inputs):
    return np.maximum(0.01 * inputs, inputs)


def leaky_relu_cp(inputs):
    return cp.maximum(0.01 * inputs, inputs)


leaky_relu_cpkern = cp.ElementwiseKernel(
    "float64 x", "float64 y", "y = (x * (x > 0)) + ((x <= 0) * x * 0.01)", "leakyurelu"
)


@pytest.mark.benchmark(group="leaky_relu")
def test_leaky_relu_np(benchmark):
    benchmark.pedantic(
        leaky_relu_np,
        random_mats_np(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


@pytest.mark.benchmark(group="leaky_relu")
def test_leaky_relu_cp(benchmark):
    benchmark.pedantic(
        leaky_relu_cp,
        random_mats_cp(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


@pytest.mark.benchmark(group="leaky_relu")
def test_leaky_relu_cpkern(benchmark):
    benchmark.pedantic(
        leaky_relu_cpkern,
        random_mats_cp(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


@pytest.mark.benchmark(group="leaky_relu")
def test_leaky_relu_cy(benchmark):
    benchmark.pedantic(
        leaky_relu_cy,
        random_mats_cp(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


def tanh_np(inputs):
    return np.tanh(inputs)


def tanh_cp(inputs):
    return cp.tanh(inputs)


@pytest.mark.benchmark(group="tanh")
def test_tanh_np(benchmark):
    benchmark.pedantic(
        tanh_np,
        random_mats_np(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


@pytest.mark.benchmark(group="tanh")
def test_tanh_cp(benchmark):
    benchmark.pedantic(
        tanh_cp,
        random_mats_cp(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


@pytest.mark.benchmark(group="tanh")
def test_tanh_cy(benchmark):
    benchmark.pedantic(
        tanh_cy,
        random_mats_cp(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


def swish_np(inputs):
    return inputs * (1 / (1 + np.exp(-inputs)))


def swish_cp(inputs):
    return inputs * (1 / (1 + cp.exp(-inputs)))


@pytest.mark.benchmark(group="swish")
def test_swish_np(benchmark):
    benchmark.pedantic(
        swish_np,
        random_mats_np(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


@pytest.mark.benchmark(group="swish")
def test_swish_cp(benchmark):
    benchmark.pedantic(
        swish_cp,
        random_mats_cp(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


@pytest.mark.benchmark(group="swish")
def test_swish_cy(benchmark):
    benchmark.pedantic(
        swish_cy,
        random_mats_cp(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


def scale_np(inputs):
    return inputs / inputs.shape[1]


def scale_cp(inputs):
    return inputs / inputs.shape[1]


@pytest.mark.benchmark(group="scale")
def test_scale_np(benchmark):
    benchmark.pedantic(
        scale_np,
        random_mats_np(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


@pytest.mark.benchmark(group="scale")
def test_scale_cp(benchmark):
    benchmark.pedantic(
        scale_cp,
        random_mats_cp(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


@pytest.mark.benchmark(group="scale")
def test_scale_cy(benchmark):
    benchmark.pedantic(
        scale_cy,
        random_mats_cp(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


def norm_np(inputs):
    return inputs / np.max(inputs, axis=1, keepdims=True)


def norm_cp(inputs):
    return inputs / cp.max(inputs, axis=1, keepdims=True)


@pytest.mark.benchmark(group="norm")
def test_norm_np(benchmark):
    benchmark.pedantic(
        norm_np,
        random_mats_np(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


@pytest.mark.benchmark(group="norm")
def test_norm_cp(benchmark):
    benchmark.pedantic(
        norm_cp,
        random_mats_cp(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


@pytest.mark.benchmark(group="norm")
def test_norm_cy(benchmark):
    benchmark.pedantic(
        norm_cy,
        random_mats_cp(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


def l2norm_np(inputs):
    return inputs / np.linalg.norm(inputs, axis=1, keepdims=True)


def l2norm_cp(inputs):
    return inputs / cp.linalg.norm(inputs, axis=1, keepdims=True)


@pytest.mark.benchmark(group="l2norm")
def test_l2norm_np(benchmark):
    benchmark.pedantic(
        l2norm_np,
        random_mats_np(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


@pytest.mark.benchmark(group="l2norm")
def test_l2norm_cp(benchmark):
    benchmark.pedantic(
        l2norm_cp,
        random_mats_cp(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )


@pytest.mark.benchmark(group="l2norm")
def test_l2norm_cy(benchmark):
    benchmark.pedantic(
        l2norm_cy,
        random_mats_cp(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, 0.1, 8.0),
        rounds=NUM_ROUNDS,
        warmup_rounds=10,
    )
