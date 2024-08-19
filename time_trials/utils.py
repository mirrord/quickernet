
import numpy as np
import cupy as cp
from functools import wraps
import random


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