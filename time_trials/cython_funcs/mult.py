import numpy as np
import cupy as cp


def matmul_npc(a, b):
    return np.matmul(a, b)


def matmul_cpc(a, b):
    return cp.matmul(a, b)
