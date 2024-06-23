
# TODO: monkey patching for cupy and numpy
def cupy_or_numpy():
    try:
        import cupy as cp
        return cp
    except ImportError:
        import numpy as np
        return np
