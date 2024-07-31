import cupy as np


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    a[:] = a[p]
    b[:] = b[p]


def binarize(y, num_classes):
    y = y.astype(int)
    targets = np.zeros((len(y), num_classes), np.float32)
    for i in range(targets.shape[0]):
        targets[i][y[i]] = 1
    return targets


def debinarize(y):
    return y.argmax(axis=1)
