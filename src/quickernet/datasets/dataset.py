import hashlib
import requests
import gzip
from pathlib import Path

import cupy as np


class DatasetManager:
    _storage_path = None

    def __init__(self, store_path: str = "datasets"):
        self._storage_path = Path(
            __file__).parent.parent.parent.parent / store_path

    def gz_fetch(self, url):
        fp = self._storage_path / hashlib.md5(url.encode("utf-8")).hexdigest()
        if fp.is_file():
            with open(fp, "rb") as f:
                data = f.read()
        else:
            with open(fp, "wb") as f:
                data = requests.get(url).content
                f.write(data)
        return np.array(np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy())

    def fetch_separate(self, x_loc, y_loc, x_test_loc, y_test_loc, x_shape=None, y_shape=None, x_test_shape=None, y_test_shape=None, x_start=16, y_start=8):
        x = self.gz_fetch(x_loc)[x_start:]
        if x_shape:
            x = x.reshape(x_shape)
        y = self.gz_fetch(y_loc)[y_start:]
        if y_shape:
            y = y.reshape(y_shape)
        x_test = self.gz_fetch(x_test_loc)[x_start:]
        if x_test_shape:
            x_test = x_test.reshape(x_test_shape)
        y_test = self.gz_fetch(y_test_loc)[y_start:]
        if y_test_shape:
            y_test = y_test.reshape(y_test_shape)
        return Dataset(x, y), Dataset(x_test, y_test)


class Dataset:
    def __init__(self, inputs, labels):
        self._inputs = inputs
        self._labels = labels

    def __len__(self):
        return len(self._inputs)

    def shuffle(self):
        p = np.random.permutation(len(self))
        self._inputs = self._inputs[p]
        self._labels = self._labels[p]

    def binarize(self, num_classes):
        # TODO: test this against my utils function!
        self._labels = np.eye(num_classes)[self._labels]

    def split(self, ratio: float):
        split_idx = int(len(self) * ratio)
        return Dataset(self._inputs[:split_idx], self._labels[:split_idx]), Dataset(self._inputs[split_idx:], self._labels[split_idx:])

    def split_validation(self, test_ratio: float, validation_ratio: float):
        test_split = int(len(self) * test_ratio)
        val_split = int(len(self) * validation_ratio) + test_split
        return Dataset(self._inputs[:test_split], self._labels[:test_split]), Dataset(self._inputs[test_split:val_split], self._labels[test_split:val_split]), Dataset(self._inputs[test_split + val_split:], self._labels[test_split + val_split:])

    def batch(self, batch_size: int):
        for i in range(0, len(self), batch_size):
            yield Dataset(self._inputs[i:i + batch_size], self._labels[i:i + batch_size])
