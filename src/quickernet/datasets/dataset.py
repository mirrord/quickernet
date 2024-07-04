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

# TODO: generalize for multiple input channels


class Dataset:
    def __init__(self, inputs, labels):
        self._inputs = inputs
        self._labels = labels
        self._binarized = sum(self._labels[..., -1]) == 1
        # TODO: generalize this for multiple input & output channels
        self._binshape = labels.shape if self._binarized else (
            len(self._inputs), self.num_classes())

    def __len__(self):
        return len(self._inputs)

    def shuffle(self):
        p = np.random.permutation(len(self))
        self._inputs = self._inputs[p]
        self._labels = self._labels[p]

    def binarize(self, num_classes):
        if self._binarized:
            return
        # TODO: test this against my utils function!
        self._labels = np.eye(num_classes)[
            self._labels].reshape(self._binshape)
        self._binarized = True

    def debinarize(self):
        if not self._binarized:
            return
        self._binshape = self._labels.shape
        self._labels = self._labels.argmax(axis=-1).flatten()
        self._binarized = False

    def num_classes(self):
        output_length = self._labels.shape[-1]
        if output_length == 1 or output_length == len(self._labels):
            return (max(self._labels.flatten()) + 1).item()
        return output_length

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

    def random_item(self):
        idx = np.random.randint(len(self))
        return self._inputs[idx], self._labels[idx]

    def random_subset(self, num_items: int):
        idxs = np.random.randint(len(self), size=num_items)
        return Dataset(self._inputs[idxs], self._labels[idxs])

    def __iter__(self):
        return iter(zip(self._inputs, self._labels))
