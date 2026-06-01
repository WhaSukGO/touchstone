"""CIFAR-10 loader (runs inside the container). Reads the canonical python batch files
from LAB_DATA. No torchvision dependency — just pickle + numpy."""
import os
import pickle

import numpy as np

_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype="float32").reshape(1, 3, 1, 1)
_STD = np.array([0.2470, 0.2435, 0.2616], dtype="float32").reshape(1, 3, 1, 1)


def _load_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    data = d[b"data"].astype("float32") / 255.0
    data = data.reshape(-1, 3, 32, 32)
    labels = np.array(d[b"labels"], dtype="int64")
    return data, labels


def load_split(data_dir):
    """If a test_batch is present, return it (held-out); else concat the train batches."""
    files = sorted(os.listdir(data_dir))
    if "test_batch" in files:
        x, y = _load_batch(os.path.join(data_dir, "test_batch"))
    else:
        xs, ys = [], []
        for b in [f for f in files if f.startswith("data_batch")]:
            bx, by = _load_batch(os.path.join(data_dir, b))
            xs.append(bx)
            ys.append(by)
        x, y = np.concatenate(xs), np.concatenate(ys)
    x = (x - _MEAN) / _STD
    return x, y
