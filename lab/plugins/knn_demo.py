"""k-NN demo task for the Stage-7 Implementer (live demo).

A clean, ungameable "implement an algorithm" task: the agent gets the training set AND the
test FEATURES (no labels) and must output predictions; the held-out test LABELS are
evaluator-only, so it can't cheat — it has to actually implement k-NN. The harness eval is
generic (accuracy of predictions vs hidden labels), so the agent isn't writing its grader.

Dataset is generated deterministically with the stdlib (no host numpy, no download)."""
from __future__ import annotations

import csv
import random
from pathlib import Path

from ..agents.implementer import ImplementationTask
from ..models import DatasetRef, FrameworkSpec

CLASSES = [(2.0, 2.0), (-2.0, 2.0), (0.0, -2.5)]   # 3 well-separated 2-D blobs
_N_TRAIN, _N_TEST, _SPREAD, _SEED = 300, 150, 0.9, 0


def _make(n: int, rng: random.Random):
    rows = []
    for _ in range(n):
        c = rng.randrange(len(CLASSES))
        cx, cy = CLASSES[c]
        rows.append((cx + rng.gauss(0, _SPREAD), cy + rng.gauss(0, _SPREAD), c))
    return rows


def _write(path: Path, rows, with_label: bool):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for x, y, c in rows:
            w.writerow([f"{x:.5f}", f"{y:.5f}"] + ([c] if with_label else []))


class KnnDatasetProvider:
    """Deterministic: train.csv + test_features.csv in the train split; test_labels.csv in
    the held-out split. Both fetch calls regenerate the same data (fixed seed)."""

    def fetch(self, ref: DatasetRef, dest: Path) -> None:
        rng = random.Random(_SEED)
        train = _make(_N_TRAIN, rng)
        test = _make(_N_TEST, rng)
        if ref.held_out:
            _write(dest / "test_labels.csv", [(0, 0, c) for _, _, c in test], with_label=True)
        else:
            _write(dest / "train.csv", train, with_label=True)
            _write(dest / "test_features.csv", test, with_label=False)


_EVAL_CODE = '''\
import csv, json, os
def col(path, idx):
    with open(path) as f:
        return [int(float(r[idx])) for r in csv.reader(f)]
pred = col(os.path.join(os.environ["LAB_ARTIFACTS"], "predictions.csv"), 0)
true = col(os.path.join(os.environ["LAB_DATA"], "test_labels.csv"), -1)
acc = sum(p == t for p, t in zip(pred, true)) / len(true)
json.dump({"acc": round(acc, 4)}, open(os.path.join(os.environ["LAB_EVAL_OUT"], "heldout.json"), "w"))
print("held-out acc", acc)
'''


def knn_task() -> ImplementationTask:
    return ImplementationTask(
        description=("Implement a k-nearest-neighbours classifier (NumPy). Read the labelled "
                     "training set $LAB_DATA/train.csv (columns: x1,x2,label) and the "
                     "unlabelled $LAB_DATA/test_features.csv (columns: x1,x2). Predict a "
                     "label for each test row and write them, one integer per line, to "
                     "$LAB_ARTIFACTS/predictions.csv (same order as test_features.csv)."),
        framework=FrameworkSpec("torch", "2.4", "12.1"),   # pytorch image ships numpy
        entry_command="python3 $LAB_CODE/main.py",
        eval_command="python3 $LAB_CODE/eval.py",
        eval_code=_EVAL_CODE,
        metric="acc", op=">=", threshold=0.80,
        datasets=[DatasetRef(name="knn-train", source="synthetic"),
                  DatasetRef(name="knn-heldout", source="synthetic", held_out=True)],
    )
