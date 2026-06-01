"""CIFAR-10 domain plugin (Stage 2 calibration target).

Harness-process side: downloads CIFAR-10 ONCE (a harness job, zero tokens), splits it
into a generator-visible train set and an evaluator-only held-out test set, and supplies
the fixed reference contract. Calibration uses fixed reference scripts (not the LLM) on
purpose — it validates the harness + evaluator against a known answer."""
from __future__ import annotations

import shutil
import tarfile
import urllib.request
from pathlib import Path

from ..models import (
    BudgetSpec, Criterion, DatasetRef, ExperimentContract, ExperimentRecord, FrameworkSpec,
    OracleRef, Status, Usage, VerifiedResult,
)
from ..util import ensure_dir

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CODE_DIR = str((Path(__file__).parent / "cifar_ref").resolve())

# Loose, robustly-reproducible bar: a 2-epoch SmallCNN clears this comfortably, while a
# random (0-epoch) model sits near 0.10 — a wide margin so calibration is not flaky.
_CRITERION = Criterion(metric="top1", op=">=", value=0.45, tolerance=0.0)

_TRAIN_BATCHES = [f"data_batch_{i}" for i in range(1, 6)]
_TEST_BATCH = "test_batch"


class CifarDatasetProvider:
    """Downloads + extracts CIFAR-10 once; materializes train vs held-out per DatasetRef."""

    def __init__(self, raw_root: str | Path):
        self.raw_root = ensure_dir(raw_root)
        self._batches = self.raw_root / "cifar-10-batches-py"

    def _ensure_raw(self) -> None:
        if self._batches.exists() and (self._batches / _TEST_BATCH).exists():
            return
        tgz = self.raw_root / "cifar-10-python.tar.gz"
        if not tgz.exists():
            urllib.request.urlretrieve(CIFAR_URL, tgz)
        with tarfile.open(tgz, "r:gz") as t:
            t.extractall(self.raw_root)

    def fetch(self, ref: DatasetRef, dest: Path) -> None:
        self._ensure_raw()
        wanted = [_TEST_BATCH] if ref.held_out else _TRAIN_BATCHES
        for name in wanted:
            shutil.copy(self._batches / name, dest / name)


class CifarMetricExtractor:
    def extract(self, artifacts_dir: str) -> dict:
        import json
        p = Path(artifacts_dir) / "metrics.json"
        return json.loads(p.read_text()) if p.exists() else {}


def _contract(poison: bool) -> ExperimentContract:
    epochs = 0 if poison else 2
    env = (("LAB_POISON=1 " if poison else "") + f"LAB_EPOCHS={epochs} LAB_SEED=0 ")
    return ExperimentContract(
        success_definition="held-out CIFAR-10 top-1 >= 0.45",
        gradable_criteria=[_CRITERION],
        framework=FrameworkSpec(name="torch", version="2.4", cuda="12.1"),
        datasets=[
            DatasetRef(name="cifar10-train", source=CIFAR_URL),
            DatasetRef(name="cifar10-heldout", source=CIFAR_URL, held_out=True),
        ],
        command=f"{env}python /code/train.py",
        eval_command="python /code/eval.py",
        code_dir=CODE_DIR,
        budget=BudgetSpec(max_tokens=50_000, max_wall_s=1800, max_retries=1),
        oracle=OracleRef(criterion=_CRITERION, source="cifar10-smallcnn-calibration"),
        seed=0,
    )


class CifarPlanner:
    """Calibration planner: fixed contract keyed off the hypothesis (good vs poison)."""

    def propose_contract(self, rec: ExperimentRecord) -> tuple[ExperimentContract, Usage]:
        poison = "poison" in rec.hypothesis.lower()
        return _contract(poison), Usage()

    def decide_next(self, result: VerifiedResult, rec: ExperimentRecord
                    ) -> tuple[ExperimentRecord | None, Usage]:
        return None, Usage()


def seed_experiment(hypothesis: str, *, exp_id: str, priority: int = 0) -> ExperimentRecord:
    return ExperimentRecord(id=exp_id, hypothesis=hypothesis,
                            status=Status.PROPOSED, priority=priority)


def cifar_contract(poison: bool) -> ExperimentContract:
    """Public access to the fixed calibration contract."""
    return _contract(poison)


def calibration_records() -> tuple[ExperimentRecord, ExperimentRecord]:
    """Pre-contracted positive/negative calibration records. Because the contract is
    pre-set, the planner is skipped — calibration always uses the fixed reference
    scripts, even when an LLM planner is wired for research experiments."""
    pos = ExperimentRecord(id="cal-pos", hypothesis="cifar positive control",
                           status=Status.PROPOSED, contract=_contract(False), priority=100)
    neg = ExperimentRecord(id="cal-neg", hypothesis="cifar negative control",
                           status=Status.PROPOSED, contract=_contract(True), priority=100)
    return pos, neg
