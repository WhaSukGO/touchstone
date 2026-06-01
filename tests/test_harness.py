"""Stage 1 harness tests — all run with no GPU, Docker, or model (local job mode)."""
from __future__ import annotations

from pathlib import Path

import pytest

from lab.budget import Budget
from lab.dataset_cache import DatasetCache
from lab.factory import build_dummy_harness
from lab.gpu_lease import GpuLease, LeaseTimeout
from lab.image_registry import ImageRegistry, NoImageError
from lab.models import (
    Criterion, DatasetRef, ExperimentRecord, FrameworkSpec, Status,
)
from lab.plugins.dummy import seed_experiment
from lab.registry import Registry
from lab.serde import from_dict, to_jsonable


# ---- serde + registry -------------------------------------------------------
def test_record_roundtrip_through_registry(tmp_path):
    reg = Registry(tmp_path / "r.db")
    rec = ExperimentRecord(id="x1", hypothesis="h", priority=3)
    reg.upsert(rec)
    got = reg.get("x1")
    assert got is not None
    assert got.id == "x1" and got.status == Status.PROPOSED and got.priority == 3
    # nested dataclass survives the round-trip
    again = from_dict(ExperimentRecord, to_jsonable(got))
    assert again == got


def test_registry_priority_ordering(tmp_path):
    reg = Registry(tmp_path / "r.db")
    reg.upsert(ExperimentRecord(id="lo", hypothesis="", priority=1))
    reg.upsert(ExperimentRecord(id="hi", hypothesis="", priority=9))
    ordered = reg.query(statuses=[Status.PROPOSED])
    assert [r.id for r in ordered] == ["hi", "lo"]


# ---- gpu lease (single-GPU mutex) -------------------------------------------
def test_gpu_lease_is_mutually_exclusive(tmp_path):
    lease = GpuLease(tmp_path / "gpu.lock", poll_s=0.05)
    t1 = lease.acquire("train", timeout_s=2)
    with pytest.raises(LeaseTimeout):
        lease.acquire("eval", timeout_s=0.3)  # blocked while held
    lease.release(t1)
    t2 = lease.acquire("eval", timeout_s=2)   # now available
    lease.release(t2)


# ---- image registry ---------------------------------------------------------
def test_image_resolve_hit_and_miss():
    reg = ImageRegistry("images/registry.yaml")
    img = reg.resolve(FrameworkSpec(name="torch", version="2.4", cuda="12.1"))
    assert "cuda12.1" in img.image
    with pytest.raises(NoImageError):
        reg.resolve(FrameworkSpec(name="jax", version="0.4", cuda="12.1"))


# ---- dataset cache (download once) ------------------------------------------
def test_dataset_cache_fetches_once(tmp_path):
    calls = {"n": 0}

    class CountingProvider:
        def fetch(self, ref, dest: Path):
            calls["n"] += 1
            (dest / "f.txt").write_text("x")

    cache = DatasetCache(tmp_path / "cache", CountingProvider())
    ref = DatasetRef(name="ds", source="x://ds")
    e1 = cache.ensure(ref)
    e2 = cache.ensure(ref)
    assert calls["n"] == 1                 # second ensure is a cache hit
    assert e1.sha256 == e2.sha256
    assert cache.verify("ds", deep=True)


# ---- budget (IO is never charged) -------------------------------------------
def test_budget_io_not_charged(tmp_path):
    b = Budget(max_total_tokens=100, max_experiments=10,
               state_path=tmp_path / "b.json")
    b.note_io("e", 999.0)                   # huge IO wait...
    assert b.remaining_tokens() == 100      # ...costs zero budget
    b.charge_tokens("e", 10, 5)
    assert b.remaining_tokens() == 85
    assert b.can_spawn()


def test_criterion_semantics():
    c = Criterion(metric="acc", op=">=", value=0.9, tolerance=0.01)
    assert c.satisfied(0.90) and c.satisfied(0.895)  # within tolerance
    assert not c.satisfied(0.80)


# ---- the crown jewel: calibration gate --------------------------------------
def test_calibration_gate_opens_only_when_evaluator_is_honest(tmp_path):
    h = build_dummy_harness(tmp_path / "lab", images_path="images/registry.yaml")
    pos = seed_experiment("positive control", exp_id="cal-pos")
    neg = seed_experiment("poison negative control", exp_id="cal-neg")

    assert h.calibration_gate(pos, neg) is True
    assert h.autonomy_enabled is True

    pos_rec = h.registry.get("cal-pos")
    neg_rec = h.registry.get("cal-neg")
    assert pos_rec.status == Status.VERIFIED
    assert neg_rec.status == Status.REJECTED

    # The negative control LIED (reported 0.99) but the evaluator measured the real
    # checkpoint (0.10) and rejected it. This is the whole point of the lab.
    assert neg_rec.reported_metrics.get("score") == 0.99
    assert neg_rec.verdict.measured_metrics.get("score") == 0.10
    assert neg_rec.verdict.verdict == "FAIL"
    assert neg_rec.verdict.signed_by  # evaluator signed it


def test_autonomous_loop_requires_gate(tmp_path):
    h = build_dummy_harness(tmp_path / "lab2", images_path="images/registry.yaml")
    with pytest.raises(RuntimeError):
        h.loop(require_gate=True)           # locked until calibration passes


def test_loop_runs_after_gate(tmp_path):
    h = build_dummy_harness(tmp_path / "lab3", images_path="images/registry.yaml")
    h.autonomy_enabled = True               # pretend the gate already opened
    h.queue.push(seed_experiment("a real experiment", exp_id="exp-1"))
    h.loop(require_gate=True)
    assert h.registry.get("exp-1").status == Status.VERIFIED
