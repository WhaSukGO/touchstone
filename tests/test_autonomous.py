"""Stage 4 tests — research history + autonomous lineage loop. Offline (dummy harness)."""
from __future__ import annotations

from lab.factory import build_dummy_harness
from lab.history import ResearchHistory
from lab.models import ExperimentRecord, Status, Usage, VerifiedResult
from lab.plugins.dummy import ScriptedPlanner, seed_experiment
from lab.registry import Registry


# ---- research history -------------------------------------------------------
def test_history_summary_and_best(tmp_path):
    reg = Registry(tmp_path / "r.db")
    reg.upsert(ExperimentRecord(id="a", hypothesis="h", status=Status.VERIFIED,
                                verdict=VerifiedResult("a", "PASS", measured_metrics={"top1": 0.60})))
    reg.upsert(ExperimentRecord(id="b", hypothesis="h", status=Status.VERIFIED,
                                verdict=VerifiedResult("b", "PASS", measured_metrics={"top1": 0.70})))
    reg.upsert(ExperimentRecord(id="c", hypothesis="h", status=Status.REJECTED,
                                verdict=VerifiedResult("c", "FAIL", measured_metrics={"top1": 0.20})))
    h = ResearchHistory(reg)
    assert h.best("top1") == ("b", 0.70)        # only PASS counts, highest wins
    s = h.summary()
    assert "a:" in s and "b:" in s and "c:" in s


# ---- autonomous lineage -----------------------------------------------------
class LineagePlanner:
    """Runs `total` experiments then declines. propose uses the dummy good contract."""

    def __init__(self, total: int):
        self.total = total
        self.count = 0
        self._sp = ScriptedPlanner()
        self.decide_calls: list = []

    def propose_contract(self, rec):
        return self._sp.propose_contract(rec)

    def decide_next(self, result, rec):
        self.decide_calls.append(result.verdict if result else None)
        self.count += 1
        if self.count < self.total:
            return ExperimentRecord(id=f"auto-{self.count}", hypothesis="good",
                                    parent_id=rec.id), Usage()
        return None, Usage()


def _harness(tmp_path, name):
    h = build_dummy_harness(tmp_path / name, images_path="images/registry.yaml")
    h.autonomy_enabled = True
    return h


def test_loop_runs_lineage_until_planner_declines(tmp_path):
    h = _harness(tmp_path, "lab1")
    h.planner = LineagePlanner(total=3)
    h.queue.push(seed_experiment("good", exp_id="auto-0"))

    summary = h.loop(require_gate=True)

    assert summary["experiments_ran"] == 3
    assert len(h.registry.query(statuses=[Status.VERIFIED])) == 3
    # each experiment links to its parent (a lineage, not isolated runs)
    assert h.registry.get("auto-2").parent_id == "auto-1"


def test_loop_stops_on_stall(tmp_path):
    h = _harness(tmp_path, "lab2")
    h.planner = LineagePlanner(total=999)        # never declines on its own
    h.queue.push(seed_experiment("good", exp_id="auto-0"))

    summary = h.loop(require_gate=True, goal_metric="score", max_stall=2)

    # dummy good always measures score=0.95: best on auto-0, then 2 stalls -> stop
    assert summary["experiments_ran"] == 3
    assert summary["best"] == 0.95


def test_decide_next_called_after_rejection(tmp_path):
    h = _harness(tmp_path, "lab3")
    planner = LineagePlanner(total=2)
    h.planner = planner
    h.queue.push(seed_experiment("poison", exp_id="auto-0"))   # -> REJECTED

    h.loop(require_gate=True)

    # the loop consulted the planner even after a REJECTED experiment (recovery, not just
    # chaining successes), and the lineage continued with a fresh experiment
    assert planner.decide_calls[0] == "FAIL"
    assert h.registry.get("auto-0").status == Status.REJECTED
    assert h.registry.get("auto-1").status == Status.VERIFIED


def test_loop_respects_experiment_cap(tmp_path):
    h = build_dummy_harness(tmp_path / "lab4", images_path="images/registry.yaml",
                            max_experiments=2)
    h.autonomy_enabled = True
    h.planner = LineagePlanner(total=999)
    h.queue.push(seed_experiment("good", exp_id="auto-0"))

    summary = h.loop(require_gate=True)
    assert summary["experiments_ran"] == 2       # hard cap stops the lineage
