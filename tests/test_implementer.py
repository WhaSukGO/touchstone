"""Stage 7 tests — Implementer -> verification handoff. Offline (fake author, local mode).

Proves the core safety property: agent-authored code is only accepted if the INDEPENDENT
evaluator measures it passing the task's fixed oracle on held-out. Good code -> VERIFIED,
bad code -> REJECTED. The authored code actually runs (real subprocess); only the LLM that
would write it is faked."""
from __future__ import annotations

from pathlib import Path

from lab.agents.implementer import ImplementationTask
from lab.factory import build_implementer_harness
from lab.models import ExperimentRecord, FrameworkSpec, Status, Usage

# The harness owns the evaluator; the agent never writes this.
_EVAL_CODE = (
    "import json, os\n"
    "s = float(open(os.path.join(os.environ['LAB_ARTIFACTS'], 'result.txt')).read())\n"
    "json.dump({'score': s}, open(os.path.join(os.environ['LAB_EVAL_OUT'], 'heldout.json'), 'w'))\n"
)


def _task() -> ImplementationTask:
    return ImplementationTask(
        description="Author main.py that writes a held-out score >= 0.85",
        framework=FrameworkSpec("dummy", "0", ""),
        entry_command="python3 $LAB_CODE/main.py",
        eval_command="python3 $LAB_CODE/eval.py",
        eval_code=_EVAL_CODE,
        metric="score", op=">=", threshold=0.85,
    )


def _author(score: str):
    def author(task, code_dir: Path, rec) -> Usage:
        (code_dir / "main.py").write_text(
            "import os\n"
            f"open(os.path.join(os.environ['LAB_ARTIFACTS'], 'result.txt'), 'w').write('{score}')\n")
        return Usage(50, 30)
    return author


def test_implementer_good_code_is_verified(tmp_path):
    h = build_implementer_harness(tmp_path / "lab", _task(), _author("0.9"), job_mode="local")
    rec = h.run_experiment(ExperimentRecord(id="impl-1", hypothesis="implement it"))

    assert rec.status == Status.VERIFIED
    assert rec.verdict.measured_metrics["score"] == 0.9
    assert rec.verdict.signed_by == "evaluator-impl"
    # the agent did NOT author the evaluator
    assert (Path(rec.contract.code_dir) / "eval.py").exists()
    assert (Path(rec.contract.code_dir) / "main.py").exists()


def test_implementer_bad_code_is_rejected(tmp_path):
    h = build_implementer_harness(tmp_path / "lab2", _task(), _author("0.5"), job_mode="local")
    rec = h.run_experiment(ExperimentRecord(id="impl-1", hypothesis="implement it"))

    assert rec.status == Status.REJECTED      # ran fine, but missed the oracle -> not accepted
    assert rec.verdict.measured_metrics["score"] == 0.5


def test_implementer_broken_code_fails_safe(tmp_path):
    def broken(task, code_dir: Path, rec) -> Usage:
        (code_dir / "main.py").write_text("raise SystemExit('boom')\n")  # writes no artifact
        return Usage(10, 5)

    h = build_implementer_harness(tmp_path / "lab3", _task(), broken, job_mode="local")
    rec = h.run_experiment(ExperimentRecord(id="impl-1", hypothesis="implement it"))
    assert rec.status == Status.FAILED        # entry command errored -> FAILED, loop survives
