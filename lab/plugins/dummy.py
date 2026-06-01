"""Deterministic dummy plugin — LLM-free stand-ins that let the whole harness + the
calibration gate be proven without GPU, Docker, ImageNet, or any model.

It also demonstrates the core anti-pattern defense: the 'poisoned' experiment writes a
truthful-but-bad checkpoint while REPORTING an inflated score. A passive reviewer that
trusts reported_metrics would be fooled; the independent evaluator measures the
checkpoint on the held-out split and catches the lie. That is the negative control."""
from __future__ import annotations

import json
from pathlib import Path

from ..gpu_lease import LeaseToken
from ..job_runner import JobRunner, JobSpec
from ..models import (
    BudgetSpec, Criterion, DatasetRef, ExperimentContract, ExperimentRecord, FrameworkSpec,
    OracleRef, Provenance, Status, Usage, VerifiedResult,
)
from ..paths import Layout
from ..util import now_iso

# A "good" run: checkpoint quality 0.95, honestly reported.
_TRAIN_GOOD = (
    "python3 -c \"import json,os; a=os.environ['LAB_ARTIFACTS']; "
    "open(os.path.join(a,'ckpt.txt'),'w').write('0.95'); "
    "json.dump({'score':0.95}, open(os.path.join(a,'metrics.json'),'w'))\""
)
# A "poisoned" run: checkpoint quality 0.10, but REPORTS 0.99 (the lie).
_TRAIN_POISONED = (
    "python3 -c \"import json,os; a=os.environ['LAB_ARTIFACTS']; "
    "open(os.path.join(a,'ckpt.txt'),'w').write('0.10'); "
    "json.dump({'score':0.99}, open(os.path.join(a,'metrics.json'),'w'))\""
)
# Independent evaluation: read the actual checkpoint, measure held-out score from IT.
_EVAL = (
    "python3 -c \"import json,os; "
    "s=float(open(os.path.join(os.environ['LAB_ARTIFACTS'],'ckpt.txt')).read()); "
    "json.dump({'score':s}, open(os.path.join(os.environ['LAB_EVAL_OUT'],'heldout.json'),'w'))\""
)

_CRITERION = Criterion(metric="score", op=">=", value=0.90, tolerance=0.0)


class DummyDatasetProvider:
    def fetch(self, ref: DatasetRef, dest: Path) -> None:
        (dest / "data.txt").write_text(f"dummy dataset for {ref.name}\n")


class DummyMetricExtractor:
    def extract(self, artifacts_dir: str) -> dict:
        p = Path(artifacts_dir) / "metrics.json"
        return json.loads(p.read_text()) if p.exists() else {}


def _contract(poisoned: bool) -> ExperimentContract:
    return ExperimentContract(
        success_definition="held-out score >= 0.90",
        gradable_criteria=[_CRITERION],
        framework=FrameworkSpec(name="dummy", version="0", cuda=""),
        datasets=[
            DatasetRef(name="dummy-train", source="dummy://train"),
            DatasetRef(name="dummy-heldout", source="dummy://heldout", held_out=True),
        ],
        command=_TRAIN_POISONED if poisoned else _TRAIN_GOOD,
        eval_command=_EVAL,
        budget=BudgetSpec(max_tokens=10_000, max_wall_s=60, max_retries=1),
        oracle=OracleRef(criterion=_CRITERION, source="dummy-oracle"),
        seed=0,
    )


class ScriptedPlanner:
    """Stand-in for the Agent SDK planner. Contract is keyed off the hypothesis text."""

    def propose_contract(self, rec: ExperimentRecord) -> tuple[ExperimentContract, Usage]:
        poisoned = "poison" in rec.hypothesis.lower()
        return _contract(poisoned), Usage(tokens_in=0, tokens_out=0)

    def decide_next(self, result: VerifiedResult, rec: ExperimentRecord
                    ) -> tuple[ExperimentRecord | None, Usage]:
        return None, Usage()  # self-test runs a single experiment per call


class DeterministicEvaluator:
    """Independent evaluator: separate context, measures on its own, distrusts reports."""

    def __init__(self, layout: Layout, job_runner: JobRunner, *, mode: str = "local",
                 session_id: str = "evaluator-deterministic"):
        self.layout = layout
        self.job_runner = job_runner
        self.mode = mode
        self.session_id = session_id

    def evaluate(self, rec: ExperimentRecord, contract: ExperimentContract,
                 artifacts_dir: str, lease: LeaseToken) -> tuple[VerifiedResult, Usage]:
        eval_out = self.layout.eval_out(rec.id)
        log_path = self.layout.logs(rec.id) / "eval.log"
        spec = JobSpec(
            exp_id=rec.id, command=contract.eval_command or "true",
            workdir=str(self.layout.workspace(rec.id)),
            artifacts_dir=artifacts_dir, eval_out_dir=str(eval_out),
            log_path=str(log_path), mode=self.mode,
        )
        result = self.job_runner.run(spec)
        measured = {}
        heldout = eval_out / "heldout.json"
        if heldout.exists():
            measured = json.loads(heldout.read_text())

        criterion = (contract.oracle.criterion if contract.oracle
                     else (contract.gradable_criteria[0] if contract.gradable_criteria else None))
        score = float(measured.get(criterion.metric)) if (criterion and criterion.metric in measured) else None
        passed = bool(result.ok and criterion is not None and score is not None
                      and criterion.satisfied(score))

        comparison = None
        if criterion is not None:
            comparison = {
                "metric": criterion.metric, "op": criterion.op,
                "expected": criterion.value, "tolerance": criterion.tolerance,
                "measured": score, "within": passed,
            }
        notes = ("evaluator ran eval_command on held-out and measured independently; "
                 f"reported={rec.reported_metrics}, measured={measured}")
        verdict = VerifiedResult(
            experiment_id=rec.id,
            verdict="PASS" if passed else "FAIL",
            measured_metrics=measured,
            oracle_comparison=comparison,
            provenance=Provenance(config_hash=rec.config_hash, image=rec.env_image,
                                  seed=contract.seed),
            evaluator_notes=notes,
            signed_by=self.session_id,
            signed_at=now_iso(),
        )
        return verdict, Usage(tokens_in=0, tokens_out=0)


def seed_experiment(hypothesis: str, *, priority: int = 0, exp_id: str | None = None
                    ) -> ExperimentRecord:
    return ExperimentRecord(
        id=exp_id or ("poison-001" if "poison" in hypothesis.lower() else "repro-001"),
        hypothesis=hypothesis, status=Status.PROPOSED, priority=priority,
    )
