"""Generic independent evaluator (design §6).

Separate context from the generator. It mounts the HELD-OUT split (which the generator
never sees), the produced checkpoint, and the reference code, runs the contract's
eval_command, and measures the metric itself — never trusting reported_metrics. Used by
real domain plugins (e.g. CIFAR). The dummy plugin keeps its own tiny variant."""
from __future__ import annotations

import json
from pathlib import Path

from .dataset_cache import DatasetCache
from .gpu_lease import LeaseToken
from .image_registry import ImageRegistry, NoImageError
from .job_runner import JobRunner, JobSpec
from .models import ExperimentContract, ExperimentRecord, Provenance, Usage, VerifiedResult
from .paths import Layout
from .util import now_iso


class ScriptEvaluator:
    def __init__(self, layout: Layout, job_runner: JobRunner, dataset_cache: DatasetCache,
                 image_registry: ImageRegistry, *, mode: str = "docker",
                 session_id: str = "evaluator-script"):
        self.layout = layout
        self.job_runner = job_runner
        self.dataset_cache = dataset_cache
        self.image_registry = image_registry
        self.mode = mode
        self.session_id = session_id

    def _heldout_dir(self, contract: ExperimentContract) -> str | None:
        held = [d for d in contract.datasets if d.held_out]
        if not held:
            return None
        entry = self.dataset_cache.manifest().get(held[0].name)
        return entry.path if entry else None

    def evaluate(self, rec: ExperimentRecord, contract: ExperimentContract,
                 artifacts_dir: str, lease: LeaseToken) -> tuple[VerifiedResult, Usage]:
        image = None
        if self.mode == "docker" and contract.framework is not None:
            try:
                image = self.image_registry.resolve(contract.framework).image
            except NoImageError:
                image = None

        eval_out = self.layout.eval_out(rec.id)
        log_path = self.layout.logs(rec.id) / "eval.log"
        result = self.job_runner.run(JobSpec(
            exp_id=rec.id, command=contract.eval_command or "true",
            workdir=str(self.layout.workspace(rec.id)),
            artifacts_dir=artifacts_dir, eval_out_dir=str(eval_out),
            data_dir=self._heldout_dir(contract), code_dir=contract.code_dir,
            log_path=str(log_path), image=image, mode=self.mode,
        ))

        measured: dict = {}
        heldout = eval_out / "heldout.json"
        if heldout.exists():
            measured = json.loads(heldout.read_text())

        criterion = (contract.oracle.criterion if contract.oracle
                     else (contract.gradable_criteria[0] if contract.gradable_criteria else None))
        score = None
        if criterion and criterion.metric in measured:
            score = float(measured[criterion.metric])
        passed = bool(result.ok and criterion is not None and score is not None
                      and criterion.satisfied(score))

        comparison = None
        if criterion is not None:
            comparison = {
                "metric": criterion.metric, "op": criterion.op,
                "expected": criterion.value, "tolerance": criterion.tolerance,
                "measured": score, "within": passed,
            }
        ds_hashes = {n: e.sha256 for n, e in self.dataset_cache.manifest().items()}
        verdict = VerifiedResult(
            experiment_id=rec.id,
            verdict="PASS" if passed else "FAIL",
            measured_metrics=measured,
            oracle_comparison=comparison,
            provenance=Provenance(config_hash=rec.config_hash, image=image,
                                  dataset_hashes=ds_hashes, seed=contract.seed),
            evaluator_notes=(f"independent eval on held-out; exit={result.exit_code}; "
                             f"reported={rec.reported_metrics}, measured={measured}"),
            signed_by=self.session_id, signed_at=now_iso(),
        )
        return verdict, Usage(tokens_in=0, tokens_out=0)
