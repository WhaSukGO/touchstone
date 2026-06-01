"""Lab — a composable research unit (Stage 5).

Wraps a Harness behind a clean boundary: a lab takes a hypothesis and produces signed
VerifiedResults. That signed result (with provenance) is the ONLY thing other labs
consume — never a raw generator claim — which is what lets labs collaborate and trust
each other's work (design §6.2)."""
from __future__ import annotations

from .exchange import Publication, is_trustworthy
from .loop import Harness
from .models import ExperimentRecord, Status, VerifiedResult


class Lab:
    def __init__(self, name: str, harness: Harness):
        self.name = name
        self.harness = harness

    # --- producing work ------------------------------------------------------
    def run_one(self, hypothesis: str, *, exp_id: str) -> ExperimentRecord:
        """Run a single experiment to a terminal state (bypasses the autonomy gate)."""
        return self.harness.run_experiment(
            ExperimentRecord(id=exp_id, hypothesis=hypothesis))

    def pursue(self, hypothesis: str, *, exp_id: str, goal_metric: str | None = None,
               max_stall: int | None = None, require_gate: bool = True) -> dict:
        """Run an autonomous lineage toward a goal (Stage 4 loop)."""
        self.harness.queue.push(ExperimentRecord(id=exp_id, hypothesis=hypothesis))
        return self.harness.loop(require_gate=require_gate, goal_metric=goal_metric,
                                 max_stall=max_stall)

    # --- publishing / consuming ---------------------------------------------
    def published(self) -> list[VerifiedResult]:
        return [r.verdict for r in self.harness.registry.query(statuses=[Status.VERIFIED])
                if r.verdict]

    def publish(self, exp_id: str) -> Publication:
        rec = self.harness.registry.get(exp_id)
        if rec is None or rec.verdict is None or rec.contract is None:
            raise ValueError(f"{exp_id}: no verified result to publish")
        ok, reason = is_trustworthy(rec.verdict)
        if not ok:
            raise ValueError(f"{exp_id}: not publishable ({reason})")
        return Publication(lab=self.name, result=rec.verdict, contract=rec.contract,
                           artifact_dir=str(self.harness.layout.artifacts(exp_id)))

    def accept(self, pub: Publication) -> bool:
        """Trust gate before consuming a foreign result as input."""
        ok, _ = is_trustworthy(pub.result)
        return ok
