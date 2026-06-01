"""Inter-lab collaboration (Stage 5).

A lab's only export is a VerifiedResult signed by its independent evaluator. Other labs
consume these through a trust gate (signed + provenance + PASS) — never a raw generator
claim. The strongest collaboration primitive is cross-lab PEER REVIEW: a second lab
independently re-measures a published artifact with its own evaluator (separate context
and registry) before the result is trusted downstream. This extends the generator !=
evaluator principle across lab boundaries — even a lab's own evaluator can be wrong or
compromised, so an independent lab re-checks."""
from __future__ import annotations

from dataclasses import dataclass

from .gpu_lease import hold
from .models import ExperimentContract, ExperimentRecord, VerifiedResult


def is_trustworthy(result: VerifiedResult | None) -> tuple[bool, str]:
    """The trust gate a consuming lab applies before relying on a foreign result."""
    if result is None:
        return False, "no result"
    if result.verdict != "PASS":
        return False, f"verdict is {result.verdict}, not PASS"
    if not result.signed_by:
        return False, "unsigned (no evaluator signature)"
    if result.provenance is None:
        return False, "missing provenance"
    return True, "ok"


@dataclass
class Publication:
    """What a lab publishes: the signed result plus what a reviewer needs to re-verify it."""
    lab: str
    result: VerifiedResult
    contract: ExperimentContract
    artifact_dir: str


class ResultExchange:
    """Shared store labs publish to and consume from. Refuses untrustworthy results."""

    def __init__(self):
        self._pubs: dict[str, Publication] = {}

    def publish(self, pub: Publication) -> None:
        ok, reason = is_trustworthy(pub.result)
        if not ok:
            raise ValueError(f"refuse to publish untrustworthy result: {reason}")
        self._pubs[pub.result.experiment_id] = pub

    def get(self, experiment_id: str) -> Publication | None:
        return self._pubs.get(experiment_id)

    def all(self) -> list[Publication]:
        return list(self._pubs.values())

    def best(self, metric: str) -> Publication | None:
        best: tuple[Publication, float] | None = None
        for p in self._pubs.values():
            v = p.result.measured_metrics.get(metric)
            if v is not None and (best is None or float(v) > best[1]):
                best = (p, float(v))
        return best[0] if best else None


@dataclass
class PeerReview:
    reviewer: str
    original_lab: str
    experiment_id: str
    metric: str
    claimed: float | None
    measured: float | None
    agree: bool
    reviewer_verdict: VerifiedResult
    note: str = ""


def _metric_of(contract: ExperimentContract) -> str:
    if contract.oracle:
        return contract.oracle.criterion.metric
    if contract.gradable_criteria:
        return contract.gradable_criteria[0].metric
    return ""


def peer_review(reviewer, pub: Publication, *, tolerance: float = 0.05) -> PeerReview:
    """`reviewer` (a Lab) independently re-measures the published artifact with its own
    evaluator and compares to the claim. Returns CONFIRMED only if the re-measured metric
    matches within tolerance AND the verdict agrees."""
    metric = _metric_of(pub.contract)
    rec = ExperimentRecord(
        id=f"review-{pub.result.experiment_id}",
        hypothesis=f"peer review of {pub.lab}/{pub.result.experiment_id}",
        reported_metrics=pub.result.measured_metrics,  # the claim under review, untrusted
    )
    # the reviewer needs its OWN copy of the held-out split to measure independently
    for ds in pub.contract.datasets:
        if ds.held_out:
            reviewer.harness.dataset_cache.ensure(ds)
    with hold(reviewer.harness.gpu_lease, f"peer-review:{rec.id}",
              timeout_s=reviewer.harness.lease_timeout_s) as token:
        verdict, _ = reviewer.harness.evaluator.evaluate(
            rec, pub.contract, pub.artifact_dir, token)

    claimed = pub.result.measured_metrics.get(metric)
    measured = verdict.measured_metrics.get(metric)
    agree = (claimed is not None and measured is not None
             and abs(float(claimed) - float(measured)) <= tolerance
             and verdict.verdict == pub.result.verdict)
    note = ("confirmed" if agree else
            f"DISPUTED: {pub.lab} claimed {metric}={claimed}, "
            f"{reviewer.name} measured {measured}")
    return PeerReview(reviewer=reviewer.name, original_lab=pub.lab,
                      experiment_id=pub.result.experiment_id, metric=metric,
                      claimed=claimed, measured=measured, agree=agree,
                      reviewer_verdict=verdict, note=note)
