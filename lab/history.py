"""Research history (Stage 4 memory).

The autonomous loop runs a lineage of experiments; without memory the committee would
re-tread the same configs (the C-compiler post's lesson: keep a running doc of what was
tried). This reads the registry and produces a compact digest the agents can reason over,
plus the best result so far on a metric."""
from __future__ import annotations

from .models import ExperimentRecord
from .registry import Registry


class ResearchHistory:
    def __init__(self, registry: Registry):
        self.registry = registry

    def all(self) -> list[ExperimentRecord]:
        return sorted(self.registry.query(), key=lambda r: r.created_at)

    def summary(self, *, limit: int = 15) -> str:
        recs = self.all()
        if not recs:
            return "(no prior experiments yet)"
        lines = []
        for r in recs[-limit:]:
            cmd = r.contract.command if r.contract else "?"
            measured = r.verdict.measured_metrics if r.verdict else {}
            verdict = r.verdict.verdict if r.verdict else "-"
            concerns = ""
            if r.verdict and r.verdict.evaluator_notes:
                # surface only a short tail of the evaluator's reasoning
                concerns = " | note: " + r.verdict.evaluator_notes.strip()[-160:]
            lines.append(f"- {r.id}: {r.status.value}/{verdict} measured={measured} "
                         f"cmd=({cmd}){concerns}")
        return "\n".join(lines)

    def best(self, metric: str) -> tuple[str, float] | None:
        best: tuple[str, float] | None = None
        for r in self.all():
            if r.verdict and r.verdict.verdict == "PASS":
                v = r.verdict.measured_metrics.get(metric)
                if v is not None and (best is None or float(v) > best[1]):
                    best = (r.id, float(v))
        return best

    def tried_config_hashes(self) -> set[str]:
        return {r.config_hash for r in self.all() if r.config_hash}
