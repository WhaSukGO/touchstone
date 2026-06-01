"""Experiment queue — a thin priority view over the Registry (design §4.1).

The Registry is the single source of truth; the queue just selects what to work on
next. Priority order: calibration > user-pinned > planner-proposed."""
from __future__ import annotations

from .models import ExperimentRecord, Status
from .registry import Registry


class Queue:
    def __init__(self, registry: Registry):
        self.registry = registry

    def push(self, rec: ExperimentRecord) -> ExperimentRecord:
        return self.registry.upsert(rec)

    def pop_next(self) -> ExperimentRecord | None:
        """Highest-priority PROPOSED experiment (priority desc, then oldest first)."""
        rows = self.registry.query(statuses=[Status.PROPOSED], limit=1)
        return rows[0] if rows else None

    def interrupted(self) -> list[ExperimentRecord]:
        """In-flight experiments to resume before starting new ones (crash recovery)."""
        return self.registry.interrupted()

    def requeue(self, rec: ExperimentRecord, *, priority: int | None = None) -> None:
        if priority is not None:
            rec.priority = priority
        rec.status = Status.PROPOSED
        self.registry.upsert(rec)
