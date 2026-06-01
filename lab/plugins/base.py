"""Plugin interfaces — the seams where domain logic and the LLM agents plug in.

Stage 1 (this skeleton) is LLM-free: the harness depends only on these Protocols, and
the dummy plugin (or, in Stage 2, Agent SDK sessions) satisfies them. This is what lets
the whole loop + calibration gate be proven deterministically before any model is wired.

Critical boundary (design §6): the Evaluator runs in a SEPARATE context from the
Planner/generator. It measures on the held-out split and never trusts reported metrics."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..gpu_lease import LeaseToken
from ..models import (
    DatasetRef, ExperimentContract, ExperimentRecord, OracleRef, Usage, VerifiedResult,
)


@runtime_checkable
class DatasetProviderP(Protocol):
    def fetch(self, ref: DatasetRef, dest: Path) -> None: ...


@runtime_checkable
class Oracle(Protocol):
    def expected(self, rec: ExperimentRecord) -> OracleRef: ...


@runtime_checkable
class MetricExtractor(Protocol):
    """Deterministic: turns artifacts into the generator's reported metrics (no LLM)."""
    def extract(self, artifacts_dir: str) -> dict: ...


@runtime_checkable
class Planner(Protocol):
    """The reasoning seat (Agent SDK in Stage 2). Returns Usage for token accounting."""
    def propose_contract(self, rec: ExperimentRecord) -> tuple[ExperimentContract, Usage]: ...

    def decide_next(
        self, result: VerifiedResult | None, rec: ExperimentRecord
    ) -> tuple[ExperimentRecord | None, Usage]: ...


@runtime_checkable
class Evaluator(Protocol):
    """Independent, skeptical, separate-context. Acquires the GPU lease and measures."""
    def evaluate(
        self, rec: ExperimentRecord, contract: ExperimentContract,
        artifacts_dir: str, lease: LeaseToken,
    ) -> tuple[VerifiedResult, Usage]: ...
