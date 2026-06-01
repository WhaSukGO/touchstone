"""Data models for the CV research lab harness.

Design refs (claudedocs/design_cv_lab_harness_skeleton_2026-06-01.md):
  §5 schemas, §6 evaluator boundary. The lab's only public output is a
  VerifiedResult signed by the independent evaluator — never a generator claim."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field


class Status(str, enum.Enum):
    PROPOSED = "PROPOSED"
    CONTRACTED = "CONTRACTED"
    ENV_READY = "ENV_READY"
    DATA_READY = "DATA_READY"
    RUNNING = "RUNNING"
    ARTIFACTS_READY = "ARTIFACTS_READY"
    EVALUATING = "EVALUATING"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


TERMINAL = {Status.VERIFIED, Status.REJECTED, Status.FAILED}


@dataclass
class Usage:
    """Token accounting for a single agent (LLM) call. IO does not produce Usage."""
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class Criterion:
    """A single gradable success condition. The backbone of 'not just runs, but works'."""
    metric: str
    op: str  # one of >= <= > < == ~=
    value: float
    tolerance: float = 0.0

    def satisfied(self, measured: float) -> bool:
        if self.op == ">=":
            return measured >= self.value - self.tolerance
        if self.op == "<=":
            return measured <= self.value + self.tolerance
        if self.op == ">":
            return measured > self.value - self.tolerance
        if self.op == "<":
            return measured < self.value + self.tolerance
        if self.op in ("==", "~="):
            return abs(measured - self.value) <= self.tolerance
        raise ValueError(f"unknown comparison op: {self.op!r}")


@dataclass
class FrameworkSpec:
    name: str            # e.g. "torch"
    version: str         # e.g. "2.4"
    cuda: str            # e.g. "12.1"


@dataclass
class DatasetRef:
    name: str
    source: str          # url / registry id / provider key
    sha256: str | None = None
    held_out: bool = False  # evaluator-only split; generator container never mounts it


@dataclass
class OracleRef:
    """Known-answer reference for reproduction / calibration."""
    criterion: Criterion
    source: str          # paper / reference impl / checkpoint id


@dataclass
class BudgetSpec:
    max_tokens: int
    max_wall_s: float
    max_retries: int


@dataclass
class ExperimentContract:
    """Negotiated BEFORE any run: the agreed definition of 'done' (sprint contract)."""
    success_definition: str
    gradable_criteria: list[Criterion] = field(default_factory=list)
    framework: FrameworkSpec | None = None
    datasets: list[DatasetRef] = field(default_factory=list)
    command: str = ""            # train/produce command (runs in container)
    eval_command: str | None = None  # evaluator's independent measurement command
    code_dir: str | None = None  # host dir of reference/experiment code, mounted ro at /code
    budget: BudgetSpec | None = None
    oracle: OracleRef | None = None
    seed: int = 0


@dataclass
class Provenance:
    config_hash: str
    image: str | None
    dataset_hashes: dict = field(default_factory=dict)
    seed: int = 0
    git_commit: str | None = None


@dataclass
class ArtifactRef:
    path: str
    sha256: str


@dataclass
class VerifiedResult:
    """The lab's ONLY official output. Measured by the independent evaluator,
    not reported by the generator. Carries provenance so other labs can trust it."""
    experiment_id: str
    verdict: str                 # "PASS" | "FAIL"
    measured_metrics: dict = field(default_factory=dict)
    oracle_comparison: dict | None = None
    artifacts: list[ArtifactRef] = field(default_factory=list)
    provenance: Provenance | None = None
    evaluator_notes: str = ""
    signed_by: str = ""          # evaluator session id (separate context)
    signed_at: str = ""


@dataclass
class ExperimentRecord:
    id: str
    hypothesis: str
    status: Status = Status.PROPOSED
    parent_id: str | None = None
    contract: ExperimentContract | None = None
    config_hash: str = ""
    env_image: str | None = None
    datasets: list[str] = field(default_factory=list)
    workdir: str | None = None
    log_path: str | None = None
    reported_metrics: dict = field(default_factory=dict)  # generator claim — recorded, not trusted
    verdict: VerifiedResult | None = None                 # evaluator-signed truth
    tokens_in: int = 0
    tokens_out: int = 0
    wall_seconds: float = 0.0
    retries: int = 0
    priority: int = 0
    created_at: str = ""
    updated_at: str = ""
