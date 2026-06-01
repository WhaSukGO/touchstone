"""CV Research Lab — deterministic harness skeleton (Stage 1).

See claudedocs/design_cv_lab_harness_skeleton_2026-06-01.md for the full spec."""
from __future__ import annotations

from .budget import Budget
from .dataset_cache import DatasetCache
from .factory import build_dummy_harness
from .gpu_lease import GpuLease
from .image_registry import ImageRegistry
from .job_runner import JobRunner
from .loop import Harness
from .models import (
    Criterion, ExperimentContract, ExperimentRecord, Status, Usage, VerifiedResult,
)
from .paths import Layout
from .queue import Queue
from .registry import Registry

__all__ = [
    "Budget", "DatasetCache", "GpuLease", "ImageRegistry", "JobRunner", "Harness",
    "Layout", "Queue", "Registry", "build_dummy_harness",
    "Criterion", "ExperimentContract", "ExperimentRecord", "Status", "Usage",
    "VerifiedResult",
]
