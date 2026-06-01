"""Assembly helpers. Stage 1 wires the deterministic dummy plugin so the entire harness
+ calibration gate runs with no GPU, Docker, or model. In Stage 2, swap ScriptedPlanner
and DeterministicEvaluator for Agent SDK sessions; nothing else changes."""
from __future__ import annotations

from pathlib import Path

from .budget import Budget
from .dataset_cache import DatasetCache
from .gpu_lease import GpuLease
from .image_registry import ImageRegistry
from .job_runner import JobRunner
from .loop import Harness
from .notebook import Notebook
from .paths import Layout
from .plugins.dummy import (
    DeterministicEvaluator, DummyDatasetProvider, DummyMetricExtractor, ScriptedPlanner,
)
from .queue import Queue
from .registry import Registry
from .util import ensure_dir


def build_dummy_harness(root: str | Path, *, job_mode: str = "local",
                        images_path: str | Path = "images/registry.yaml",
                        max_total_tokens: int = 1_000_000,
                        max_experiments: int = 100) -> Harness:
    layout = Layout(Path(root))
    ensure_dir(layout.state)
    registry = Registry(layout.registry_db)
    queue = Queue(registry)
    gpu_lease = GpuLease(layout.gpu_lock)
    image_registry = ImageRegistry(images_path)
    dataset_cache = DatasetCache(layout.cache, DummyDatasetProvider())
    job_runner = JobRunner(default_mode=job_mode)
    budget = Budget(max_total_tokens=max_total_tokens, max_experiments=max_experiments,
                    state_path=layout.budget_state)
    notebook = Notebook(notebook_path=layout.notebook, failed_path=layout.failed)
    evaluator = DeterministicEvaluator(layout, job_runner, mode=job_mode)
    return Harness(
        layout=layout, registry=registry, queue=queue, gpu_lease=gpu_lease,
        image_registry=image_registry, dataset_cache=dataset_cache,
        job_runner=job_runner, budget=budget, notebook=notebook,
        planner=ScriptedPlanner(), evaluator=evaluator,
        metric_extractor=DummyMetricExtractor(), job_mode=job_mode,
    )
