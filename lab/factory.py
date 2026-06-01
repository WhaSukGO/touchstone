"""Assembly helpers. Stage 1 wires the deterministic dummy plugin so the entire harness
+ calibration gate runs with no GPU, Docker, or model. In Stage 2, swap ScriptedPlanner
and DeterministicEvaluator for Agent SDK sessions; nothing else changes."""
from __future__ import annotations

from pathlib import Path

from .budget import Budget
from .dataset_cache import DatasetCache
from .evaluator import ScriptEvaluator
from .gpu_lease import GpuLease
from .image_registry import ImageRegistry
from .job_runner import JobRunner
from .loop import Harness
from .notebook import Notebook
from .paths import Layout
from .plugins.cifar import CifarDatasetProvider, CifarMetricExtractor, CifarPlanner
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


def build_cifar_harness(root: str | Path, *, job_mode: str = "docker",
                        images_path: str | Path = "images/registry.yaml",
                        max_total_tokens: int = 1_000_000,
                        max_experiments: int = 100,
                        lease_timeout_s: float = 1800.0) -> Harness:
    """Stage 2 calibration harness: real CIFAR-10 on the GPU via Docker+torch, verified
    by the independent ScriptEvaluator on the held-out split."""
    layout = Layout(Path(root))
    ensure_dir(layout.state)
    registry = Registry(layout.registry_db)
    queue = Queue(registry)
    gpu_lease = GpuLease(layout.gpu_lock)
    image_registry = ImageRegistry(images_path)
    dataset_cache = DatasetCache(layout.cache, CifarDatasetProvider(layout.cache / "raw"))
    job_runner = JobRunner(default_mode=job_mode)
    budget = Budget(max_total_tokens=max_total_tokens, max_experiments=max_experiments,
                    state_path=layout.budget_state)
    notebook = Notebook(notebook_path=layout.notebook, failed_path=layout.failed)
    evaluator = ScriptEvaluator(layout, job_runner, dataset_cache, image_registry,
                                mode=job_mode, session_id="evaluator-cifar")
    return Harness(
        layout=layout, registry=registry, queue=queue, gpu_lease=gpu_lease,
        image_registry=image_registry, dataset_cache=dataset_cache,
        job_runner=job_runner, budget=budget, notebook=notebook,
        planner=CifarPlanner(), evaluator=evaluator,
        metric_extractor=CifarMetricExtractor(), job_mode=job_mode,
        lease_timeout_s=lease_timeout_s,
    )


def build_cifar_agent_harness(root: str | Path, *, job_mode: str = "docker",
                              images_path: str | Path = "images/registry.yaml",
                              model: str | None = None,
                              max_total_tokens: int = 2_000_000,
                              max_experiments: int = 100,
                              lease_timeout_s: float = 1800.0) -> Harness:
    """CIFAR domain wired with the REAL Agent SDK: SdkPlanner proposes research
    experiments and the skeptical SdkEvaluator (separate context) judges on top of the
    deterministic held-out measurement. Calibration records carry a fixed contract, so
    the gate still uses the reference scripts. Live use needs ANTHROPIC_API_KEY (billed)."""
    from .agents import DEFAULT_MODEL, SdkEvaluator, SdkPlanner

    h = build_cifar_harness(root, job_mode=job_mode, images_path=images_path,
                            max_total_tokens=max_total_tokens,
                            max_experiments=max_experiments, lease_timeout_s=lease_timeout_s)
    m = model or DEFAULT_MODEL
    h.planner = SdkPlanner(model=m)
    h.evaluator = SdkEvaluator(h.evaluator, model=m)  # wrap the deterministic ScriptEvaluator
    return h


def build_implementer_harness(root: str | Path, task, author_fn, *, job_mode: str = "local",
                              images_path: str | Path = "images/registry.yaml",
                              max_total_tokens: int = 2_000_000, max_experiments: int = 20,
                              lease_timeout_s: float = 1800.0) -> Harness:
    """Stage 7: experiments whose code is AUTHORED by an Implementer, then graded by the
    unchanged independent ScriptEvaluator on held-out vs the task's fixed oracle. The
    author step is injected so the verification handoff is testable offline."""
    from .agents.implementer import Implementer

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
    evaluator = ScriptEvaluator(layout, job_runner, dataset_cache, image_registry,
                                mode=job_mode, session_id="evaluator-impl")
    return Harness(
        layout=layout, registry=registry, queue=queue, gpu_lease=gpu_lease,
        image_registry=image_registry, dataset_cache=dataset_cache,
        job_runner=job_runner, budget=budget, notebook=notebook,
        planner=Implementer(task, layout, author_fn=author_fn), evaluator=evaluator,
        metric_extractor=DummyMetricExtractor(), job_mode=job_mode,
        lease_timeout_s=lease_timeout_s,
    )


def build_cifar_committee_harness(root: str | Path, *, job_mode: str = "docker",
                                  images_path: str | Path = "images/registry.yaml",
                                  model: str | None = None,
                                  max_total_tokens: int = 4_000_000,
                                  max_experiments: int = 100,
                                  lease_timeout_s: float = 1800.0) -> Harness:
    """Stage 3: experiments proposed by an expert COMMITTEE constrained to the CIFAR menu
    (it can only select+parameterize a vetted recipe), judged by the skeptical
    SdkEvaluator. Calibration records still carry fixed contracts. Live use is billed."""
    from .agents import DEFAULT_MODEL, Committee, SdkEvaluator
    from .history import ResearchHistory
    from .plugins.cifar import cifar_menu

    h = build_cifar_harness(root, job_mode=job_mode, images_path=images_path,
                            max_total_tokens=max_total_tokens,
                            max_experiments=max_experiments, lease_timeout_s=lease_timeout_s)
    m = model or DEFAULT_MODEL
    h.planner = Committee(cifar_menu(), model=m, notebook=h.notebook,
                          history=ResearchHistory(h.registry))
    h.evaluator = SdkEvaluator(h.evaluator, model=m)
    return h
