"""Filesystem layout (design §9). One place that defines where everything lives so the
loop and the independent evaluator agree on conventions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Layout:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    # state
    @property
    def state(self) -> Path: return self.root / "state"
    @property
    def registry_db(self) -> Path: return self.state / "registry.db"
    @property
    def gpu_lock(self) -> Path: return self.state / "gpu.lock"
    @property
    def budget_state(self) -> Path: return self.state / "budget.json"
    @property
    def notebook(self) -> Path: return self.state / "lab_notebook.md"
    @property
    def failed(self) -> Path: return self.state / "failed_approaches.md"

    # caches & per-experiment dirs
    @property
    def cache(self) -> Path: return self.root / "cache"

    def logs(self, exp_id: str) -> Path: return self.root / "logs" / exp_id
    def workspace(self, exp_id: str) -> Path: return self.root / "workspaces" / exp_id
    def artifacts(self, exp_id: str) -> Path: return self.workspace(exp_id) / "artifacts"
    def eval_out(self, exp_id: str) -> Path: return self.workspace(exp_id) / "eval"
