"""Budget accountant (design §4.6).

The lesson from the prior failed attempt: budget must be measured in TOKENS + experiment
count, never in 'turns'. IO waits (download/build/train) are recorded for observability
via note_io() but never charged — they must not consume the reasoning budget."""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class BudgetState:
    total_tokens: int = 0          # tokens spent (in+out) across all agent calls
    experiments_started: int = 0
    io_wall_seconds: float = 0.0   # observability only — does NOT affect budget
    per_experiment_io: dict = field(default_factory=dict)


class Budget:
    def __init__(self, *, max_total_tokens: int, max_experiments: int,
                 state_path: str | Path | None = None):
        self.max_total_tokens = max_total_tokens
        self.max_experiments = max_experiments
        self.state_path = Path(state_path) if state_path else None
        self._lock = threading.Lock()
        self.state = self._load()

    def _load(self) -> BudgetState:
        if self.state_path and self.state_path.exists():
            return BudgetState(**json.loads(self.state_path.read_text()))
        return BudgetState()

    def _persist(self) -> None:
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(asdict(self.state), indent=2))

    def charge_tokens(self, exp_id: str, tokens_in: int, tokens_out: int) -> None:
        with self._lock:
            self.state.total_tokens += tokens_in + tokens_out
            self._persist()

    def note_io(self, exp_id: str, wall_seconds: float) -> None:
        """Record IO wall time for visibility. Intentionally does not touch the budget."""
        with self._lock:
            self.state.io_wall_seconds += wall_seconds
            self.state.per_experiment_io[exp_id] = (
                self.state.per_experiment_io.get(exp_id, 0.0) + wall_seconds
            )
            self._persist()

    def mark_started(self) -> None:
        with self._lock:
            self.state.experiments_started += 1
            self._persist()

    def remaining_tokens(self) -> int:
        return max(0, self.max_total_tokens - self.state.total_tokens)

    def can_spawn(self) -> bool:
        return (self.remaining_tokens() > 0
                and self.state.experiments_started < self.max_experiments)
