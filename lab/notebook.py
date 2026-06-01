"""Lab notebook + failed-approaches log (design §4.7).

Append-only, human-readable. The failed log lets the planner avoid re-treading dead
ends (C-compiler lesson: maintain a running doc of failed approaches)."""
from __future__ import annotations

from pathlib import Path

from .models import ExperimentRecord
from .util import now_iso


class Notebook:
    def __init__(self, *, notebook_path: str | Path, failed_path: str | Path):
        self.notebook_path = Path(notebook_path)
        self.failed_path = Path(failed_path)
        for p in (self.notebook_path, self.failed_path):
            p.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, rec: ExperimentRecord, event: str) -> None:
        line = f"- {now_iso()} `{rec.id}` [{rec.status.value}] {event}\n"
        with self.notebook_path.open("a") as f:
            f.write(line)

    def log_failed(self, rec: ExperimentRecord, reason: str) -> None:
        line = f"- {now_iso()} `{rec.id}` | {rec.hypothesis} | {reason}\n"
        with self.failed_path.open("a") as f:
            f.write(line)

    def failed_summary(self, limit: int = 50) -> str:
        if not self.failed_path.exists():
            return ""
        return "\n".join(self.failed_path.read_text().splitlines()[-limit:])
