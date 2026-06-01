"""Logging policy (design §4.7, C-compiler lesson: don't print thousands of useless bytes).

Full logs live on disk under logs/<exp_id>/. Agents only ever see a compact summary:
the tail + standardized ERROR lines + aggregates. Keeps context clean and cheap."""
from __future__ import annotations

from pathlib import Path

ERROR_PREFIX = "ERROR:"


def error_line(reason: str) -> str:
    """Standardized one-line error so summaries are greppable."""
    return f"{ERROR_PREFIX} {reason.strip().splitlines()[0] if reason.strip() else reason}"


def tail(path: str | Path, n: int = 40) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    lines = p.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def summarize_log(path: str | Path, *, tail_lines: int = 40) -> str:
    """What an agent is allowed to see: error lines + the tail. Never the full dump."""
    p = Path(path)
    if not p.exists():
        return "(no log)"
    text = p.read_text(errors="replace")
    errors = [ln for ln in text.splitlines() if ERROR_PREFIX in ln]
    parts = []
    if errors:
        parts.append("== errors ==\n" + "\n".join(errors[-20:]))
    parts.append("== tail ==\n" + "\n".join(text.splitlines()[-tail_lines:]))
    return "\n".join(parts)
