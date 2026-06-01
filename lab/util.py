"""Deterministic helpers used across the harness. No LLM, no side effects beyond hashing/time."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    """Wall-clock timestamp (UTC, ISO-8601). The harness owns the clock, not the agents."""
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash(obj: Any) -> str:
    """Order-independent hash of a JSON-able object. Used for config_hash."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_bytes(payload.encode("utf-8"))


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(path: str | os.PathLike[str]) -> str:
    """Deterministic hash of a directory tree (sorted relpaths + file bytes).

    Cheap-enough for small datasets; for large corpora prefer trusting the manifest
    marker rather than re-hashing (see DatasetCache.verify deep=False)."""
    root = Path(path)
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            h.update(sha256_file(p).encode("ascii"))
            h.update(b"\0")
    return h.hexdigest()


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
