"""Dataset / weights cache (design §4.4).

Datasets are fetched ONCE in a lifetime and reused across all sessions. Fetching is a
harness job (zero tokens), never something an agent babysits turn-by-turn. The cache
records a manifest with content hashes; containers mount it read-only so experiments
can't corrupt it."""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from .models import DatasetRef
from .util import ensure_dir, sha256_dir


class DatasetProvider:
    """Domain plugin: knows how to materialize a dataset into `dest`. Injected."""

    def fetch(self, ref: DatasetRef, dest: Path) -> None:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class CacheEntry:
    name: str
    path: str
    sha256: str
    held_out: bool


class DatasetCache:
    def __init__(self, root: str | Path, provider: DatasetProvider):
        self.root = ensure_dir(root)
        self.provider = provider
        self.manifest_path = self.root / "manifest.json"
        self._lock = threading.Lock()
        self._manifest: dict[str, CacheEntry] = self._load()

    def _load(self) -> dict[str, CacheEntry]:
        if self.manifest_path.exists():
            raw = json.loads(self.manifest_path.read_text())
            return {k: CacheEntry(**v) for k, v in raw.items()}
        return {}

    def _save(self) -> None:
        self.manifest_path.write_text(
            json.dumps({k: asdict(v) for k, v in self._manifest.items()}, indent=2)
        )

    def ensure(self, ref: DatasetRef) -> CacheEntry:
        """Return cached path if present (and complete), else fetch once and record."""
        with self._lock:
            entry = self._manifest.get(ref.name)
            dest = self.root / ("heldout" if ref.held_out else "data") / ref.name
            marker = dest / ".complete"
            if entry and marker.exists():
                return entry

            ensure_dir(dest)
            self.provider.fetch(ref, dest)
            marker.write_text("ok")
            digest = sha256_dir(dest)
            if ref.sha256 and ref.sha256 != digest:
                raise ValueError(
                    f"dataset {ref.name} hash mismatch: expected {ref.sha256}, got {digest}"
                )
            entry = CacheEntry(name=ref.name, path=str(dest), sha256=digest,
                               held_out=ref.held_out)
            self._manifest[ref.name] = entry
            self._save()
            return entry

    def verify(self, name: str, *, deep: bool = False) -> bool:
        entry = self._manifest.get(name)
        if not entry:
            return False
        if not (Path(entry.path) / ".complete").exists():
            return False
        if deep:
            return sha256_dir(entry.path) == entry.sha256
        return True

    def manifest(self) -> dict[str, CacheEntry]:
        return dict(self._manifest)
