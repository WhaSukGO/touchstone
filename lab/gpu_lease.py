"""Single-GPU lease (design §4.2).

One GPU = one mutex. Both the generator run and the independent evaluator run must
acquire it, which serializes all GPU work on this box. Implemented with flock: if the
holder process dies, the OS releases the lock automatically (stale-lease safety).

Note: flock is tied to the open file description, so a second acquire() from the SAME
process (a different fd) still blocks — correctly serializing evaluator vs generator."""
from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path


class LeaseTimeout(RuntimeError):
    pass


@dataclass
class LeaseToken:
    holder: str
    fd: int
    info_path: str


class GpuLease:
    def __init__(self, lock_path: str | Path, *, poll_s: float = 0.25):
        self.lock_path = Path(lock_path)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.info_path = self.lock_path.with_suffix(self.lock_path.suffix + ".info")
        self.poll_s = poll_s

    def acquire(self, holder: str, *, timeout_s: float = 3600.0) -> LeaseToken:
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    raise LeaseTimeout(
                        f"GPU lease not acquired within {timeout_s}s (held by another job)"
                    )
                time.sleep(self.poll_s)
        self.info_path.write_text(json.dumps(
            {"holder": holder, "pid": os.getpid(), "acquired_at": time.time()}
        ))
        return LeaseToken(holder=holder, fd=fd, info_path=str(self.info_path))

    def release(self, token: LeaseToken) -> None:
        try:
            fcntl.flock(token.fd, fcntl.LOCK_UN)
        finally:
            os.close(token.fd)
            try:
                self.info_path.unlink()
            except FileNotFoundError:
                pass

    def held_by(self) -> dict | None:
        if self.info_path.exists():
            try:
                return json.loads(self.info_path.read_text())
            except json.JSONDecodeError:
                return None
        return None


class _LeaseCtx:
    def __init__(self, lease: GpuLease, holder: str, timeout_s: float):
        self._lease, self._holder, self._timeout = lease, holder, timeout_s
        self._token: LeaseToken | None = None

    def __enter__(self) -> LeaseToken:
        self._token = self._lease.acquire(self._holder, timeout_s=self._timeout)
        return self._token

    def __exit__(self, *exc) -> None:
        if self._token:
            self._lease.release(self._token)


def hold(lease: GpuLease, holder: str, *, timeout_s: float = 3600.0) -> _LeaseCtx:
    """`with hold(lease, "exp-001:train"): ...`"""
    return _LeaseCtx(lease, holder, timeout_s)
