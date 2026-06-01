"""SQLite-backed Experiment Registry — the single source of truth.

Every state transition is committed here, so a crash or context reset resumes from
the last persisted state (design §3, §5.1). Stores each record as a JSON blob plus
indexed columns for queue ordering."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .models import ExperimentRecord, Status, TERMINAL
from .serde import from_dict, to_jsonable
from .util import now_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    priority    INTEGER NOT NULL DEFAULT 0,
    parent_id   TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    data        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_status ON experiments(status);
"""


class Registry:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def upsert(self, rec: ExperimentRecord) -> ExperimentRecord:
        if not rec.created_at:
            rec.created_at = now_iso()
        rec.updated_at = now_iso()
        blob = json.dumps(to_jsonable(rec))
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO experiments "
                "(id, status, priority, parent_id, created_at, updated_at, data) "
                "VALUES (?,?,?,?,?,?,?)",
                (rec.id, rec.status.value, rec.priority, rec.parent_id,
                 rec.created_at, rec.updated_at, blob),
            )
            self._conn.commit()
        return rec

    def get(self, exp_id: str) -> ExperimentRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM experiments WHERE id=?", (exp_id,)
            ).fetchone()
        if row is None:
            return None
        return from_dict(ExperimentRecord, json.loads(row["data"]))

    def query(self, *, statuses: list[Status] | None = None,
              limit: int | None = None) -> list[ExperimentRecord]:
        sql = "SELECT data FROM experiments"
        params: list = []
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            sql += f" WHERE status IN ({placeholders})"
            params.extend(s.value for s in statuses)
        sql += " ORDER BY priority DESC, created_at ASC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [from_dict(ExperimentRecord, json.loads(r["data"])) for r in rows]

    def interrupted(self) -> list[ExperimentRecord]:
        """Records left mid-flight by a crash (started but not terminal, not freshly proposed)."""
        in_flight = [s for s in Status
                     if s not in TERMINAL and s != Status.PROPOSED]
        return self.query(statuses=in_flight)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
