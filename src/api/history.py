"""
SQLite-backed audit log for simulation runs.

Every call to /simulate, /compare, /batch, /policy, /sensitivity, /benchmark
writes one row. The /history endpoint queries this table.

Configuration
-------------
  HISTORY_DB_PATH   filesystem path for the SQLite file (default :memory:)
                    Use a volume-mounted path in production for persistence.
  HISTORY_MAX_ROWS  row cap — oldest rows are pruned when exceeded (default 10000)

Schema
------
  runs(id, timestamp, endpoint, params_hash, params_json,
       duration_ms, status_code, cached, api_key_prefix)
"""

import os
import json
import time
import hashlib
import sqlite3
import logging
import threading
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_DB_PATH  = os.environ.get("HISTORY_DB_PATH", ":memory:")
_MAX_ROWS = int(os.environ.get("HISTORY_MAX_ROWS", "10000"))

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    endpoint        TEXT    NOT NULL,
    params_hash     TEXT,
    params_json     TEXT,
    duration_ms     REAL,
    status_code     INTEGER NOT NULL DEFAULT 200,
    cached          INTEGER NOT NULL DEFAULT 0,
    api_key_prefix  TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_endpoint  ON runs(endpoint);
CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(timestamp);
"""


def _params_hash(params: dict) -> str:
    raw = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class SimulationHistory:
    """
    Append-only run log with SQLite backend.

    All methods are thread-safe via an internal lock.
    """

    def __init__(self, db_path: str = _DB_PATH, max_rows: int = _MAX_ROWS):
        self._db_path = db_path
        self._max_rows = max_rows
        self._lock = threading.Lock()
        # Keep a single persistent connection so :memory: DBs survive across calls
        self._conn_obj = sqlite3.connect(db_path, check_same_thread=False)
        self._conn_obj.row_factory = sqlite3.Row
        self._init_db()
        logger.info("Simulation history DB: %s (max_rows=%d)", db_path, max_rows)

    def _conn(self) -> sqlite3.Connection:
        return self._conn_obj

    def _init_db(self) -> None:
        with self._lock:
            self._conn_obj.executescript(_CREATE_SQL)

    def record(
        self,
        endpoint: str,
        params: dict,
        duration_ms: float,
        status_code: int = 200,
        cached: bool = False,
        api_key: str = "",
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        ph = _params_hash(params)
        pj = json.dumps(params, default=str)
        kp = (api_key[:8] + "…") if api_key else None

        with self._lock:
            conn = self._conn()
            conn.execute(
                """INSERT INTO runs
                   (timestamp, endpoint, params_hash, params_json,
                    duration_ms, status_code, cached, api_key_prefix)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (ts, endpoint, ph, pj, round(duration_ms, 1),
                 status_code, int(cached), kp),
            )
            count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            if count > self._max_rows:
                overage = count - self._max_rows
                conn.execute(
                    "DELETE FROM runs WHERE id IN "
                    "(SELECT id FROM runs ORDER BY id ASC LIMIT ?)",
                    (overage,),
                )
            conn.commit()

    def recent(
        self,
        limit: int = 50,
        endpoint: Optional[str] = None,
        offset: int = 0,
    ) -> list[dict]:
        sql = """SELECT id, timestamp, endpoint, params_hash,
                        duration_ms, status_code, cached, api_key_prefix
                 FROM runs"""
        args: list = []
        if endpoint:
            sql += " WHERE endpoint = ?"
            args.append(endpoint)
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        args += [limit, offset]

        with self._lock:
            conn = self._conn()
            rows = conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: int) -> Optional[dict]:
        with self._lock:
            conn = self._conn()
            row = conn.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def aggregate_stats(self) -> dict:
        with self._lock:
            conn = self._conn()
            total = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            cached = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE cached=1"
            ).fetchone()[0]
            by_endpoint = conn.execute(
                """SELECT endpoint,
                          COUNT(*) as n,
                          ROUND(AVG(duration_ms), 1) as avg_ms,
                          ROUND(MIN(duration_ms), 1) as min_ms,
                          ROUND(MAX(duration_ms), 1) as max_ms
                   FROM runs
                   GROUP BY endpoint
                   ORDER BY n DESC"""
            ).fetchall()
            errors = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE status_code >= 400"
            ).fetchone()[0]
        return {
            "total_runs":    total,
            "cached_runs":   cached,
            "error_runs":    errors,
            "by_endpoint":   [dict(r) for r in by_endpoint],
        }


# Module-level singleton
simulation_history = SimulationHistory()
