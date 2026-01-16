from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List


# -------------------------------------------------
# Utils
# -------------------------------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)


def _id() -> str:
    return uuid.uuid4().hex


# -------------------------------------------------
# ChatStateRepo (SQLite)
# -------------------------------------------------

class SqliteChatStateRepo:
    """
    Persiste el estado transitorio del chat:
    - pending_run_id
    - active_run_id
    - last_run_id
    - last_run_status
    - last_run_ts
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_state (
                    chat_id TEXT PRIMARY KEY,
                    pending_run_id TEXT,
                    active_run_id TEXT,
                    last_run_id TEXT,
                    last_run_status TEXT,
                    last_run_ts INTEGER,
                    updated_ts INTEGER
                )
                """
            )
            c.commit()

    def get_state(self, chat_id: str) -> Dict[str, Any]:
        """
        Devuelve SOLO keys con valor != None para evitar confusiones de guard.
        """
        with self._lock, self._conn() as c:
            row = c.execute(
                "SELECT * FROM chat_state WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()

            if not row:
                return {}

            data = dict(row)
            data.pop("chat_id", None)
            data.pop("updated_ts", None)

            # ✅ evita pending_run_id=None "fantasma"
            return {k: v for k, v in data.items() if v is not None}

    def set_state(self, chat_id: str, **kwargs: Any) -> None:
        allowed = {
            "pending_run_id",
            "active_run_id",
            "last_run_id",
            "last_run_status",
            "last_run_ts",
        }

        data = {k: v for k, v in kwargs.items() if k in allowed}
        if not data:
            return

        data["updated_ts"] = _now_ms()

        cols = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        updates = ", ".join([f"{k}=excluded.{k}" for k in data.keys()])

        values = list(data.values())

        with self._lock, self._conn() as c:
            c.execute(
                f"""
                INSERT INTO chat_state (chat_id, {cols})
                VALUES (?, {placeholders})
                ON CONFLICT(chat_id) DO UPDATE SET {updates}
                """,
                [chat_id, *values],
            )
            c.commit()


# -------------------------------------------------
# PlanRunState (modelo persistido)
# -------------------------------------------------

@dataclass
class PlanRunState:
    run_id: str
    chat_id: str
    plan_id: str
    goal: str

    status: str
    created_ts: int
    updated_ts: int

    current_step_path: Optional[str] = None
    last_event: Optional[str] = None
    plan: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# -------------------------------------------------
# PlanRunStore (SQLite)
# -------------------------------------------------

class SqlitePlanRunStore:
    """
    Store persistente de runs de planes.
    Reemplaza al PlanRunStore en RAM.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS plan_runs (
                    run_id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_ts INTEGER NOT NULL,
                    updated_ts INTEGER NOT NULL,
                    current_step_path TEXT,
                    last_event TEXT,
                    plan_json TEXT,
                    error TEXT
                )
                """
            )
            c.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_plan_runs_chat_status_updated
                ON plan_runs(chat_id, status, updated_ts)
                """
            )
            c.commit()

    # ---------------- Create ----------------

    def create(self, *, chat_id: str, plan_id: str, goal: str) -> PlanRunState:
        run_id = _id()
        now = _now_ms()

        with self._lock, self._conn() as c:
            c.execute(
                """
                INSERT INTO plan_runs (
                    run_id, chat_id, plan_id, goal,
                    status, created_ts, updated_ts
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    chat_id,
                    plan_id,
                    goal,
                    "draft",
                    now,
                    now,
                ),
            )
            c.commit()

        return PlanRunState(
            run_id=run_id,
            chat_id=chat_id,
            plan_id=plan_id,
            goal=goal,
            status="draft",
            created_ts=now,
            updated_ts=now,
        )

    # ---------------- Read ----------------

    def get(self, run_id: str) -> Optional[PlanRunState]:
        with self._lock, self._conn() as c:
            row = c.execute(
                "SELECT * FROM plan_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()

            return self._row_to_state(row) if row else None

    def get_latest_draft_run_id(self, chat_id: str) -> Optional[str]:
        with self._lock, self._conn() as c:
            row = c.execute(
                """
                SELECT run_id FROM plan_runs
                WHERE chat_id = ? AND status = 'draft'
                ORDER BY updated_ts DESC
                LIMIT 1
                """,
                (chat_id,),
            ).fetchone()

            return row["run_id"] if row else None

    # ✅ usado por routes.py (recovery)
    def list_all(self) -> List[PlanRunState]:
        with self._lock, self._conn() as c:
            rows = c.execute(
                "SELECT * FROM plan_runs ORDER BY updated_ts DESC"
            ).fetchall()
            return [self._row_to_state(r) for r in rows]

    # (opcional, útil para debug)
    def list_by_chat(self, chat_id: str, limit: int = 50) -> List[PlanRunState]:
        with self._lock, self._conn() as c:
            rows = c.execute(
                """
                SELECT * FROM plan_runs
                WHERE chat_id = ?
                ORDER BY updated_ts DESC
                LIMIT ?
                """,
                (chat_id, int(limit)),
            ).fetchall()
            return [self._row_to_state(r) for r in rows]

    # ✅ usado por routes.py (GET /api/runs/{id})
    def to_dict(self, r: PlanRunState) -> Dict[str, Any]:
        d = asdict(r)
        # limpieza opcional: no devolver None para UI
        return {k: v for k, v in d.items() if v is not None}

    # ---------------- Update ----------------

    def update(
        self,
        run_id: str,
        *,
        status: Optional[str] = None,
        current_step_path: Optional[str] = None,
        last_event: Optional[str] = None,
        plan: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> bool:
        fields: Dict[str, Any] = {"updated_ts": _now_ms()}

        if status is not None:
            fields["status"] = status
        if current_step_path is not None:
            fields["current_step_path"] = current_step_path
        if last_event is not None:
            fields["last_event"] = last_event
        if plan is not None:
            fields["plan_json"] = json.dumps(plan, ensure_ascii=False)
        if error is not None:
            fields["error"] = error

        sets = ", ".join([f"{k}=?" for k in fields.keys()])
        values = list(fields.values())

        with self._lock, self._conn() as c:
            cur = c.execute(
                f"""
                UPDATE plan_runs
                SET {sets}
                WHERE run_id = ?
                """,
                (*values, run_id),
            )
            c.commit()
            return cur.rowcount == 1

    # ---------------- CAS (confirmación) ----------------

    def try_mark_queued(self, run_id: str) -> bool:
        with self._lock, self._conn() as c:
            cur = c.execute(
                """
                UPDATE plan_runs
                SET status = 'queued',
                    updated_ts = ?,
                    last_event = 'confirm_accepted'
                WHERE run_id = ? AND status = 'draft'
                """,
                (_now_ms(), run_id),
            )
            c.commit()
            return cur.rowcount == 1

    # ---------------- Helpers ----------------

    def _row_to_state(self, row: sqlite3.Row) -> PlanRunState:
        plan = None
        if row["plan_json"]:
            try:
                plan = json.loads(row["plan_json"])
            except Exception:
                plan = None

        return PlanRunState(
            run_id=row["run_id"],
            chat_id=row["chat_id"],
            plan_id=row["plan_id"],
            goal=row["goal"],
            status=row["status"],
            created_ts=row["created_ts"],
            updated_ts=row["updated_ts"],
            current_step_path=row["current_step_path"],
            last_event=row["last_event"],
            plan=plan,
            error=row["error"],
        )
