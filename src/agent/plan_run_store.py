from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, List
from src.observability.logger import get_logger
import threading
import time
import uuid

log = get_logger("-")


def _id() -> str:
    return uuid.uuid4().hex


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class PlanRunState:
    run_id: str
    chat_id: str
    plan_id: str
    goal: str

    status: str = "draft"
    created_ts: int = field(default_factory=_now_ms)
    updated_ts: int = field(default_factory=_now_ms)

    current_step_path: Optional[str] = None
    last_event: Optional[str] = None

    plan: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PlanRunStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: Dict[str, PlanRunState] = {}

    def create(self, *, chat_id: str, plan_id: str, goal: str) -> PlanRunState:
        r = PlanRunState(run_id=_id(), chat_id=chat_id, plan_id=plan_id, goal=goal)
        with self._lock:
            self._runs[r.run_id] = r

        log.info(
            "event=planrun.create chat_id=%s run_id=%s status=%s plan_id=%s",
            chat_id,
            r.run_id,
            r.status,
            plan_id,
        )
        return r


    def get(self, run_id: str) -> Optional[PlanRunState]:
        with self._lock:
            return self._runs.get(run_id)

        
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
        with self._lock:
            r = self._runs.get(run_id)
            if not r:
                log.info("event=planrun.update run_id=%s ok=false reason=not_found", run_id)
                return False

            before_status = r.status
            before_event = r.last_event
            before_step = r.current_step_path
            had_error_before = bool(r.error)

            if status is not None:
                r.status = status
            if current_step_path is not None:
                r.current_step_path = current_step_path
            if last_event is not None:
                r.last_event = last_event
            if plan is not None:
                r.plan = plan
            if error is not None:
                r.error = error

            r.updated_ts = _now_ms()

            log.info(
                "event=planrun.update run_id=%s ok=true status=%s->%s last_event=%s->%s step=%s->%s err=%s->%s",
                run_id,
                before_status,
                r.status,
                before_event,
                r.last_event,
                before_step,
                r.current_step_path,
                "yes" if had_error_before else "no",
                "yes" if r.error else "no",
            )
            return True



    def to_dict(self, r: PlanRunState) -> Dict[str, Any]:
        return asdict(r)

    def list_all(self) -> List[PlanRunState]:
        with self._lock:
            return list(self._runs.values())

    # -------------------------------------------------
    # NUEVO: recuperar último run por chat_id + status
    # -------------------------------------------------
    def get_latest_by_status(self, chat_id: str, status: str) -> Optional[PlanRunState]:
        """
        Devuelve el PlanRunState más reciente (por updated_ts) para un chat_id y status.
        """
        with self._lock:
            candidates = [
                r for r in self._runs.values()
                if r.chat_id == chat_id and r.status == status
            ]
            if not candidates:
                return None
            candidates.sort(key=lambda x: (x.updated_ts, x.created_ts), reverse=True)
            return candidates[0]

    def get_latest_draft_run_id(self, chat_id: str) -> Optional[str]:
        """
        Devuelve run_id del draft más reciente de ese chat.
        """
        r = self.get_latest_by_status(chat_id, "draft")
        return r.run_id if r else None
    
    # -------------------------------------------------
    # NUEVO: CAS draft -> running (idempotencia confirmación)
    # -------------------------------------------------
    def try_mark_running(self, run_id: str) -> bool:
        """
        Pasa un run de draft -> running de forma atómica.
        Devuelve True si lo logró. False si no existe o ya no estaba en draft.
        """
        with self._lock:
            r = self._runs.get(run_id)
            if not r:
                return False
            if r.status != "draft":
                return False
            r.status = "running"
            r.updated_ts = _now_ms()
            r.last_event = "confirm_accepted"
            return True
        

    def try_mark_queued(self, run_id: str) -> bool:
        """
        Pasa un run de draft -> queued de forma atómica.
        Devuelve True si lo logró. False si no existe o ya no estaba en draft.
        """
        with self._lock:
            r = self._runs.get(run_id)
            if not r:
                log.info("event=planrun.cas_queued run_id=%s ok=false reason=not_found", run_id)
                return False

            if r.status != "draft":
                log.info(
                    "event=planrun.cas_queued run_id=%s ok=false reason=bad_status status=%s",
                    run_id,
                    r.status,
                )
                return False

            r.status = "queued"
            r.updated_ts = _now_ms()
            r.last_event = "confirm_accepted"

            log.info("event=planrun.cas_queued run_id=%s ok=true", run_id)
            return True





