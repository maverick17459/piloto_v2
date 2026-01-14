from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
import threading
import time
import uuid


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
                return False

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
            return True

    def to_dict(self, r: PlanRunState) -> Dict[str, Any]:
        return asdict(r)
    
    def list_all(self):
        with self._lock:
            return list(self._runs.values())

