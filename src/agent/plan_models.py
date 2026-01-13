from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import uuid

def _now_ms() -> int:
    return int(time.time() * 1000)

def _id() -> str:
    return uuid.uuid4().hex

@dataclass
class PlanStep:
    id: str = field(default_factory=_id)
    title: str = ""
    type: str = "note"  # "mcp_call" | "note" | "subplan"

    # Para mcp_call
    mcp_id: Optional[str] = None
    method: Optional[str] = None
    path: Optional[str] = None
    query: Optional[Dict[str, Any]] = None
    body: Any = None

    # Subpasos
    substeps: List["PlanStep"] = field(default_factory=list)

    # Estado de ejecución
    status: str = "pending"  # pending|running|done|error|skipped
    started_ts: Optional[int] = None
    ended_ts: Optional[int] = None
    error: Optional[str] = None

    # Resultado
    result_summary: Optional[str] = None
    result_raw: Any = None

@dataclass
class PlanRun:
    id: str = field(default_factory=_id)
    goal: str = ""
    steps: List[PlanStep] = field(default_factory=list)
    status: str = "pending"  # pending|running|done|error
    created_ts: int = field(default_factory=_now_ms)
    ended_ts: Optional[int] = None
    current_step_path: str = ""  # ej "2" o "2.1"
