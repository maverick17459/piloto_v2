from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from src.agent.plan_models import PlanRun  # nuevo import
from src.observability.logger import get_logger


import time
import uuid


def _id() -> str:
    return uuid.uuid4().hex


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str
    ts: int = field(default_factory=_now_ms)


@dataclass
class Chat:
    id: str
    project_id: str
    title: str
    messages: List[Message] = field(default_factory=list)

    # NUEVO: planes asociados al chat
    plan_runs: List[PlanRun] = field(default_factory=list)

    created_ts: int = field(default_factory=_now_ms)
    updated_ts: int = field(default_factory=_now_ms)


@dataclass
class Project:
    id: str
    name: str
    context: str = ""  # contexto del proyecto (memoria/instrucciones)
    mcp_ids: List[str] = field(default_factory=list)  # MCPs activos para este proyecto
    created_ts: int = field(default_factory=_now_ms)
    updated_ts: int = field(default_factory=_now_ms)


class MemoryStore:
    """
    Store en memoria (RAM). Se pierde al reiniciar, excepto lo que deleguemos a state_repo.
    - Proyecto: agrupa chats + tiene contexto global (instrucciones/memoria)
    - Chat: historial propio (user/assistant)
    - Chat state: (pending_run_id, active_run_id, etc.) puede persistirse via state_repo (SQLite)
    """

    def __init__(self, base_system_prompt: str, state_repo=None) -> None:
        self.base_system_prompt = base_system_prompt

        # data in-memory
        self.projects: Dict[str, Project] = {}
        self.chats: Dict[str, Chat] = {}

        # fallback in-memory state
        self.chat_state: Dict[str, Dict[str, Any]] = {}

        # optional persistent repo for chat_state
        self._state_repo = state_repo

        # Proyecto por defecto
        default = self.create_project("Default", context="")
        self.create_chat(default.id, "New chat")

    # ---------------- Projects ----------------
    def list_projects(self) -> List[Project]:
        return sorted(self.projects.values(), key=lambda p: p.updated_ts, reverse=True)

    def get_project(self, project_id: str) -> Optional[Project]:
        return self.projects.get(project_id)

    def create_project(self, name: str, context: str = "", mcp_ids: Optional[List[str]] = None) -> Project:
        p = Project(
            id=_id(),
            name=(name or "").strip() or "New project",
            context=context or "",
            mcp_ids=list(mcp_ids or []),
        )
        self.projects[p.id] = p
        return p

    def update_project(
        self,
        project_id: str,
        *,
        name: Optional[str] = None,
        context: Optional[str] = None,
        mcp_ids: Optional[List[str]] = None,
    ) -> bool:
        p = self.projects.get(project_id)
        if not p:
            return False

        if name is not None:
            name = name.strip()
            if name:
                p.name = name

        if context is not None:
            p.context = context

        if mcp_ids is not None:
            cleaned: List[str] = []
            seen = set()
            for x in mcp_ids:
                s = (x or "").strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                cleaned.append(s)
            p.mcp_ids = cleaned

        p.updated_ts = _now_ms()
        return True

    def delete_project(self, project_id: str) -> bool:
        if project_id not in self.projects:
            return False

        chat_ids = [cid for cid, c in self.chats.items() if c.project_id == project_id]
        for cid in chat_ids:
            self.chats.pop(cid, None)

        self.projects.pop(project_id, None)
        return True

    # ---------------- Chats ----------------
    def list_chats(self, project_id: str) -> List[Chat]:
        chats = [c for c in self.chats.values() if c.project_id == project_id]
        return sorted(chats, key=lambda c: c.updated_ts, reverse=True)

    def get_chat(self, chat_id: str) -> Optional[Chat]:
        return self.chats.get(chat_id)

    def create_chat(self, project_id: str, title: str = "New chat") -> Chat:
        if project_id not in self.projects:
            raise ValueError("Project not found")

        c = Chat(
            id=_id(),
            project_id=project_id,
            title=(title or "").strip() or "New chat",
        )
        self.chats[c.id] = c

        self.projects[project_id].updated_ts = _now_ms()
        return c

    def rename_chat(self, chat_id: str, title: str) -> bool:
        c = self.chats.get(chat_id)
        if not c:
            return False

        c.title = (title or "").strip() or c.title
        c.updated_ts = _now_ms()

        p = self.projects.get(c.project_id)
        if p:
            p.updated_ts = _now_ms()
        return True

    def add_message(self, chat_id: str, role: str, content: str) -> None:
        c = self.chats.get(chat_id)
        if not c:
            raise ValueError(f"Chat not found: {chat_id}")

        c.messages.append(Message(role=role, content=content))
        c.updated_ts = _now_ms()

        p = self.projects.get(c.project_id)
        if p:
            p.updated_ts = _now_ms()

    # ---------------- Chat state ----------------
    def get_state(self, chat_id: str) -> Dict[str, Any]:
        """
        Estado transitorio del chat:
        - pending_run_id
        - active_run_id
        - last_run_id / last_run_status / last_run_ts
        Si hay state_repo (SQLite) delega ahí.
        """
        if self._state_repo:
            # repo devuelve un dict (puede ser vacío)
            return self._state_repo.get_state(chat_id)
        return self.chat_state.setdefault(chat_id, {})

    def set_state(self, chat_id: str, **kwargs: Any) -> None:
        log = get_logger("-")

        # Antes (solo para debug). Si repo, leemos state actual desde repo.
        before = {}
        try:
            before = dict(self.get_state(chat_id))
        except Exception:
            before = {}

        if self._state_repo:
            self._state_repo.set_state(chat_id, **kwargs)
            after = {}
            try:
                after = dict(self._state_repo.get_state(chat_id))
            except Exception:
                after = {}
        else:
            st = self.get_state(chat_id)
            st.update(kwargs)
            after = dict(st)

        log.info(
            "event=chat.state.set chat_id=%s keys=%s before=%s after=%s",
            chat_id,
            list(kwargs.keys()),
            {k: before.get(k) for k in kwargs.keys()},
            {k: after.get(k) for k in kwargs.keys()},
        )

    # ---------------- Plans attached to chat (in-memory list) ----------------
    def add_plan(self, chat_id: str, plan: "PlanRun") -> None:
        c = self.get_chat(chat_id)
        if not c:
            raise ValueError("Chat not found")
        c.plan_runs.append(plan)
        c.updated_ts = _now_ms()

    def get_last_plan(self, chat_id: str) -> Optional["PlanRun"]:
        c = self.get_chat(chat_id)
        if not c or not getattr(c, "plan_runs", None):
            return None
        if not c.plan_runs:
            return None
        return c.plan_runs[-1]

    # ---------------- LLM payload ----------------
    def get_messages_payload(self, chat_id: str) -> List[dict]:
        c = self.chats[chat_id]
        p = self.projects.get(c.project_id)

        project_context = (p.context.strip() if p else "")
        if project_context:
            system = f"{self.base_system_prompt}\n\nContexto del proyecto:\n{project_context}"
        else:
            system = self.base_system_prompt

        payload = [{"role": "system", "content": system}]
        payload += [{"role": m.role, "content": m.content} for m in c.messages]
        return payload

    def chat_preview_title(self, chat_id: str) -> None:
        c = self.chats[chat_id]
        if c.title.lower() != "new chat":
            return

        for m in c.messages:
            if m.role == "user" and m.content.strip():
                t = m.content.strip().split("\n")[0][:40]
                c.title = t or c.title
                c.updated_ts = _now_ms()

                p = self.projects.get(c.project_id)
                if p:
                    p.updated_ts = _now_ms()
                return
