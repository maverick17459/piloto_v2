from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from src.agent.plan_models import PlanRun  # nuevo import

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
    Store en memoria (RAM). Se pierde al reiniciar.
    - Proyecto: agrupa chats + tiene contexto global (instrucciones/memoria)
    - Chat: historial propio (user/assistant)
    """

    def __init__(self, base_system_prompt: str) -> None:
        self.base_system_prompt = base_system_prompt
        self.projects: Dict[str, Project] = {}
        self.chats: Dict[str, Chat] = {}

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
            # Normaliza: str, sin vacíos, sin duplicados, orden estable
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
        """
        Elimina un proyecto y todos sus chats asociados.
        """
        if project_id not in self.projects:
            return False

        # 1) borrar chats del proyecto
        chat_ids = [cid for cid, c in self.chats.items() if c.project_id == project_id]
        for cid in chat_ids:
            self.chats.pop(cid, None)

        # 2) borrar proyecto
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

        # Actualiza timestamp del proyecto
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
        c = self.chats[chat_id]
        c.messages.append(Message(role=role, content=content))
        c.updated_ts = _now_ms()

        p = self.projects.get(c.project_id)
        if p:
            p.updated_ts = _now_ms()

    def get_messages_payload(self, chat_id: str) -> List[dict]:
        """
        Arma el payload al LLM:
        - 1 mensaje system con prompt base + contexto del proyecto
        - luego todos los mensajes del chat (user/assistant)
        """
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
        """
        Si el chat sigue llamándose "New chat", usa el primer mensaje del usuario como preview.
        """
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
            
        def add_plan(self, chat_id: str, plan: PlanRun) -> None:
            c = self.get_chat(chat_id)
            if not c:
                raise ValueError("Chat not found")
            c.plan_runs.append(plan)
            c.updated_ts = _now_ms()

        def get_last_plan(self, chat_id: str) -> Optional[PlanRun]:
            c = self.get_chat(chat_id)
            if not c or not c.plan_runs:
                return None
            return c.plan_runs[-1]



