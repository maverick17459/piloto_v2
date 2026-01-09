from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
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
    messages: List[Message] = field(default_factory=list)  # guardamos solo user/assistant (sin system)
    created_ts: int = field(default_factory=_now_ms)
    updated_ts: int = field(default_factory=_now_ms)

@dataclass
class Project:
    id: str
    name: str
    context: str = ""  # CONTEXTO DEL PROYECTO
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

        # proyecto por defecto
        default = self.create_project("Default", context="")
        self.create_chat(default.id, "New chat")

    # -------- Projects --------
    def list_projects(self) -> List[Project]:
        return sorted(self.projects.values(), key=lambda p: p.updated_ts, reverse=True)

    def get_project(self, project_id: str) -> Optional[Project]:
        return self.projects.get(project_id)

    def create_project(self, name: str, context: str = "") -> Project:
        p = Project(id=_id(), name=name.strip() or "Untitled", context=context or "")
        self.projects[p.id] = p
        return p

    def update_project(self, project_id: str, name: Optional[str] = None, context: Optional[str] = None) -> bool:
        p = self.projects.get(project_id)
        if not p:
            return False
        if name is not None:
            p.name = name.strip() or p.name
        if context is not None:
            p.context = context.strip()
        p.updated_ts = _now_ms()
        return True

    # -------- Chats --------
    def list_chats(self, project_id: str) -> List[Chat]:
        chats = [c for c in self.chats.values() if c.project_id == project_id]
        return sorted(chats, key=lambda c: c.updated_ts, reverse=True)

    def get_chat(self, chat_id: str) -> Optional[Chat]:
        return self.chats.get(chat_id)

    def create_chat(self, project_id: str, title: str = "New chat") -> Chat:
        if project_id not in self.projects:
            raise ValueError("Project not found")
        c = Chat(id=_id(), project_id=project_id, title=title.strip() or "New chat")
        self.chats[c.id] = c
        self.projects[project_id].updated_ts = _now_ms()
        return c

    def rename_chat(self, chat_id: str, title: str) -> bool:
        c = self.chats.get(chat_id)
        if not c:
            return False
        c.title = title.strip() or c.title
        c.updated_ts = _now_ms()
        self.projects[c.project_id].updated_ts = _now_ms()
        return True

    def add_message(self, chat_id: str, role: str, content: str) -> None:
        c = self.chats[chat_id]
        c.messages.append(Message(role=role, content=content))
        c.updated_ts = _now_ms()
        self.projects[c.project_id].updated_ts = _now_ms()

    def get_messages_payload(self, chat_id: str) -> List[dict]:
        """
        Armamos el contexto así:
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
        c = self.chats[chat_id]
        if c.title.lower() != "new chat":
            return
        for m in c.messages:
            if m.role == "user" and m.content.strip():
                t = m.content.strip().split("\n")[0][:40]
                c.title = t or c.title
                c.updated_ts = _now_ms()
                self.projects[c.project_id].updated_ts = _now_ms()
                return
            

def delete_project(self, project_id: str) -> bool:
    if project_id not in self.projects:
        return False

    # eliminar chats del proyecto
    for cid in list(self.chats.keys()):
        if self.chats[cid].project_id == project_id:
            del self.chats[cid]

    del self.projects[project_id]
    return True

