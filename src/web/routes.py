import uuid
from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.config import get_settings
from src.llm.openai_client import OpenAIChatClient
from src.chat.store import MemoryStore
from src.chat.prompts import DEFAULT_SYSTEM_PROMPT

router = APIRouter()
templates = Jinja2Templates(directory="src/web/templates")

settings = get_settings()
client = OpenAIChatClient(api_key=settings.api_key, model=settings.model)

# Store en memoria (RAM). Se reinicia al reiniciar el server.
store = MemoryStore(base_system_prompt=DEFAULT_SYSTEM_PROMPT)

SESSION_COOKIE_NAME = "chat_session_id"


# ---------------- Schemas ----------------
class CreateProjectIn(BaseModel):
    name: str
    context: str | None = None


class UpdateProjectIn(BaseModel):
    name: str | None = None
    context: str | None = None


class CreateChatIn(BaseModel):
    title: str | None = None


class RenameChatIn(BaseModel):
    title: str


class SendMessageIn(BaseModel):
    chat_id: str
    message: str


# ---------------- Page ----------------
@router.get("/", response_class=HTMLResponse)
def index(request: Request, response: Response):
    """
    Página principal.
    Creamos cookie de sesión (por ahora solo para futuro soporte multi-sesión).
    """
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=session_id,
            httponly=True,
            samesite="lax",
        )

    return templates.TemplateResponse(
        "index.html",
        {"request": request},
        headers=response.headers,
    )


# ---------------- API: Projects ----------------
@router.get("/api/projects")
def api_list_projects():
    """Lista proyectos."""
    return {
        "projects": [
            {"id": p.id, "name": p.name, "updated_ts": p.updated_ts}
            for p in store.list_projects()
        ]
    }


@router.post("/api/projects")
def api_create_project(payload: CreateProjectIn):
    """
    MEJORA (A):
    Al crear un proyecto, creamos un chat por defecto para que el proyecto sea usable
    inmediatamente y el frontend siempre pueda enviar mensajes.
    """
    p = store.create_project(payload.name, context=payload.context or "")

    # Crear un chat por defecto dentro del proyecto
    c = store.create_chat(p.id, "New chat")

    return {
        "project": {"id": p.id, "name": p.name, "updated_ts": p.updated_ts},
        "chat": {"id": c.id, "title": c.title, "updated_ts": c.updated_ts},
    }


@router.get("/api/projects/{project_id}")
def api_get_project(project_id: str):
    """Obtiene un proyecto (incluye context)."""
    p = store.get_project(project_id)
    if not p:
        return JSONResponse({"error": "Project not found"}, status_code=404)

    return {
        "project": {
            "id": p.id,
            "name": p.name,
            "context": p.context,
            "updated_ts": p.updated_ts,
        }
    }


@router.patch("/api/projects/{project_id}")
def api_update_project(project_id: str, payload: UpdateProjectIn):
    """Actualiza nombre y/o contexto del proyecto."""
    ok = store.update_project(project_id, name=payload.name, context=payload.context)
    if not ok:
        return JSONResponse({"error": "Project not found"}, status_code=404)
    return {"ok": True}


@router.delete("/api/projects/{project_id}")
def api_delete_project(project_id: str):
    """Elimina proyecto y todos sus chats."""
    ok = store.delete_project(project_id)
    if not ok:
        return JSONResponse({"error": "Project not found"}, status_code=404)
    return {"ok": True}


# ---------------- API: Chats ----------------
@router.get("/api/projects/{project_id}/chats")
def api_list_chats(project_id: str):
    """Lista chats dentro de un proyecto."""
    if not store.get_project(project_id):
        return JSONResponse({"error": "Project not found"}, status_code=404)

    return {
        "chats": [
            {"id": c.id, "title": c.title, "updated_ts": c.updated_ts}
            for c in store.list_chats(project_id)
        ]
    }


@router.post("/api/projects/{project_id}/chats")
def api_create_chat(project_id: str, payload: CreateChatIn):
    """Crea un chat dentro de un proyecto."""
    try:
        c = store.create_chat(project_id, payload.title or "New chat")
    except ValueError:
        return JSONResponse({"error": "Project not found"}, status_code=404)
    return {"chat": {"id": c.id, "title": c.title, "updated_ts": c.updated_ts}}


@router.patch("/api/chats/{chat_id}")
def api_rename_chat(chat_id: str, payload: RenameChatIn):
    """Renombra un chat."""
    ok = store.rename_chat(chat_id, payload.title)
    if not ok:
        return JSONResponse({"error": "Chat not found"}, status_code=404)
    return {"ok": True}


@router.get("/api/chats/{chat_id}/messages")
def api_get_messages(chat_id: str):
    """Devuelve mensajes del chat (sin system porque system se arma en payload al LLM)."""
    c = store.get_chat(chat_id)
    if not c:
        return JSONResponse({"error": "Chat not found"}, status_code=404)

    msgs = [{"role": m.role, "content": m.content, "ts": m.ts} for m in c.messages]
    return {"chat": {"id": c.id, "title": c.title}, "messages": msgs}


# ---------------- API: Send message ----------------
@router.post("/api/send")
def api_send(payload: SendMessageIn):
    """
    Envía un mensaje al chat:
    - Guarda user
    - Construye payload incluyendo contexto del proyecto
    - Llama al modelo
    - Guarda assistant
    """
    chat_id = payload.chat_id
    text = (payload.message or "").strip()
    if not text:
        return JSONResponse({"error": "Mensaje vacío"}, status_code=400)

    c = store.get_chat(chat_id)
    if not c:
        return JSONResponse({"error": "Chat not found"}, status_code=404)

    store.add_message(chat_id, "user", text)

    messages_for_llm = store.get_messages_payload(chat_id)  # incluye contexto de proyecto
    try:
        reply = client.chat(messages_for_llm, temperature=settings.temperature)
    except Exception as e:
        return JSONResponse({"error": f"Error API: {e}"}, status_code=500)

    store.add_message(chat_id, "assistant", reply)
    store.chat_preview_title(chat_id)

    return {"reply": reply}
