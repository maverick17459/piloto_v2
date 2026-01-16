import uuid
import json
import time
import re
import asyncio
from typing import Optional, List, Any, Dict

from fastapi import APIRouter, Request, Response

from dataclasses import asdict

from src.agent.plan_models import PlanRun, PlanStep

from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.app_state import client, store, mcp_store, mcp_service, plan_run_store

from src.mcp.invoke_sync import invoke_mcp_sync, MCPCall, MCPInvokeError


from src.agent.plan_background_runner import run_plan_in_background

from src.observability.logger import get_logger, get_prompt_logger, prompts_enabled
from src.observability.prompt_debug import (
    summarize_messages,
    serialize_messages_for_promptlog,
    serialize_text_for_promptlog,
)

router = APIRouter()
templates = Jinja2Templates(directory="src/web/templates")


SESSION_COOKIE_NAME = "chat_session_id"

_PLAN_TEXT_PATTERNS = [
    r"\bplan\b",
    r"\bplan\s+propuest[oa]\b",
    r"\bpropuest[oa]\s+de\s+plan\b",
    r"\bpaso(s)?\b",
    r"\bpaso\s+\d+\b",
    r"\bstep(s)?\b",
    r"\bpropuest[oa]\b",
    r"\brun(ning)?\b",
    r"\baction\b",
]

_CONFIRMATION_TEXT_PATTERNS = [
    r"\bconfirmo\b",
    r"\bconfirma(r|ción)?\b",
    r"\bcancela(r|do|ción)?\b",
    r"\bpara\s+ejecutar\b",
    r"\bpara\s+descartar\b",
    r"\b(responde|di|escribe)\b.*\b(confirmo|confirma|confirm)\b",
    r"\b(responde|di|escribe)\b.*\b(cancela|cancelar|cancel)\b",
    r"\b(reply|type)\b.*\bconfirm\b",
    r"\b(reply|type)\b.*\bcancel\b",
]


MCP_REQUEST_TOOL = {
    "type": "function",
    "function": {
        "name": "mcp_request",
        "description": "Execute a request against a connected MCP server",
        "parameters": {
            "type": "object",
            "required": ["mcp_id", "method", "path"],
            "properties": {
                "mcp_id": {
                    "type": "string",
                    "description": "Target MCP identifier"
                },
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"]
                },
                "path": {
                    "type": "string",
                    "description": "Endpoint path exposed by the MCP"
                },
                "query": {
                    "type": "object",
                    "additionalProperties": True
                },
                "body": {
                    "type": ["object", "string", "null"]
                }
            },
            "additionalProperties": False
        }
    }
}


# ---------------- Schemas ----------------

class CreateProjectIn(BaseModel):
    name: str
    context: str | None = None
    mcp_ids: list[str] | None = None


class UpdateProjectIn(BaseModel):
    name: str | None = None
    context: str | None = None
    mcp_ids: list[str] | None = None


class CreateChatIn(BaseModel):
    title: str | None = None


class RenameChatIn(BaseModel):
    title: str


class SendMessageIn(BaseModel):
    chat_id: str
    message: str


# ---------------- MCP Schemas ----------------

class MCPRegisterIn(BaseModel):
    address: str
    name: Optional[str] = None
    docs_url: Optional[str] = None
    save_openapi_raw: bool = False


class MCPUpdateIn(BaseModel):
    name: Optional[str] = None
    base_url: Optional[str] = None
    docs_url: Optional[str] = None


class MCPSetActiveIn(BaseModel):
    active: bool


class MCPEndpointOut(BaseModel):
    path: str
    method: str
    operation_id: Optional[str] = None
    summary: Optional[str] = None
    tags: List[str] = []


class MCPOut(BaseModel):
    id: str
    name: str
    base_url: str
    docs_url: Optional[str] = None
    openapi_url: Optional[str] = None
    is_active: bool
    endpoints: List[MCPEndpointOut] = []
    created_ts: int
    updated_ts: int


# ---------------- Page ----------------

@router.get("/", response_class=HTMLResponse)
def index(request: Request, response: Response):
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


# ---------------- Helpers ----------------

def _looks_like_plan_text(text: str) -> bool:
    """
    Detecta cuando el MODELO escribió un 'plan' en texto (anti-plan-fantasma).
    No intenta inferir intención del usuario.
    """
    if not text:
        return False

    t = text.lower()
    hits = 0
    for pat in _PLAN_TEXT_PATTERNS:
        if re.search(pat, t):
            hits += 1
            if hits >= 2:
                return True
    return False



def _looks_like_confirmation_prompt(text: str) -> bool:
    """
    Detecta cuando el MODELO pide confirmación en texto.
    Debe usarse SIEMPRE junto con verificación de pending_run_id.
    """
    if not text:
        return False
    t = text.lower()
    return any(re.search(p, t) for p in _CONFIRMATION_TEXT_PATTERNS)




def _extract_command(text: str) -> Optional[str]:
    # simple: "ejecuta el comando ipconfig" / "ejecuta ipconfig"
    m = re.search(r"(?:ejecuta(?:me)?(?:\s+el)?\s+comando\s+|ejecuta\s+)(.+)$", (text or "").strip(), re.I)
    if not m:
        return None
    cmd = m.group(1).strip().strip("`")
    return cmd if cmd else None

def _pick_command_mcp_id(tools_ctx: list[dict]) -> Optional[str]:
    # tools_ctx: lista de MCPs del proyecto con endpoints
    for m in tools_ctx:
        mcp_id = m.get("id") or m.get("mcp_id")
        for ep in (m.get("endpoints") or []):
            if (ep.get("method") or "").upper() == "POST" and (ep.get("path") or "") == "/command":
                return mcp_id
    return None


def _mcp_to_out(m) -> dict:
    return {
        "id": m.id,
        "name": m.name,
        "base_url": m.base_url,
        "docs_url": m.docs_url,
        "openapi_url": m.openapi_url,
        "is_active": m.is_active,
        "endpoints": [
            {
                "path": e.path,
                "method": e.method,
                "operation_id": e.operation_id,
                "summary": e.summary,
                "tags": e.tags,
            }
            for e in (m.endpoints or [])
        ],
        "created_ts": m.created_ts,
        "updated_ts": m.updated_ts,
    }


def _build_tools_ctx_for_project(project) -> list[dict]:
    tools: list[dict] = []
    for mcp_id in (project.mcp_ids or []):
        m = mcp_store.get_mcp(mcp_id)
        if not m or not m.is_active:
            continue

        endpoints = []
        for e in (m.endpoints or []):
            endpoints.append(
                {
                    "method": e.method,
                    "path": e.path,
                    "operation_id": e.operation_id,
                    "summary": e.summary,
                    "tags": e.tags,
                }
            )

        tools.append(
            {
                "mcp_id": m.id,
                "name": m.name,
                "base_url": m.base_url,
                "endpoints": endpoints,
            }
        )

    return tools



async def _start_run_from_draft(run_id: str, request: Request) -> JSONResponse | dict:
    trace_id = getattr(request.state, "trace_id", request.headers.get("X-Trace-Id", "-"))
    log = get_logger(trace_id)

    r = plan_run_store.get(run_id)
    if not r:
        return JSONResponse({"error": "Run not found"}, status_code=404)

    # Acepta draft o queued:
    # - draft: aún no arrancó (puede haber llegado sin CAS previo)
    # - queued: confirmación ya ganó CAS y quedó listo para arrancar
    if r.status not in ("draft", "queued"):
        return JSONResponse(
            {"error": f"Run no está listo para iniciar (status={r.status})"},
            status_code=409,
        )

    plan_dict = r.plan
    if not isinstance(plan_dict, dict):
        # LOG FUERTE para depuración (no asumir)
        log.info(
            "event=run.start.missing_plan run_id=%s chat_id=%s status=%s last_event=%s",
            run_id,
            r.chat_id,
            r.status,
            getattr(r, "last_event", None),
        )
        return JSONResponse({"error": "Run draft sin plan"}, status_code=500)

    def step_from_dict(d: dict) -> PlanStep:
        st = PlanStep(
            title=d.get("title", ""),
            type=d.get("type", "note"),
            mcp_id=d.get("mcp_id"),
            method=d.get("method"),
            path=d.get("path"),
            query=d.get("query"),
            body=d.get("body"),
        )
        subs = d.get("substeps")
        if isinstance(subs, list):
            st.substeps = [step_from_dict(x) for x in subs if isinstance(x, dict)]
            if st.substeps and st.type != "subplan":
                st.type = "subplan"
        return st

    plan = PlanRun(
        goal=plan_dict.get("goal", "Plan"),
        steps=[step_from_dict(s) for s in (plan_dict.get("steps") or []) if isinstance(s, dict)],
    )
    if "id" in plan_dict:
        try:
            plan.id = plan_dict["id"]
        except Exception:
            pass

    chat = store.get_chat(r.chat_id)
    proj = store.get_project(chat.project_id) if chat else None

    # -------------------------------------------------
    # Arranque idempotente: si estaba draft, gana CAS aquí
    # -------------------------------------------------
    if r.status == "draft":
        ok = False
        try:
            ok = plan_run_store.try_mark_queued(run_id)
        except Exception as e:
            log.info(
                "event=plan.try_mark_queued.error run_id=%s err=%s msg=%s",
                run_id,
                type(e).__name__,
                e,
            )

        if not ok:
            # Otro request pudo haber ganado o el run cambió de estado
            rr = None
            try:
                rr = plan_run_store.get(run_id)
            except Exception:
                pass
            st = rr.status if rr else "unknown"
            return JSONResponse(
                {"error": f"Run ya no está en draft (status={st})"},
                status_code=409,
            )

        # Marcamos evento de arranque (sin cambiar status, ya está queued)
        plan_run_store.update(run_id, last_event="run_start_queued")

    else:
        # ya estaba queued
        plan_run_store.update(run_id, last_event="run_start_queued")

    store.add_message(r.chat_id, "assistant", f"Confirmado. Ejecutando plan… (run_id={run_id})")

    # Limpia pending y marca active (persistido via state_repo)
    try:
        store.set_state(r.chat_id, pending_run_id=None, active_run_id=run_id)
    except Exception as e:
        log.info(
            "event=chat.state.set_active.error run_id=%s chat_id=%s err=%s msg=%s",
            run_id,
            r.chat_id,
            type(e).__name__,
            e,
        )

    task = asyncio.create_task(
        run_plan_in_background(
            run_id=run_id,
            chat_id=r.chat_id,
            plan=plan,
            store=store,
            mcp_store=mcp_store,
            proj=proj,
            trace_id=trace_id,
            log=log,
            run_store=plan_run_store,  # SqlitePlanRunStore
            client=client,
        )
    )

    # -------------------------------------------------
    # IMPORTANTE:
    # - NO limpiar estado aquí (evita duplicación con run_plan_in_background.finally)
    # - Solo loguear errores del task
    # -------------------------------------------------
    def _on_done(t: asyncio.Task):
        try:
            t.result()
        except Exception as e:
            log.exception(
                "event=plan.bg.task_error run_id=%s err=%s msg=%s",
                run_id,
                type(e).__name__,
                e,
            )

    task.add_done_callback(_on_done)

    return {"ok": True, "run_id": run_id, "status": "queued"}







def recover_stale_runs(plan_run_store, store, log):
    """
    Marca como error los runs que quedaron queued/running tras un reload.
    Evita estados colgados eternos.
    """
    try:
        runs = plan_run_store.list_all()
    except Exception:
        log.exception("recover_stale_runs: cannot list runs")
        return

    for r in runs:
        if r.status in ("queued", "running"):
            log.warning(f"Recovering stale run {r.run_id} (status={r.status})")
            plan_run_store.update(
                r.run_id,
                status="error",
                last_event="recovered_after_reload",
                error="Run detenido por recarga del servidor",
            )
            try:
                store.add_message(
                    r.chat_id,
                    "assistant",
                    f"El plan fue detenido por una recarga del servidor.\n(run_id={r.run_id})",
                )
            except Exception:
                pass






# ---------------- API: Projects ----------------

@router.get("/api/projects")
def api_list_projects():
    return {
        "projects": [
            {"id": p.id, "name": p.name, "updated_ts": p.updated_ts}
            for p in store.list_projects()
        ]
    }


@router.post("/api/projects")
def api_create_project(payload: CreateProjectIn):
    p = store.create_project(payload.name, context=payload.context or "", mcp_ids=payload.mcp_ids or [])
    c = store.create_chat(p.id, "New chat")
    return {
        "project": {"id": p.id, "name": p.name, "updated_ts": p.updated_ts},
        "chat": {"id": c.id, "title": c.title, "updated_ts": c.updated_ts},
    }


@router.get("/api/projects/{project_id}")
def api_get_project(project_id: str):
    p = store.get_project(project_id)
    if not p:
        return JSONResponse({"error": "Project not found"}, status_code=404)
    return {
        "project": {
            "id": p.id,
            "name": p.name,
            "context": p.context,
            "mcp_ids": p.mcp_ids,
            "updated_ts": p.updated_ts,
        }
    }


@router.patch("/api/projects/{project_id}")
def api_update_project(project_id: str, payload: UpdateProjectIn):
    ok = store.update_project(project_id, name=payload.name, context=payload.context, mcp_ids=payload.mcp_ids)
    if not ok:
        return JSONResponse({"error": "Project not found"}, status_code=404)
    return {"ok": True}


@router.delete("/api/projects/{project_id}")
def api_delete_project(project_id: str):
    ok = store.delete_project(project_id)
    if not ok:
        return JSONResponse({"error": "Project not found"}, status_code=404)
    return {"ok": True}


# ---------------- API: Chats ----------------

@router.get("/api/projects/{project_id}/chats")
def api_list_chats(project_id: str):
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
    try:
        c = store.create_chat(project_id, payload.title or "New chat")
    except ValueError:
        return JSONResponse({"error": "Project not found"}, status_code=404)
    return {"chat": {"id": c.id, "title": c.title, "updated_ts": c.updated_ts}}


@router.patch("/api/chats/{chat_id}")
def api_rename_chat(chat_id: str, payload: RenameChatIn):
    ok = store.rename_chat(chat_id, payload.title)
    if not ok:
        return JSONResponse({"error": "Chat not found"}, status_code=404)
    return {"ok": True}



@router.get("/api/chats/{chat_id}/messages")
def api_get_messages(chat_id: str, request: Request):
    trace_id = getattr(request.state, "trace_id", request.headers.get("X-Trace-Id", "-"))
    log = get_logger(trace_id)

    log.info(f"event=messages.get.start chat_id={chat_id}")

    c = store.get_chat(chat_id)
    if not c:
        log.info(f"event=messages.get.missing chat_id={chat_id}")
        return JSONResponse({"error": "Chat not found"}, status_code=404)

    # Estado del chat (clave para debug de polling)
    try:
        state = store.get_state(chat_id)
    except Exception as e:
        log.info(f"event=chat.state.error chat_id={chat_id} err={type(e).__name__}")
        state = {}

    pending_run_id = state.get("pending_run_id")
    active_run_id = state.get("active_run_id")

    msgs = [{"role": m.role, "content": m.content, "ts": m.ts} for m in c.messages]
    count = len(msgs)
    last_ts = msgs[-1]["ts"] if msgs else None

    log.info(
        f"event=messages.get.ok chat_id={chat_id} count={count} last_ts={last_ts} "
        f"pending_run_id={pending_run_id} active_run_id={active_run_id}"
    )

    return {
        "chat": {"id": c.id, "title": c.title},
        "messages": msgs,
        "state": {"pending_run_id": pending_run_id, "active_run_id": active_run_id},
        "meta": {"count": count, "last_ts": last_ts},
    }



# ---------------- API: Send ----------------

@router.post("/api/send")
async def api_send(payload: SendMessageIn, request: Request):
    trace_id = getattr(request.state, "trace_id", request.headers.get("X-Trace-Id", "-"))
    log = get_logger(trace_id)
    plog = get_prompt_logger(trace_id)

    t_total = time.time()

    chat_id = payload.chat_id
    text = (payload.message or "").strip()

    log.info(f"event=send.start chat_id={chat_id} user_len={len(text)}")

    if not text:
        log.info("event=send.reject reason=empty_text")
        return JSONResponse({"error": "Mensaje vacío"}, status_code=400)

    c = store.get_chat(chat_id)
    if not c:
        log.info("event=send.reject reason=chat_not_found")
        return JSONResponse({"error": "Chat not found"}, status_code=404)

    # -------------------------------------------------
    # 0) Estado actual del chat
    # -------------------------------------------------
    try:
        state = store.get_state(chat_id)
    except Exception as e:
        log.info(f"event=chat.state.error err={type(e).__name__}")
        state = {}

    pending_run_id = state.get("pending_run_id")
    active_run_id = state.get("active_run_id")

    # extra: idempotencia confirmaciones tardías/duplicadas
    last_run_id = state.get("last_run_id")
    last_run_status = state.get("last_run_status")
    last_run_ts = state.get("last_run_ts")

    log.info(
        f"event=chat.state chat_id={chat_id} "
        f"pending_run_id={pending_run_id} active_run_id={active_run_id} "
        f"last_run_id={last_run_id} last_run_status={last_run_status} last_run_ts={last_run_ts}"
    )

    # -------------------------------------------------
    # 1) Guardar mensaje del usuario
    # -------------------------------------------------
    store.add_message(chat_id, "user", text)

    cmd = _extract_command(text)
    if cmd:
        proj = store.get_project(c.project_id)
        tools_ctx = _build_tools_ctx_for_project(proj) if proj else []
        mcp_id_cmd = _pick_command_mcp_id(tools_ctx)

        if not mcp_id_cmd:
            reply = "No tengo un MCP activo con endpoint POST /command para este proyecto."
            store.add_message(chat_id, "assistant", reply)
            store.chat_preview_title(chat_id)
            return {"reply": reply}

        # crea plan draft sin LLM
        method = "POST"
        path = "/command"
        body = {"cmd": cmd}

        goal = f"{method} {path}"
        step = PlanStep(title=goal, type="mcp_call", mcp_id=mcp_id_cmd, method=method, path=path, query=None, body=body)
        plan = PlanRun(goal=goal, steps=[step])

        run = plan_run_store.create(chat_id=chat_id, plan_id=plan.id, goal=plan.goal)
        plan_run_store.update(run.run_id, status="draft", plan=asdict(plan), last_event="plan_draft")

        store.set_state(chat_id, pending_run_id=run.run_id)

        reply = (
            f"📌 **Plan propuesto**\n"
            f"- MCP: `{mcp_id_cmd}`\n"
            f"- Acción: `{method} {path}`\n\n"
            f"Comando: `{cmd}`\n\n"
            f"Responde **confirmo** para ejecutarlo o **cancela** para descartarlo."
        )

        store.add_message(chat_id, "assistant", reply)
        store.chat_preview_title(chat_id)
        log.info(f"event=plan.draft.fastpath chat_id={chat_id} run_id={run.run_id} cmd={cmd}")

        return {"run_id": run.run_id, "status": "draft", "reply": reply}

    # -------------------------------------------------
    # 2) Confirmación / Cancelación
    # -------------------------------------------------
    t = text.lower().strip()

    CONFIRM_WORDS = {"confirmo", "sí", "si", "ok", "dale", "ejecuta", "proceder", "continuar"}
    CANCEL_WORDS = {"cancela", "cancelar", "no", "detener", "para"}

    is_confirm = t in CONFIRM_WORDS
    is_cancel = t in CANCEL_WORDS

    # -------------------------------------------------
    # Guard robusto: confirmaciones fuera de timing
    # -------------------------------------------------
    if (is_confirm or is_cancel) and pending_run_id is None:

        # 1) Si hay un run activo
        if active_run_id:
            reply = f"Ese plan ya fue confirmado y está en ejecución (run_id={active_run_id})."
            store.add_message(chat_id, "assistant", reply)
            store.chat_preview_title(chat_id)
            log.info(f"event=send.confirmation.late chat_id={chat_id} user_text={t} active_run_id={active_run_id}")
            log.info(f"event=send.done duration_ms={int((time.time()-t_total)*1000)} mode=confirm_late_active")
            return {"reply": reply, "active_run_id": active_run_id}

        # 2) Intentar recuperar el último draft del chat
        recovered_run_id = None
        try:
            recovered_run_id = plan_run_store.get_latest_draft_run_id(chat_id)
        except Exception as e:
            log.info(f"event=plan.draft.recover.error err={type(e).__name__}")

        if recovered_run_id and is_confirm:
            log.info(f"event=plan.draft.recovered chat_id={chat_id} run_id={recovered_run_id}")

            # FIX 1: CAS ANTES de tocar state (evita dejar pending sucio)
            ok = False
            try:
                ok = plan_run_store.try_mark_queued(recovered_run_id)
            except Exception as e:
                log.info(f"event=plan.try_mark_queued.error err={type(e).__name__}")

            if not ok:
                st = None
                try:
                    rr = plan_run_store.get(recovered_run_id)
                    st = rr.status if rr else None
                except Exception:
                    pass

                reply = f"Ese plan ya fue confirmado (run_id={recovered_run_id}, estado={st or 'unknown'})."
                store.add_message(chat_id, "assistant", reply)
                store.chat_preview_title(chat_id)
                log.info(
                    f"event=send.confirmation.idempotent chat_id={chat_id} run_id={recovered_run_id} status={st}"
                )
                log.info(
                    f"event=send.done duration_ms={int((time.time()-t_total)*1000)} mode=confirm_idempotent_recovered"
                )
                return {"reply": reply, "run_id": recovered_run_id, "status": st or "unknown"}

            # Solo si ganaste CAS, actualiza state y arranca
            try:
                store.set_state(chat_id, pending_run_id=None, active_run_id=recovered_run_id)
            except Exception as e:
                log.info(f"event=chat.state.set_pending.error err={type(e).__name__}")

            return await _start_run_from_draft(recovered_run_id, request)

        # 2.5) Idempotencia: confirmación duplicada/tardía reciente
        now_ms = int(time.time() * 1000)
        if last_run_id and last_run_ts:
            try:
                if (now_ms - int(last_run_ts)) < 120_000:  # 2 min
                    reply = (
                        f"Ese plan ya fue confirmado y terminó "
                        f"(estado={last_run_status}, run_id={last_run_id})."
                    )
                    store.add_message(chat_id, "assistant", reply)
                    store.chat_preview_title(chat_id)
                    log.info(
                        f"event=send.confirmation.duplicate chat_id={chat_id} user_text={t} "
                        f"last_run_id={last_run_id} last_run_status={last_run_status}"
                    )
                    log.info(
                        f"event=send.done duration_ms={int((time.time()-t_total)*1000)} mode=confirm_duplicate_recent"
                    )
                    return {"reply": reply, "run_id": last_run_id, "status": last_run_status}
            except Exception:
                pass

        # 3) No hay pending, no hay active, no hay draft recuperable, no hay last reciente
        reply = "No hay ningún plan pendiente para confirmar o cancelar. Pídeme la acción y te propongo un plan."
        store.add_message(chat_id, "assistant", reply)
        store.chat_preview_title(chat_id)
        log.info(f"event=send.confirmation.orphan chat_id={chat_id} user_text={t}")
        log.info(f"event=send.done duration_ms={int((time.time()-t_total)*1000)} mode=confirm_orphan")
        return {"reply": reply}

    # -------------------------------------------------
    # Confirmación normal: hay pending_run_id
    # -------------------------------------------------
    if pending_run_id is not None:
        log.info(f"event=send.confirmation.detected chat_id={chat_id} pending_run_id={pending_run_id} user_text={t}")

        # FIX 2: si pending apunta a run inexistente, limpiar y responder claro
        try:
            pr = plan_run_store.get(pending_run_id)
        except Exception:
            pr = None

        if not pr:
            try:
                store.set_state(chat_id, pending_run_id=None, active_run_id=None)

            except Exception:
                pass
            reply = (
                "El plan pendiente ya no existe (posible reinicio/limpieza). "
                "Pídeme la acción de nuevo y te propongo un plan."
            )
            store.add_message(chat_id, "assistant", reply)
            store.chat_preview_title(chat_id)
            log.info(f"event=send.pending.missing chat_id={chat_id} pending_run_id={pending_run_id}")
            log.info(f"event=send.done duration_ms={int((time.time()-t_total)*1000)} mode=pending_missing")
            return {"reply": reply}

        if is_confirm:
            # CAS: solo el primer confirm pasa draft -> queued
            ok = False
            try:
                ok = plan_run_store.try_mark_queued(pending_run_id)
            except Exception as e:
                log.info(f"event=plan.try_mark_queued.error err={type(e).__name__}")

            if not ok:
                st = None
                try:
                    rr = plan_run_store.get(pending_run_id)
                    st = rr.status if rr else None
                except Exception:
                    pass

                reply = f"Ese plan ya fue confirmado (run_id={pending_run_id}, estado={st or 'unknown'})."
                store.add_message(chat_id, "assistant", reply)
                store.chat_preview_title(chat_id)
                log.info(
                    f"event=send.confirmation.idempotent chat_id={chat_id} run_id={pending_run_id} status={st}"
                )
                log.info(f"event=send.done duration_ms={int((time.time()-t_total)*1000)} mode=confirm_idempotent")
                return {"reply": reply, "run_id": pending_run_id, "status": st or "unknown"}

            log.info(f"event=send.confirmation.accept chat_id={chat_id} run_id={pending_run_id}")

            store.set_state(
                chat_id,
                pending_run_id=None,
                active_run_id=pending_run_id,
            )

            res = await _start_run_from_draft(pending_run_id, request)

            log.info(f"event=send.confirmation.accept.done chat_id={chat_id} run_id={pending_run_id}")
            return res

        if is_cancel:
            log.info(f"event=send.confirmation.cancel chat_id={chat_id} run_id={pending_run_id}")
            try:
                store.set_state(chat_id, pending_run_id=None, active_run_id=None)

            except Exception as e:
                log.info(f"event=chat.state.clear_pending.error err={type(e).__name__}")

            reply = "Listo, cancelado. Dime qué quieres cambiar del plan y lo ajusto."
            store.add_message(chat_id, "assistant", reply)
            store.chat_preview_title(chat_id)
            log.info(f"event=send.done duration_ms={int((time.time()-t_total)*1000)} mode=pending_cancelled")
            return {"reply": reply}

        reply = "Tengo un plan pendiente. Responde **confirmo** para ejecutarlo o **cancela** para descartarlo."
        store.add_message(chat_id, "assistant", reply)
        store.chat_preview_title(chat_id)
        log.info(f"event=send.done duration_ms={int((time.time()-t_total)*1000)} mode=pending_wait")
        return {"reply": reply, "pending_run_id": pending_run_id}

    # -------------------------------------------------
    # 3) Mensajes para el LLM (sin router)
    # -------------------------------------------------
    messages = store.get_messages_payload(chat_id)
    plog.info("event=prompt.chat.content\n" + serialize_messages_for_promptlog(messages))

    proj = store.get_project(c.project_id)
    tools_ctx = _build_tools_ctx_for_project(proj) if proj else []

    tools_catalog_system = {
        "role": "system",
        "content": (
            "TOOLS_CATALOG:\n"
            + json.dumps(tools_ctx, ensure_ascii=False)
            + "\n\n"
            "Rules:\n"
            "- If you call mcp_request, you MUST pick mcp_id/method/path from TOOLS_CATALOG.\n"
            "- If required info is missing, ask the user instead of guessing.\n"
            "\n"
            "Hard rules:\n"
            "- Decide between (A) a normal chat response and (B) a tool action proposal (mcp_request).\n"
            "- You MUST produce an mcp_request when the user is asking to DO something that requires an external action/tool, such as:\n"
            "  executing a command, calling an API endpoint, creating/updating/deleting resources,\n"
            "  fetching data from an MCP, or actions expressed as: \"haz\", \"ejecuta\", \"corre\", \"llama\",\n"
            "  \"consulta\", \"crea\", \"actualiza\", \"borra\", \"descarga\", \"sube\", \"abre\", \"revisa\".\n"
            "- You MUST NOT answer with plain text if the request is actionable via an MCP and all required info is available.\n"
            "- If the request is actionable but ambiguous or missing required parameters, ask a short clarifying question.\n"
            "\n"
            "Command execution rule (/command):\n"
            "- If the user asks to run or execute a system command, you MUST call mcp_request with method POST and path /command.\n"
            "- The command MUST be placed in body.cmd as a string (example: {\"cmd\": \"ipconfig\"}).\n"
            "- Do NOT use any other body shape for /command.\n"
            "\n"
            "Catalog constraints:\n"
            "- Never invent MCPs, tools, methods, or paths that are not present in TOOLS_CATALOG.\n"
        ),
    }

    llm_messages = [tools_catalog_system] + messages

    # -------------------------------------------------
    # 4) Llamada al modelo con tool-calling (mode=message)
    # -------------------------------------------------
    t_llm = time.time()
    try:
        msg = client.chat(
            llm_messages,
            tools=[MCP_REQUEST_TOOL],
            tool_choice="auto",
            temperature=0,
            mode="message",
        )

        log.info(
            "event=llm.response "
            f"has_tool_calls={bool(msg.get('tool_calls'))} "
            f"content_len={len(msg.get('content') or '')}"
        )

        if msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            fn = tc.get("function") or {}
            log.info(
                "event=llm.tool_call "
                f"name={fn.get('name')} "
                f"has_args={'arguments' in fn} "
                f"args_type={type(fn.get('arguments')).__name__}"
            )

    except Exception as e:
        log.info(f"event=llm.call.error duration_ms={int((time.time()-t_llm)*1000)} err={type(e).__name__}")
        return JSONResponse({"error": f"Error API: {e}"}, status_code=500)

    log.info(f"event=llm.call.done duration_ms={int((time.time()-t_llm)*1000)}")

    tool_calls = msg.get("tool_calls")
    content = msg.get("content")

    # -----------------------------------------
    # RESPUESTA NORMAL
    # -----------------------------------------
    
    if not tool_calls:
        # Anti-plan-fantasma: si el modelo escribió un plan en texto, reintenta forzando tool-call.
        if _looks_like_plan_text(content or ""):
            log.info("event=respond.plan_text_detected retry=force_tool_choice")

            try:
                msg2 = client.chat(
                    llm_messages,
                    tools=[MCP_REQUEST_TOOL],
                    tool_choice={"type": "function", "function": {"name": "mcp_request"}},
                    temperature=0,
                    mode="message",
                )
            except Exception as e:
                log.info(f"event=llm.force_tool_call.error err={type(e).__name__}")
                reply = "No pude generar una acción ejecutable. Repite la petición."
                store.add_message(chat_id, "assistant", reply)
                store.chat_preview_title(chat_id)
                return {"reply": reply}

            tool_calls2 = msg2.get("tool_calls")
            content2 = msg2.get("content")

            log.info(
                "event=llm.force_tool_call.response "
                f"has_tool_calls={bool(tool_calls2)} content_len={len(content2 or '')}"
            )

            if tool_calls2:
                # Ahora sí hay tool_call → seguimos flujo normal
                msg = msg2
                tool_calls = tool_calls2
                content = content2
            else:
                reply = (
                    "Para ejecutar esa acción necesito generar un plan ejecutable, "
                    "pero no pude estructurarlo. Intenta de nuevo."
                )
                store.add_message(chat_id, "assistant", reply)
                store.chat_preview_title(chat_id)
                return {"reply": reply}

        #Guard final: si sigue sin tool_calls y el modelo pide confirmación, bloquear si no hay pending real.
        if (not tool_calls) and _looks_like_confirmation_prompt(content or ""):
            try:
                st2 = store.get_state(chat_id) or {}
            except Exception:
                st2 = {}

            if st2.get("pending_run_id") is None:
                log.info(
                    "event=respond.confirmation_text_without_pending blocked=1 "
                    f"content_preview={(content or '')[:160]}"
                )
                reply = (
                    "Puedo ejecutar eso, pero no pude generar un plan ejecutable. "
                    "Repite la petición con un poco más de detalle."
                )
                store.add_message(chat_id, "assistant", reply)
                store.chat_preview_title(chat_id)
                return {"reply": reply}

        # Respuesta normal
        if not tool_calls:
            reply = (content or "").strip() or "(sin respuesta)"
            store.add_message(chat_id, "assistant", reply)
            store.chat_preview_title(chat_id)
            return {"reply": reply}

    # Si llegamos aquí, hay tool_calls y continúa: TOOL CALL → PLAN DRAFT



    # (si no entró al retry, sigues con respond normal)

    # -----------------------------------------
    # TOOL CALL → PLAN DRAFT
    # -----------------------------------------
    log.info("event=decision mode=tool_call")

    call = tool_calls[0]
    args = call.get("function", {}).get("arguments")

    log.info(
        "event=tool.args "
        f"is_dict={isinstance(args, dict)} "
        f"preview={str(args)[:200]}"
    )

    if not isinstance(args, dict):
        reply = "No pude estructurar una acción ejecutable (arguments inválidos)."
        store.add_message(chat_id, "assistant", reply)
        store.chat_preview_title(chat_id)
        log.info(f"event=tool.args.invalid preview={str(args)[:200]}")
        return {"reply": reply}

    log.info(f"event=tool.args.valid keys={list(args.keys())}")

    mcp_id = (args.get("mcp_id") or "").strip()
    method = (args.get("method") or "").strip().upper()
    path = (args.get("path") or "").strip()
    query = args.get("query") if isinstance(args.get("query"), dict) else None
    body = args.get("body")

    if not (mcp_id and method and path):
        reply = "La solicitud de herramienta es inválida (faltan campos)."
        store.add_message(chat_id, "assistant", reply)
        store.chat_preview_title(chat_id)
        log.info(
            "event=tool.reject reason=missing_fields "
            f"mcp_id={bool(mcp_id)} method={bool(method)} path={bool(path)}"
        )
        return {"reply": reply}

    log.info(f"event=plan.create mcp_id={mcp_id} method={method} path={path}")

    m = mcp_store.get_mcp(mcp_id)
    if not m or not m.is_active:
        reply = "El MCP solicitado no existe o está inactivo."
        store.add_message(chat_id, "assistant", reply)
        store.chat_preview_title(chat_id)
        log.info("event=mcp.reject reason=mcp_missing_or_inactive")
        return {"reply": reply}

    if proj and mcp_id not in (proj.mcp_ids or []):
        reply = "Ese MCP no está habilitado para este proyecto."
        store.add_message(chat_id, "assistant", reply)
        store.chat_preview_title(chat_id)
        log.info("event=mcp.reject reason=mcp_not_enabled")
        return {"reply": reply}

    # Normalización específica para /command
    if method == "POST" and path == "/command":
        if isinstance(body, str) and body.strip():
            s = body.strip()
            if (s.startswith("{") and s.endswith("}")) or (s.startswith('{"') and s.endswith('"}')):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, dict):
                        body = parsed
                    else:
                        body = {"cmd": s}
                except Exception:
                    body = {"cmd": s}
            else:
                body = {"cmd": s}

        if isinstance(body, dict):
            if "cmd" not in body and "command" in body:
                body["cmd"] = body.pop("command")
            if "cmd" not in body and "text" in body:
                body["cmd"] = body.pop("text")

            log.info(
                "event=command.body.normalized "
                f"body_type={type(body).__name__} "
                f"has_cmd={isinstance(body, dict) and 'cmd' in body} "
                f"cmd_preview={(body.get('cmd')[:120] if isinstance(body, dict) and isinstance(body.get('cmd'), str) else '')}"
            )

        if not isinstance(body, dict) or "cmd" not in body or not str(body["cmd"]).strip():
            reply = "La llamada a /command requiere body con el campo 'cmd'."
            store.add_message(chat_id, "assistant", reply)
            store.chat_preview_title(chat_id)
            log.info("event=mcp.reject reason=missing_cmd")
            return {"reply": reply}

    goal = f"{method} {path}"

    step = PlanStep(
        title=goal,
        type="mcp_call",
        mcp_id=mcp_id,
        method=method,
        path=path,
        query=query,
        body=body,
    )

    plan = PlanRun(goal=goal, steps=[step])

    run = plan_run_store.create(chat_id=chat_id, plan_id=plan.id, goal=plan.goal)
    plan_run_store.update(
        run.run_id,
        status="draft",
        plan=asdict(plan),
        last_event="plan_draft",
    )

    try:
        store.set_state(chat_id, pending_run_id=run.run_id)
    except Exception as e:
        log.info(f"event=chat.state.set_pending.error err={type(e).__name__}")

    reply = (
        f"📌 **Plan propuesto**\n"
        f"- MCP: `{mcp_id}`\n"
        f"- Acción: `{method} {path}`\n\n"
        f"Responde **confirmo** para ejecutarlo o **cancela** para descartarlo."
    )

    store.add_message(chat_id, "assistant", reply)
    store.chat_preview_title(chat_id)

    log.info(
        f"event=plan.draft chat_id={chat_id} run_id={run.run_id} "
        f"method={method} path={path} mcp_id={mcp_id}"
    )
    log.info(
        f"event=send.done duration_ms={int((time.time()-t_total)*1000)} "
        f"mode=plan_draft run_id={run.run_id}"
    )

    return {"run_id": run.run_id, "status": "draft", "reply": reply}



# ---------------- MCP Routes ----------------

@router.get("/api/mcps")
def api_list_mcps():
    mcps = mcp_store.list_mcps()
    return {
        "ok": True,
        "items": [{**_mcp_to_out(m), "endpoint_count": len(m.endpoints or [])} for m in mcps],
    }


@router.post("/api/mcps")
async def api_register_mcp(payload: MCPRegisterIn):
    try:
        m = await mcp_service.register(
            address=payload.address,
            name=payload.name,
            docs_url=payload.docs_url,
            save_openapi_raw=payload.save_openapi_raw,
            allow_offline=True,
        )
        out = _mcp_to_out(m)
        if not m.openapi_url:
            return JSONResponse(
                {"ok": True, "item": out, "warning": "MCP registrado pero offline; ejecuta /refresh cuando esté disponible."},
                status_code=201,
            )
        return JSONResponse({"ok": True, "item": out}, status_code=201)
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Error inesperado: {e}"}, status_code=500)


@router.get("/api/mcps/{mcp_id}")
def api_get_mcp(mcp_id: str):
    m = mcp_store.get_mcp(mcp_id)
    if not m:
        return JSONResponse({"ok": False, "error": "MCP no encontrado"}, status_code=404)
    return {"ok": True, "item": _mcp_to_out(m)}


@router.patch("/api/mcps/{mcp_id}")
async def api_update_mcp(mcp_id: str, payload: MCPUpdateIn):
    m = mcp_store.get_mcp(mcp_id)
    if not m:
        return JSONResponse({"ok": False, "error": "MCP no encontrado"}, status_code=404)

    ok = mcp_store.update_mcp(mcp_id, name=payload.name, base_url=payload.base_url, docs_url=payload.docs_url)
    if not ok:
        return JSONResponse({"ok": False, "error": "MCP no encontrado"}, status_code=404)

    if payload.base_url is not None:
        try:
            await mcp_service.refresh(mcp_id, save_openapi_raw=False)
        except Exception as e:
            return JSONResponse(
                {"ok": True, "item": _mcp_to_out(mcp_store.get_mcp(mcp_id)), "warning": f"Actualizado, pero refresh falló: {e}"},
                status_code=200,
            )

    return {"ok": True, "item": _mcp_to_out(mcp_store.get_mcp(mcp_id))}


@router.delete("/api/mcps/{mcp_id}")
def api_delete_mcp(mcp_id: str):
    try:
        mcp_service.delete(mcp_id)
        return {"ok": True}
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)


@router.post("/api/mcps/{mcp_id}/active")
def api_set_mcp_active(mcp_id: str, payload: MCPSetActiveIn):
    try:
        m = mcp_service.set_active(mcp_id, payload.active)
        return {"ok": True, "item": _mcp_to_out(m)}
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)


@router.post("/api/mcps/{mcp_id}/refresh")
async def api_refresh_mcp(mcp_id: str):
    try:
        m = await mcp_service.refresh(mcp_id, save_openapi_raw=False)
        return {"ok": True, "item": _mcp_to_out(m)}
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Refresh falló: {e}"}, status_code=400)


# ---------------- Runs (polling) ----------------


@router.get("/api/runs/{run_id}")
def api_get_run(run_id: str, request: Request):
    trace_id = getattr(request.state, "trace_id", request.headers.get("X-Trace-Id", "-"))
    log = get_logger(trace_id)

    log.info(f"event=run.get.start run_id={run_id}")

    r = plan_run_store.get(run_id)
    if not r:
        log.info(f"event=run.get.missing run_id={run_id}")
        return JSONResponse({"error": "Run not found"}, status_code=404)

    status = getattr(r, "status", None)
    last_event = getattr(r, "last_event", None)
    current_step = getattr(r, "current_step_path", None)
    err = getattr(r, "error", None)

    # Si tu store lo tiene:
    chat_id = getattr(r, "chat_id", None)
    plan_id = getattr(r, "plan_id", None)
    updated_ts = getattr(r, "updated_ts", None) or getattr(r, "updated_at", None) or None

    log.info(
        f"event=run.get.ok run_id={run_id} status={status} last_event={last_event} "
        f"current_step={current_step} chat_id={chat_id} plan_id={plan_id} has_error={bool(err)}"
    )

    payload = plan_run_store.to_dict(r)

    # Asegurar campos útiles (aunque to_dict no los incluya)
    payload.setdefault("run_id", run_id)
    if chat_id is not None:
        payload.setdefault("chat_id", chat_id)
    if plan_id is not None:
        payload.setdefault("plan_id", plan_id)
    if updated_ts is not None:
        payload.setdefault("updated_ts", updated_ts)
    if err:
        payload.setdefault("error", err)

    return {"run": payload}




@router.post("/api/runs/{run_id}/start")
async def api_start_run(run_id: str, request: Request):
    trace_id = getattr(request.state, "trace_id", request.headers.get("X-Trace-Id", "-"))
    log = get_logger(trace_id)

    r = plan_run_store.get(run_id)
    if not r:
        return JSONResponse({"error": "Run not found"}, status_code=404)

    # Idempotencia: si ya está en queued/running/done/error, no re-ejecutar
    if r.status in ("queued", "running"):
        return {"ok": True, "run_id": run_id, "status": r.status}
    if r.status in ("done", "error", "canceled"):
        return {"ok": True, "run_id": run_id, "status": r.status}

    # Solo draft puede arrancar
    if r.status != "draft":
        return {"ok": True, "run_id": run_id, "status": r.status}

    # CAS: draft -> queued (solo una vez)
    try:
        ok = plan_run_store.try_mark_queued(run_id)  # debes tener este método
    except Exception as e:
        ok = False
        log.info(f"event=plan.try_mark_queued.error run_id={run_id} err={type(e).__name__}")

    if not ok:
        rr = plan_run_store.get(run_id)
        st = rr.status if rr else "unknown"
        return {"ok": True, "run_id": run_id, "status": st}

    # Reutiliza el flujo único (crea task + set_state + mensajes + on_done etc.)
    return await _start_run_from_draft(run_id, request)
