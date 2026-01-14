import uuid
import json
import time
import os
import asyncio
from typing import Optional, List, Any, Dict

from fastapi import APIRouter, Request, Response

from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.app_state import client, store, mcp_store, mcp_service, plan_run_store, settings


from dataclasses import asdict

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


def _router_system_prompt(tools_ctx: list[dict]) -> str:
    return (
        "Eres un router de herramientas.\n"
        "Tienes acceso a servicios MCP descritos en TOOLS.\n"
        "Tu tarea es decidir y devolver SOLO un JSON (una línea) con la acción.\n\n"
        "IMPORTANTE:\n"
        "- Responde SIEMPRE con un JSON válido en UNA SOLA LÍNEA.\n"
        "- NO uses markdown.\n"
        "- NO escribas texto fuera del JSON.\n\n"
        "ACCIONES DISPONIBLES:\n"
        "1) Responder sin llamar:\n"
        "{\"action\":\"respond\",\"text\":\"...\"}\n\n"
        "2) Llamar MCP (un solo paso):\n"
        "{\"action\":\"mcp_call\",\"mcp_id\":\"...\",\"method\":\"GET|POST|PUT|PATCH|DELETE\",\"path\":\"/...\",\"query\":{...},\"body\":{...}}\n\n"
        "3) Plan multi-step:\n"
        "{\"action\":\"plan\",\"goal\":\"...\",\"stop_on_error\":true,\"steps\":["
        "{\"type\":\"mcp_call\",\"title\":\"...\",\"mcp_id\":\"...\",\"method\":\"POST\",\"path\":\"/command\",\"body\":{\"cmd\":\"...\"}}"
        "]}\n\n"
        "REGLA DE ORO:\n"
        "- Si el usuario pide pasos/secuencia/primero-luego, DEBES devolver action=plan.\n\n"
        "REGLAS:\n"
        "- Solo puedes usar mcp_id/method/path que existan en TOOLS.\n"
        "- No inventes endpoints.\n"
        "- Si falta info crítica, pide aclaración con action=respond.\n"
        "- Para comandos usa POST /command con body {\"cmd\":\"...\"}.\n\n"
        f"TOOLS={json.dumps(tools_ctx, ensure_ascii=False)}"
    )


def _format_plan_for_chat(goal: str, steps_raw: list[dict]) -> str:
    lines: list[str] = [f"🧠 Te propongo este plan: **{goal}**", ""]
    for i, s in enumerate(steps_raw, start=1):
        title = str(s.get("title") or "").strip() or f"Paso {i}"
        lines.append(f"{i}) {title}")
        subs = s.get("substeps")
        if isinstance(subs, list) and subs:
            for j, ss in enumerate(subs, start=1):
                st = str(ss.get("title") or "").strip() or f"Subpaso {i}.{j}"
                lines.append(f"   - {i}.{j} {st}")
    lines += [
        "",
        "Si quieres que lo ejecute paso a paso, responde **confirmo**.",
        "Si quieres descartarlo, responde **cancela**.",
    ]
    return "\n".join(lines)


async def _start_run_from_draft(run_id: str, request: Request) -> JSONResponse | dict:
    trace_id = getattr(request.state, "trace_id", request.headers.get("X-Trace-Id", "-"))
    log = get_logger(trace_id)

    r = plan_run_store.get(run_id)
    if not r:
        return JSONResponse({"error": "Run not found"}, status_code=404)

    if r.status != "draft":
        return JSONResponse({"error": f"Run no está en draft (status={r.status})"}, status_code=409)

    plan_dict = r.plan
    if not isinstance(plan_dict, dict):
        return JSONResponse({"error": "Run draft sin plan"}, status_code=500)

    from src.agent.plan_models import PlanRun, PlanStep

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

    plan_run_store.update(run_id, status="queued", last_event="run_start_queued")

    store.add_message(r.chat_id, "assistant", f"Confirmado. Ejecutando plan… (run_id={run_id})")

    # Limpia el pending_run_id y marca active_run_id (si tu store.py ya tiene chat_state)
    try:
        store.set_state(r.chat_id, pending_run_id=None, active_run_id=run_id)
    except Exception:
        pass

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
            run_store=plan_run_store,
            client=client,
        )
    )

    def _on_done(t: asyncio.Task):
        try:
            t.result()
        except Exception as e:
            log.exception(f"event=plan.bg.task_error run_id={run_id} err={type(e).__name__}: {e}")

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
            log.warning(f"Recovering stale run {r.id} (status={r.status})")
            plan_run_store.update(
                r.id,
                status="error",
                last_event="recovered_after_reload",
                error="Run detenido por recarga del servidor",
            )
            try:
                store.add_message(
                    r.chat_id,
                    "assistant",
                    f"El plan fue detenido por una recarga del servidor.\n(run_id={r.id})",
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

    log.info(
        f"event=debug.env LOG_PROMPTS={os.getenv('LOG_PROMPTS')} LOG_DIR={os.getenv('LOG_DIR')} LOG_PROMPT_FILE={os.getenv('LOG_PROMPT_FILE')}"
    )
    log.info(f"event=debug.prompts_enabled value={prompts_enabled()}")
    log.info(f"event=debug.prompt_logger_handlers count={len(plog.logger.handlers)}")

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

    # 1) Guardar mensaje usuario
    store.add_message(chat_id, "user", text)

    # 2) Confirmación por CHAT (si hay plan pendiente)
    try:
        state = store.get_state(chat_id)
    except Exception as e:
        log.info(f"event=chat.state.error err={type(e).__name__}")
        state = {}

    pending_run_id = state.get("pending_run_id")
    active_run_id = state.get("active_run_id")
    log.info(f"event=chat.state chat_id={chat_id} pending_run_id={pending_run_id} active_run_id={active_run_id}")

    if pending_run_id:
        t = text.lower().strip()
        log.info(f"event=send.confirmation.detected chat_id={chat_id} pending_run_id={pending_run_id} user_text={t}")

        if t in ("confirmo", "sí", "si", "ok", "dale", "ejecuta", "proceder", "continuar"):
            log.info(f"event=send.confirmation.accept chat_id={chat_id} run_id={pending_run_id}")
            res = await _start_run_from_draft(pending_run_id, request)
            log.info(f"event=send.confirmation.accept.done chat_id={chat_id} run_id={pending_run_id}")
            return res

        if t in ("cancela", "cancelar", "no", "detener", "para"):
            log.info(f"event=send.confirmation.cancel chat_id={chat_id} run_id={pending_run_id}")
            try:
                store.set_state(chat_id, pending_run_id=None)
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

    # 3) Construcción prompts / router
    messages_for_llm = store.get_messages_payload(chat_id)

    summ = summarize_messages(messages_for_llm)
    log.info(
        f"event=prompt.base.built chat_id={chat_id} msg_count={summ['count']} total_chars={summ['total_chars']} roles={summ['roles']}"
    )
    plog.info("event=prompt.base.content\n" + serialize_messages_for_promptlog(messages_for_llm))

    proj = store.get_project(c.project_id)
    tools_ctx = _build_tools_ctx_for_project(proj) if proj else []
    router_sys = _router_system_prompt(tools_ctx)

    log.info(f"event=prompt.router.system chat_id={chat_id} chars={len(router_sys)} mcp_count={len(tools_ctx)}")
    plog.info("event=prompt.router.system.content\n" + serialize_text_for_promptlog("ROUTER_SYSTEM", router_sys))

    # Para el router: system del router + historial real del chat
    chat_messages = [{"role": m.role, "content": m.content} for m in c.messages]
    router_messages = [{"role": "system", "content": router_sys}] + chat_messages

    rs = summarize_messages(router_messages)
    log.info(
        f"event=prompt.router.built chat_id={chat_id} msg_count={rs['count']} total_chars={rs['total_chars']} roles={rs['roles']}"
    )
    plog.info("event=prompt.router.content\n" + serialize_messages_for_promptlog(router_messages))

    # 4) Router call
    t_router = time.time()
    try:
        raw = client.chat(router_messages, temperature=0)
    except Exception as e:
        log.info(f"event=router.call.error duration_ms={int((time.time()-t_router)*1000)} err={type(e).__name__}")
        return JSONResponse({"error": f"Error API: {e}"}, status_code=500)

    log.info(f"event=router.call.done duration_ms={int((time.time()-t_router)*1000)} raw_len={len(raw)}")

    try:
        decision = json.loads(raw)
    except Exception:
        log.info("event=router.json.parse_error fallback=raw_text")
        store.add_message(chat_id, "assistant", raw)
        store.chat_preview_title(chat_id)
        log.info(f"event=send.done duration_ms={int((time.time()-t_total)*1000)} mode=fallback_text")
        return {"reply": raw, "warning": "El modelo no devolvió JSON."}

    action = (decision.get("action") or "").strip()
    log.info(f"event=router.decision.parsed action={action} chat_id={chat_id}")

    # -----------------------------------------
    # respond
    # -----------------------------------------
    if action == "respond":
        reply = str(decision.get("text") or "").strip() or "(sin respuesta)"
        store.add_message(chat_id, "assistant", reply)
        store.chat_preview_title(chat_id)
        log.info(f"event=send.done duration_ms={int((time.time()-t_total)*1000)} mode=respond")
        return {"reply": reply}

    # -----------------------------------------
    # plan -> DRAFT (confirmación por chat)
    # -----------------------------------------
    if action == "plan":
        from dataclasses import asdict
        from src.agent.plan_models import PlanRun, PlanStep

        def _parse_step(obj: Dict[str, Any]) -> PlanStep:
            st = PlanStep(
                title=str(obj.get("title") or ""),
                type=str(obj.get("type") or "note"),
                mcp_id=(obj.get("mcp_id") or None),
                method=(str(obj.get("method")).upper() if obj.get("method") else None),
                path=(obj.get("path") or None),
                query=(obj.get("query") if isinstance(obj.get("query"), dict) else None),
                body=obj.get("body"),
            )
            subs = obj.get("substeps")
            if isinstance(subs, list):
                st.substeps = [_parse_step(x) for x in subs if isinstance(x, dict)]
                if st.substeps and st.type != "subplan":
                    st.type = "subplan"
            return st

        goal = str(decision.get("goal") or "").strip() or "Plan"
        steps_raw = decision.get("steps")

        if not isinstance(steps_raw, list) or not steps_raw:
            reply = "Plan inválido: faltan 'steps'."
            store.add_message(chat_id, "assistant", reply)
            store.chat_preview_title(chat_id)
            log.info("event=plan.reject reason=missing_steps")
            return {"reply": reply}

        plan = PlanRun(goal=goal, steps=[_parse_step(s) for s in steps_raw if isinstance(s, dict)])
        if not plan.steps:
            reply = "Plan inválido: 'steps' no contiene pasos válidos."
            store.add_message(chat_id, "assistant", reply)
            store.chat_preview_title(chat_id)
            log.info("event=plan.reject reason=empty_parsed_steps")
            return {"reply": reply}

        try:
            if hasattr(store, "add_plan"):
                store.add_plan(chat_id, plan)
            else:
                cc = store.get_chat(chat_id)
                if not hasattr(cc, "plan_runs") or cc.plan_runs is None:
                    cc.plan_runs = []
                cc.plan_runs.append(plan)
        except Exception as e:
            log.info(f"event=plan.store.error err={type(e).__name__}")

        run = plan_run_store.create(chat_id=chat_id, plan_id=plan.id, goal=plan.goal)
        plan_run_store.update(run.run_id, status="draft", plan=asdict(plan), last_event="plan_draft")

        try:
            store.set_state(chat_id, pending_run_id=run.run_id)
        except Exception as e:
            log.info(f"event=chat.state.set_pending.error err={type(e).__name__}")

        reply = _format_plan_for_chat(goal, steps_raw)
        store.add_message(chat_id, "assistant", reply)
        store.chat_preview_title(chat_id)

        log.info(f"event=plan.draft chat_id={chat_id} run_id={run.run_id} plan_id={plan.id} steps={len(plan.steps)}")
        log.info(f"event=send.done duration_ms={int((time.time()-t_total)*1000)} mode=plan_draft run_id={run.run_id}")

        return {"run_id": run.run_id, "status": "draft", "reply": reply}

    # -----------------------------------------
    # mcp_call -> EJECUTAR ASYNC COMO PLAN (1 paso)
    # -----------------------------------------
    if action == "mcp_call":
        from dataclasses import asdict
        from src.agent.plan_models import PlanRun, PlanStep

        mcp_id = (decision.get("mcp_id") or "").strip()
        method = (decision.get("method") or "").strip().upper()
        path = (decision.get("path") or "").strip()
        query = decision.get("query") if isinstance(decision.get("query"), dict) else None
        body = decision.get("body")

        # Normalización específica para /command
        if method == "POST" and path == "/command":
            if isinstance(body, dict):
                if "cmd" not in body and "command" in body:
                    body["cmd"] = body.pop("command")
                if "cmd" not in body and "text" in body:
                    body["cmd"] = body.pop("text")
            if not isinstance(body, dict) or "cmd" not in body or not str(body["cmd"]).strip():
                reply = "La llamada a /command requiere body JSON con el campo 'cmd'."
                store.add_message(chat_id, "assistant", reply)
                store.chat_preview_title(chat_id)
                log.info("event=mcp.reject reason=missing_cmd")
                return {"reply": reply}

        if not (mcp_id and method and path):
            reply = "La solicitud de herramienta es inválida (faltan campos)."
            store.add_message(chat_id, "assistant", reply)
            store.chat_preview_title(chat_id)
            log.info("event=mcp.reject reason=missing_fields")
            return {"reply": reply}

        # Validaciones MCP (rápidas)
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

        # Convertir a plan de 1 paso y ejecutarlo en background (SIN bloquear el request)
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
        plan_run_store.update(run.run_id, status="queued", plan=asdict(plan), last_event="run_start_queued")

        # Estado del chat: marcar active_run_id (para que la UI lo sepa si lo usa)
        try:
            store.set_state(chat_id, active_run_id=run.run_id)
        except Exception as e:
            log.info(f"event=chat.state.set_active.error err={type(e).__name__}")

        # Mensaje inmediato tipo ChatGPT
        store.add_message(chat_id, "assistant", f"✅ Entendido. Ejecutando… (run_id={run.run_id})")
        store.chat_preview_title(chat_id)

        log.info(
            f"event=mcp.converted_to_plan chat_id={chat_id} run_id={run.run_id} mcp_id={mcp_id} method={method} path={path}"
        )

        # Lanzar task
        try:
            asyncio.create_task(
                run_plan_in_background(
                    run_id=run.run_id,
                    chat_id=chat_id,
                    plan=plan,
                    store=store,
                    mcp_store=mcp_store,
                    proj=proj,
                    trace_id=trace_id,
                    log=log,
                    run_store=plan_run_store,
                    client=client,
                )
            )
            log.info(f"event=bg.task.created chat_id={chat_id} run_id={run.run_id}")
        except Exception as e:
            log.info(f"event=bg.task.create_error err={type(e).__name__}")
            plan_run_store.update(run.run_id, status="error", error=f"{type(e).__name__}: {e}", last_event="bg_create_error")
            store.add_message(chat_id, "assistant", f"❌ No pude iniciar la ejecución en background: {e}")
            store.chat_preview_title(chat_id)

        log.info(f"event=send.done duration_ms={int((time.time()-t_total)*1000)} mode=mcp_call_async run_id={run.run_id}")
        return {"ok": True, "run_id": run.run_id, "status": "queued"}

    reply = "Respuesta inválida del modelo (action desconocida)."
    store.add_message(chat_id, "assistant", reply)
    store.chat_preview_title(chat_id)
    log.info(f"event=send.done duration_ms={int((time.time()-t_total)*1000)} mode=unknown_action action={action}")
    return {"reply": reply}



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

    if r.status != "draft":
        return JSONResponse({"error": f"Run no está en draft (status={r.status})"}, status_code=409)

    plan_dict = r.plan
    if not isinstance(plan_dict, dict):
        return JSONResponse({"error": "Run draft sin plan"}, status_code=500)

    from src.agent.plan_models import PlanRun, PlanStep

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
        plan.id = str(plan_dict["id"])

    chat = store.get_chat(r.chat_id)
    proj = store.get_project(chat.project_id) if chat else None


    plan_run_store.update(run_id, status="queued", plan=plan_dict, last_event="run_start_queued")


    store.add_message(r.chat_id, "assistant", f"Confirmado. Ejecutando plan… (run_id={run_id})")

    try:
        store.set_state(r.chat_id, pending_run_id=None, active_run_id=run_id)
    except Exception:
        pass


    asyncio.create_task(
        run_plan_in_background(
            run_id=run_id,
            chat_id=r.chat_id,
            plan=plan,
            store=store,
            mcp_store=mcp_store,
            proj=proj,
            trace_id=trace_id,
            log=log,
            run_store=plan_run_store,
            client=client,  # <-- usa el cliente global correcto
        )
    )

    return {"ok": True, "run_id": run_id, "status": "queued"}


