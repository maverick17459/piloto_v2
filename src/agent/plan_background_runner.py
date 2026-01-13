from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from src.agent.plan_run_store import PlanRunStore
from src.agent.plan_models import PlanRun, PlanStep
from src.agent.plan_executor import execute_plan_run

from src.mcp.invoke_sync import MCPCall, invoke_mcp_sync, MCPInvokeError


def _short(x: Any, max_len: int = 240) -> str:
    s = str(x).replace("\r", "")
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


async def run_plan_in_background(
    *,
    run_id: str,
    chat_id: str,
    plan: PlanRun,
    store,        # MemoryStore
    mcp_store,    # MCPStore
    proj,         # Project o None
    trace_id: str,
    log,
    run_store: PlanRunStore,
) -> None:
    loop = asyncio.get_running_loop()

    def safe_add_message(role: str, content: str) -> None:
        # IMPORTANTE: siempre escribir al store desde el hilo del event loop
        loop.call_soon_threadsafe(store.add_message, chat_id, role, content)

    def safe_chat_preview_title() -> None:
        loop.call_soon_threadsafe(store.chat_preview_title, chat_id)

    run_store.update(run_id, status="running", last_event="run_start")
    safe_add_message("assistant", f"⏳ Iniciando plan: {plan.goal}\n(run_id={run_id})")

    # -----------------------------
    # Validación de pasos (reusa tu lógica)
    # -----------------------------
    def validate_step(step: PlanStep) -> Optional[str]:
        if step.type not in ("note", "mcp_call", "subplan"):
            return f"Tipo inválido: {step.type}"
        if step.type == "mcp_call":
            if not (step.mcp_id and step.method and step.path):
                return "mcp_call incompleto: requiere mcp_id, method, path"
        return None

    # -----------------------------
    # Emitir eventos a chat + run_store (THREAD-SAFE)
    # -----------------------------
    def emit(kind: str, step_path: Optional[str], title: str, detail: Optional[str] = None) -> None:
        if step_path:
            run_store.update(run_id, current_step_path=step_path, last_event=kind)

        if kind == "step_start":
            safe_add_message("assistant", f"⏳ {step_path}: {title}")
        elif kind == "step_ok":
            msg = f"✅ {step_path}: {title}"
            if detail:
                msg += f"\n{detail}"
            safe_add_message("assistant", msg)
        elif kind == "step_err":
            msg = f"❌ {step_path}: {title}"
            if detail:
                msg += f"\n{detail}"
            safe_add_message("assistant", msg)

    # -----------------------------
    # Invoke MCP (seguridad + allowlist del proyecto)
    # -----------------------------
    def invoke_step(mcp_id: str, method: str, path: str, query: Optional[Dict[str, Any]], body: Any):
        # Normalización específica para /command
        if method.upper() == "POST" and path == "/command":
            if isinstance(body, dict):
                if "cmd" not in body and "command" in body:
                    body["cmd"] = body.pop("command")
                if "cmd" not in body and "text" in body:
                    body["cmd"] = body.pop("text")
            if not isinstance(body, dict) or "cmd" not in body or not str(body["cmd"]).strip():
                return 422, {"error": "missing_cmd", "detail": "Body requiere campo 'cmd'."}

        m = mcp_store.get_mcp(mcp_id)
        if not m or not m.is_active:
            return 404, {"error": "mcp_missing_or_inactive"}

        # seguridad extra: debe pertenecer al proyecto
        if proj and mcp_id not in (proj.mcp_ids or []):
            return 403, {"error": "mcp_not_enabled_for_project"}

        try:
            return invoke_mcp_sync(
                mcp=m,
                call=MCPCall(mcp_id=m.id, method=method, path=path, query=query, body=body),
                extra_headers={},
            )
        except MCPInvokeError as e:
            return 500, {"error": "mcp_invoke_error", "detail": str(e)}

    # -----------------------------
    # Wrappers con eventos
    # -----------------------------
    def validate_step_with_emit(step: PlanStep) -> Optional[str]:
        err = validate_step(step)
        if err:
            step_path = getattr(plan, "current_step_path", None) or (step.title or "step")
            emit("step_err", step_path, step.title or "Paso", err)
        return err

    def invoke_mcp_with_emit(
        mcp_id: str,
        method: str,
        path: str,
        query: Optional[Dict[str, Any]],
        body: Any,
    ):
        step_path = getattr(plan, "current_step_path", None) or (path or "step")
        title = f"{method} {path}"
        emit("step_start", step_path, title)

        sc, res = invoke_step(mcp_id, method, path, query, body)

        if 200 <= int(sc) < 300:
            emit("step_ok", step_path, title, _short(res))
        else:
            emit("step_err", step_path, title, _short(res))

        return sc, res

    # -----------------------------
    # Ejecutar el plan en background
    # -----------------------------
    try:
        final_plan: PlanRun = await asyncio.to_thread(
            execute_plan_run,
            plan=plan,
            trace_id=trace_id,
            invoke_mcp_call=invoke_mcp_with_emit,
            validate_step=validate_step_with_emit,
        )

        final_status = "error" if getattr(final_plan, "status", "") == "error" else "done"
        run_store.update(run_id, status=final_status, plan=final_plan.__dict__, last_event="run_done")

        safe_add_message(
            "assistant",
            f"Plan '{final_plan.goal}' finalizado con estado: {final_plan.status}. Último paso: {final_plan.current_step_path or '-'}",
        )

    except Exception as e:
        run_store.update(run_id, status="error", error=f"{type(e).__name__}: {e}", last_event="run_error")
        safe_add_message("assistant", f"❌ Error ejecutando plan\n{type(e).__name__}: {e}")

    finally:
        safe_chat_preview_title()
        r = run_store.get(run_id)
        log.info(f"event=plan.bg.done run_id={run_id} status={r.status if r else 'missing'}")
