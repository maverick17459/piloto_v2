from __future__ import annotations

import asyncio
import time
import json
from dataclasses import asdict
from typing import Any, Dict, Optional, Protocol

from src.agent.plan_models import PlanRun, PlanStep
from src.agent.plan_executor import execute_plan_run
from src.agent.agent_reasoner import reason_about_command_failure
from src.mcp.invoke_sync import MCPCall, invoke_mcp_sync, MCPInvokeError


# -------------------------------------------------
# Utilidades de seguridad
# -------------------------------------------------

class PlanRunStoreLike(Protocol):
    def get(self, run_id: str): ...
    def update(self, run_id: str, **kwargs): ...


def _looks_dangerous(cmd: str) -> bool:
    deny = [
        "rm -rf /",
        "mkfs",
        "dd if=",
        "shutdown",
        "reboot",
        "poweroff",
        "format c:",
        "diskpart",
        "bcdedit",
        "reg delete",
        "del /s /q c:\\",
        "rd /s /q c:\\",
        ":(){ :|:& };:",
    ]
    c = (cmd or "").lower()
    return any(x in c for x in deny)


def _short(x: Any, max_len: int = 240) -> str:
    s = str(x).replace("\r", "")
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _is_command_call(method: str, path: str) -> bool:
    return method.upper() == "POST" and path == "/command"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _command_failed(result: Any) -> tuple[bool, str]:
    """
    Define éxito real para /command basado en payload.
    """
    if not isinstance(result, dict):
        return True, "Resultado inválido (no es JSON)."

    status = str(result.get("status") or "").lower()
    exit_code = result.get("exit_code")

    stdout = result.get("stdout")
    stderr = result.get("stderr")

    msg = ""
    if isinstance(stderr, str) and stderr.strip():
        msg = stderr.strip()
    elif isinstance(stdout, str) and stdout.strip():
        msg = stdout.strip()

    try:
        ec = int(exit_code) if exit_code is not None else 0
    except Exception:
        ec = 0

    if status == "ok" and ec == 0:
        return False, msg or "OK"

    return True, msg or f"Comando falló (status={status}, exit_code={exit_code})"


# -------------------------------------------------
# run_plan_in_background
# -------------------------------------------------

async def run_plan_in_background(
    *,
    run_id: str,
    chat_id: str,
    plan: PlanRun,
    store,
    mcp_store,
    proj,
    trace_id: str,
    log,
    run_store: PlanRunStoreLike,
    client,
) -> None:

    run_store.update(run_id, status="running", last_event="runner_started")

    # Asegura consistencia: si este run debería ser el activo, lo anotamos (sin pisar si ya hay otro)
    try:
        st = store.get_state(chat_id) or {}
        updates = {}

        if st.get("active_run_id") in (None, run_id):
            updates["active_run_id"] = run_id

        # Si por error sigue como pending, límpialo (evita "pending+active" simultáneo)
        if st.get("pending_run_id") == run_id:
            updates["pending_run_id"] = None

        if updates:
            store.set_state(chat_id, **updates)

    except Exception as e:
        log.info(
            "event=runner.ensure_active.error run_id=%s err=%s msg=%s",
            run_id,
            type(e).__name__,
            e,
        )

    loop = asyncio.get_running_loop()

    # -----------------------------
    # helpers thread-safe
    # -----------------------------

    def _ui_envelope(kind: str, payload: dict) -> str:
        """
        Sobre UI versionado para que el frontend renderice cards y bloques pre/code
        sin depender de Markdown.
        """
        return "PILOTO_MSG_V1\n" + json.dumps(
            {"v": 1, "kind": kind, **payload},
            ensure_ascii=False,
        )

    def _safe_call(fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as e:
            log.info(
                f"event=runner.safe_call.error run_id={run_id} fn={getattr(fn,'__name__',fn)} "
                f"err={type(e).__name__} msg={e}"
            )

    def safe_add_message(role: str, content: str):
        loop.call_soon_threadsafe(_safe_call, store.add_message, chat_id, role, content)

    def safe_chat_preview_title():
        loop.call_soon_threadsafe(_safe_call, store.chat_preview_title, chat_id)

    MAX_ATTEMPTS_PER_COMMAND_STEP = 3
    PLAN_TIMEOUT_S = 10 * 60

    # -----------------------------
    # inicio
    # -----------------------------

    run_store.update(run_id, last_event="run_start", current_step_path=None)
    log.info(f"event=runner.start run_id={run_id} chat_id={chat_id} goal={plan.goal}")

    # Mensaje de inicio (estructurado)
    safe_add_message(
        "assistant",
        _ui_envelope(
            "run_start",
            {
                "run_id": run_id,
                "goal": plan.goal,
                "chat_id": chat_id,
                "trace_id": trace_id,
                "ts_ms": _now_ms(),
            },
        ),
    )

    # -----------------------------
    # validación de pasos
    # -----------------------------

    def validate_step(step: PlanStep) -> Optional[str]:
        if step.type not in ("note", "mcp_call", "subplan"):
            return f"Tipo inválido: {step.type}"
        if step.type == "mcp_call":
            if not (step.mcp_id and step.method and step.path):
                return "mcp_call incompleto: requiere mcp_id, method, path"
        return None

    # -----------------------------
    # eventos UX (AHORA ESTRUCTURADOS)
    # -----------------------------

    def emit(kind: str, step_path: Optional[str], title: str, payload: Optional[dict] = None):
        run_store.update(run_id, current_step_path=step_path, last_event=kind)

        p = {
            "run_id": run_id,
            "goal": plan.goal,
            "step_path": step_path,
            "title": title,
            "ts_ms": _now_ms(),
        }
        if payload:
            p.update(payload)

        safe_add_message("assistant", _ui_envelope(kind, p))

    # -----------------------------
    # invoke MCP (centralizado)
    # -----------------------------

    def invoke_step(
        mcp_id: str,
        method: str,
        path: str,
        query: Optional[Dict[str, Any]],
        body: Any,
    ):
        m = mcp_store.get_mcp(mcp_id)
        if not m or not m.is_active:
            return 404, {"error": "mcp_missing_or_inactive"}

        if proj and mcp_id not in (proj.mcp_ids or []):
            return 403, {"error": "mcp_not_enabled_for_project"}

        try:
            return invoke_mcp_sync(
                mcp=m,
                call=MCPCall(
                    mcp_id=m.id,
                    method=method,
                    path=path,
                    query=query,
                    body=body,
                ),
                extra_headers={},
            )
        except MCPInvokeError as e:
            return 500, {"error": "mcp_invoke_error", "detail": str(e)}
        except Exception as e:
            return 500, {"error": "mcp_invoke_exception", "detail": f"{type(e).__name__}: {e}"}

    # -----------------------------
    # retry con agente GPT al final
    # -----------------------------

    def invoke_mcp_with_emit(
        mcp_id: str,
        method: str,
        path: str,
        query: Optional[Dict[str, Any]],
        body: Any,
    ):
        # Ojo: step_path es lo que se mostrará en UI; en tu plan executor probablemente ya tienes un path real.
        # Aquí mantenemos el comportamiento original (path), pero añadimos más contexto en el payload.
        step_path = path or "step"
        title = f"{method} {path}"

        # Inicio del paso con request completo (sin stringify)
        emit(
            "step_start",
            step_path,
            title,
            payload={
                "mcp_id": mcp_id,
                "method": method,
                "path": path,
                "query": query,
                "body": body,
                "attempt": 1,
            },
        )

        attempt = 1
        current_body = body

        while True:
            sc, res = invoke_step(mcp_id, method, path, query, current_body)

            # Error HTTP / MCP
            if not (200 <= int(sc) < 300):
                emit(
                    "step_err",
                    step_path,
                    title,
                    payload={
                        "mcp_id": mcp_id,
                        "method": method,
                        "path": path,
                        "query": query,
                        "body": current_body,
                        "attempt": attempt,
                        "status_code": int(sc),
                        "response": res,
                        "detail": _short(res),
                    },
                )
                return sc, res

            # Caso /command: definimos éxito real por payload
            if _is_command_call(method, path):
                failed, reason = _command_failed(res)

                # OK real
                if not failed:
                    emit(
                        "step_ok",
                        step_path,
                        title,
                        payload={
                            "mcp_id": mcp_id,
                            "method": method,
                            "path": path,
                            "query": query,
                            "body": current_body,
                            "attempt": attempt,
                            "status_code": int(sc),
                            "response": res,
                            "stdout": res.get("stdout") if isinstance(res, dict) else None,
                            "stderr": res.get("stderr") if isinstance(res, dict) else None,
                            "exit_code": res.get("exit_code") if isinstance(res, dict) else None,
                            "status": res.get("status") if isinstance(res, dict) else None,
                        },
                    )
                    return sc, res

                # FALLÓ: si ya no hay más intentos directos, invocamos reasoner para un nuevo cuerpo
                if attempt >= MAX_ATTEMPTS_PER_COMMAND_STEP:
                    stdout = str(res.get("stdout") or "") if isinstance(res, dict) else ""
                    stderr = str(res.get("stderr") or "") if isinstance(res, dict) else ""

                    # Motivo de fallo + último payload recibido
                    emit(
                        "step_retry",
                        step_path,
                        title,
                        payload={
                            "mcp_id": mcp_id,
                            "method": method,
                            "path": path,
                            "query": query,
                            "body": current_body,
                            "attempt": attempt,
                            "status_code": int(sc),
                            "response": res,
                            "reason": reason,
                            "detail": "Invocando agente reasoner para proponer un nuevo intento razonado.",
                        },
                    )

                    new_step = reason_about_command_failure(
                        client=client,
                        goal=plan.goal,
                        step=PlanStep(
                            title=title,
                            type="mcp_call",
                            mcp_id=mcp_id,
                            method=method,
                            path=path,
                            query=query,
                            body=current_body,
                        ),
                        stdout=stdout,
                        stderr=stderr,
                        attempt=attempt,
                        max_attempts=MAX_ATTEMPTS_PER_COMMAND_STEP,
                    )

                    # Si no hay propuesta o es peligrosa: error final
                    if not new_step or _looks_dangerous((new_step.body or {}).get("cmd", "")):
                        emit(
                            "step_err",
                            step_path,
                            title,
                            payload={
                                "mcp_id": mcp_id,
                                "method": method,
                                "path": path,
                                "query": query,
                                "body": current_body,
                                "attempt": attempt,
                                "status_code": int(sc),
                                "response": res,
                                "stdout": stdout,
                                "stderr": stderr,
                                "reason": reason,
                                "detail": "El comando falló y el agente no pudo proponer un retry seguro.",
                            },
                        )
                        return sc, res

                    # Aplicamos retry razonado (nuevo body)
                    attempt += 1
                    current_body = new_step.body

                    emit(
                        "step_retry",
                        step_path,
                        title,
                        payload={
                            "mcp_id": mcp_id,
                            "method": method,
                            "path": path,
                            "query": query,
                            "attempt": attempt,
                            "new_body": current_body,
                            "detail": "El agente propone un nuevo intento razonado.",
                        },
                    )
                    continue

                # Reintento simple (sin reasoner aún)
                attempt += 1
                emit(
                    "step_retry",
                    step_path,
                    title,
                    payload={
                        "mcp_id": mcp_id,
                        "method": method,
                        "path": path,
                        "query": query,
                        "attempt": attempt,
                        "status_code": int(sc),
                        "response": res,
                        "reason": reason,
                        "detail": f"Reintentando comando ({attempt}/{MAX_ATTEMPTS_PER_COMMAND_STEP})",
                    },
                )
                continue

            # llamadas MCP no /command
            emit(
                "step_ok",
                step_path,
                title,
                payload={
                    "mcp_id": mcp_id,
                    "method": method,
                    "path": path,
                    "query": query,
                    "body": current_body,
                    "attempt": attempt,
                    "status_code": int(sc),
                    "response": res,
                },
            )
            return sc, res

    # -----------------------------
    # ejecutar plan determinista
    # -----------------------------

    try:
        final_plan: PlanRun = await asyncio.wait_for(
            asyncio.to_thread(
                execute_plan_run,
                plan=plan,
                trace_id=trace_id,
                invoke_mcp_call=invoke_mcp_with_emit,
                validate_step=validate_step,
            ),
            timeout=PLAN_TIMEOUT_S,
        )

        final_status = "error" if final_plan.status == "error" else "done"

        run_store.update(
            run_id,
            status=final_status,
            plan=asdict(final_plan),
            last_event="run_done",
        )

        # Mensaje final estructurado (para card/summary)
        safe_add_message(
            "assistant",
            _ui_envelope(
                "run_done",
                {
                    "run_id": run_id,
                    "goal": final_plan.goal,
                    "status": final_plan.status,
                    "final_status": final_status,
                    "current_step_path": final_plan.current_step_path or None,
                    "ts_ms": _now_ms(),
                },
            ),
        )

        log.info(f"event=runner.finish run_id={run_id} status={final_status}")

    except asyncio.TimeoutError:
        run_store.update(run_id, status="error", error="TimeoutError: plan_timeout", last_event="run_timeout")
        safe_add_message(
            "assistant",
            _ui_envelope(
                "run_timeout",
                {
                    "run_id": run_id,
                    "goal": plan.goal,
                    "error": "TimeoutError: plan_timeout",
                    "ts_ms": _now_ms(),
                },
            ),
        )

    except asyncio.CancelledError:
        run_store.update(run_id, status="error", error="CancelledError", last_event="runner_cancelled")
        safe_add_message(
            "assistant",
            _ui_envelope(
                "run_cancelled",
                {
                    "run_id": run_id,
                    "goal": plan.goal,
                    "error": "CancelledError",
                    "ts_ms": _now_ms(),
                },
            ),
        )
        raise

    except Exception as e:
        run_store.update(run_id, status="error", error=str(e), last_event="run_error")
        safe_add_message(
            "assistant",
            _ui_envelope(
                "run_error",
                {
                    "run_id": run_id,
                    "goal": plan.goal,
                    "error": f"{type(e).__name__}: {e}",
                    "ts_ms": _now_ms(),
                },
            ),
        )

    finally:
        # Estado final del run (para idempotencia y para evitar "pisar" drafts nuevos)
        try:
            rr = run_store.get(run_id)
            final_status = rr.status if rr else "unknown"
        except Exception as e:
            final_status = "unknown"
            log.info(
                "event=runner.final_status.error run_id=%s err=%s msg=%s",
                run_id,
                type(e).__name__,
                e,
            )

        # Log final uniforme (aunque haya exception arriba)
        log.info(f"event=runner.finalize run_id={run_id} final_status={final_status}")

        # Limpieza de estado del chat SIN pisar drafts nuevos
        try:
            st = store.get_state(chat_id) or {}
            updates = {
                "last_run_id": run_id,
                "last_run_status": final_status,
                "last_run_ts": _now_ms(),
            }

            # Solo limpia active_run_id si este run sigue siendo el activo
            if st.get("active_run_id") == run_id:
                updates["active_run_id"] = None

            # Solo limpia pending_run_id si pending era este run
            if st.get("pending_run_id") == run_id:
                updates["pending_run_id"] = None

            store.set_state(chat_id, **updates)

        except Exception as e:
            # MUY importante: no silenciar. Si esto falla, vuelves al bug.
            log.info(
                "event=runner.state_cleanup.error run_id=%s chat_id=%s err=%s msg=%s",
                run_id,
                chat_id,
                type(e).__name__,
                e,
            )

        # Preview title best-effort
        try:
            safe_chat_preview_title()
        except Exception as e:
            log.info(
                "event=runner.preview_title.error run_id=%s err=%s msg=%s",
                run_id,
                type(e).__name__,
                e,
            )
