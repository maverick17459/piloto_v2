from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

from src.agent.plan_run_store import PlanRunStore
from src.agent.plan_models import PlanRun, PlanStep
from src.agent.plan_executor import execute_plan_run

from src.mcp.invoke_sync import MCPCall, invoke_mcp_sync, MCPInvokeError


def _looks_dangerous(cmd: str) -> bool:
    c = (cmd or "").lower()

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
        ":(){ :|:& };:",  # fork bomb
    ]
    return any(x in c for x in deny)


def _normalize_cmd(cmd: str) -> str:
    cmd = (cmd or "").strip()
    if len(cmd) > 2000:
        cmd = cmd[:2000]
    return cmd


def _short(x: Any, max_len: int = 240) -> str:
    s = str(x).replace("\r", "")
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _is_command_call(method: str, path: str) -> bool:
    return method.upper() == "POST" and path == "/command"


def _command_failed(result: Any) -> tuple[bool, str]:
    """
    Para /command: define éxito real por payload.status y exit_code.
    Devuelve (failed, message).
    """
    if not isinstance(result, dict):
        return True, "Resultado /command inválido (no es JSON objeto)."

    status = str(result.get("status") or "").lower().strip()
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

    reason = msg or f"/command falló (status={status or 'unknown'}, exit_code={exit_code})"
    return True, reason


def _fix_windows_del_command(cmd: str, err_text: str) -> Optional[str]:
    """
    Reparación simple (determinista) para Windows CMD cuando falla DEL por rutas con /.
    Ejemplo: del C:/Users/... -> del "C:\\Users\\..."
    """
    c = (cmd or "").strip()
    e = (err_text or "").lower()

    if not c.lower().startswith("del "):
        return None

    target = c[4:].strip()
    if not target:
        return None

    target2 = target.replace("/", "\\").strip()

    if target2.startswith('"') and target2.endswith('"'):
        return f'del {target2}'

    if "modificador no es válido" in e or "invalid switch" in e or "\\users" in target2.lower():
        return f'del "{target2}"'

    return f'del "{target2}"'


# CAMBIO: ahora devuelve (cmd, why)
def _repair_command_with_gpt(
    *,
    client,
    goal: str,
    cmd: str,
    error_text: str,
    attempt: int,
    max_attempts: int,
) -> Optional[Tuple[str, str]]:
    """
    Pide al modelo una corrección del comando.
    Devuelve (cmd_corregido, why) o None.
    """
    cmd = _normalize_cmd(cmd)
    error_text = (error_text or "").strip()
    if len(error_text) > 1500:
        error_text = error_text[:1500]

    system = (
        "Eres un asistente experto en ejecución de comandos en Windows.\n"
        "Tu tarea es corregir un comando que falló.\n"
        "Puedes usar CMD.exe o PowerShell SOLO si es necesario.\n"
        "NO uses bash, wsl, sh ni shells de Linux.\n"
        "Devuelve SIEMPRE un JSON válido, sin markdown ni texto extra.\n"
        "Responde solo con: {\"cmd\": \"...\", \"why\": \"...\"}\n"
        "Mantén la intención original del comando.\n"
        "Si falta un dato imposible de inferir, devuelve cmd vacío.\n"
        "No propongas comandos destructivos peligrosos.\n"
    )

    user = (
        f"OBJETIVO: {goal}\n"
        f"INTENTO {attempt}/{max_attempts}\n"
        f"COMANDO FALLIDO:\n{cmd}\n\n"
        f"ERROR/OUTPUT:\n{error_text}\n\n"
        "Devuelve una versión corregida del comando."
    )

    raw = client.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
    )

    try:
        import json
        obj = json.loads(raw)
    except Exception:
        return None

    new_cmd = _normalize_cmd(str(obj.get("cmd") or ""))
    why = str(obj.get("why") or "").strip()

    if not new_cmd:
        return None
    if new_cmd.strip() == cmd.strip():
        return None
    if _looks_dangerous(new_cmd):
        return None

    return new_cmd, why



async def run_plan_in_background(
    *,
    run_id: str,
    chat_id: str,
    plan: "PlanRun",
    store,
    mcp_store,
    proj,
    trace_id: str,
    log,
    run_store: "PlanRunStore",
    client,
) -> None:
    run_store.update(run_id, status="running", last_event="runner_started")

    loop = asyncio.get_running_loop()

    def _safe_call(fn, *args):
        try:
            fn(*args)
        except Exception as e:
            # IMPORTANTÍSIMO: aquí verás si se rompe por tamaño o por store
            log.info(
                f"event=runner.safe_call.error run_id={run_id} fn={getattr(fn, '__name__', str(fn))} "
                f"err={type(e).__name__} msg={e}"
            )

    def safe_add_message(role: str, content: str) -> None:
        # Log tamaño del mensaje (ipconfig puede ser grande)
        try:
            clen = len(content or "")
        except Exception:
            clen = -1

        log.info(f"event=runner.msg.enqueue run_id={run_id} role={role} chars={clen}")
        loop.call_soon_threadsafe(_safe_call, store.add_message, chat_id, role, content)

    def safe_chat_preview_title() -> None:
        loop.call_soon_threadsafe(_safe_call, store.chat_preview_title, chat_id)

    MAX_ATTEMPTS_PER_COMMAND_STEP = 3
    PLAN_TIMEOUT_S = 10 * 60  # 10 minutos

    # Estado inicio
    run_store.update(run_id, last_event="run_start", current_step_path=None)
    log.info(f"event=runner.start run_id={run_id} chat_id={chat_id} goal={plan.goal}")
    safe_add_message("assistant", f"⏳ Iniciando plan: {plan.goal}\n(run_id={run_id})")

    # -----------------------------
    # Validación de pasos
    # -----------------------------
    def validate_step(step: "PlanStep") -> Optional[str]:
        if step.type not in ("note", "mcp_call", "subplan"):
            return f"Tipo inválido: {step.type}"
        if step.type == "mcp_call":
            if not (step.mcp_id and step.method and step.path):
                return "mcp_call incompleto: requiere mcp_id, method, path"
        return None

    # -----------------------------
    # Emitir eventos
    # -----------------------------
    def emit(kind: str, step_path: Optional[str], title: str, detail: Optional[str] = None) -> None:
        # Persistir last_event + step actual para el polling
        run_store.update(run_id, current_step_path=step_path, last_event=kind)

        log.info(
            f"event=runner.emit run_id={run_id} kind={kind} step_path={step_path or '-'} "
            f"title={title} detail_len={len(detail or '')}"
        )

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
        elif kind == "step_retry":
            msg = f"⚠️ {step_path}: {title}\n{detail or 'Reintentando…'}"
            safe_add_message("assistant", msg)

    # -----------------------------
    # Invoke MCP
    # -----------------------------
    def invoke_step(mcp_id: str, method: str, path: str, query: Optional[Dict[str, Any]], body: Any):
        if _is_command_call(method, path):
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

        if proj and mcp_id not in (proj.mcp_ids or []):
            return 403, {"error": "mcp_not_enabled_for_project"}

        log.info(
            f"event=mcp.invoke.start run_id={run_id} mcp_id={mcp_id} method={method} path={path} "
            f"body_type={type(body).__name__}"
        )

        try:
            return invoke_mcp_sync(
                mcp=m,
                call=MCPCall(mcp_id=m.id, method=method, path=path, query=query, body=body),
                extra_headers={},
            )
        except MCPInvokeError as e:
            log.info(f"event=mcp.invoke.error run_id={run_id} err=MCPInvokeError msg={e}")
            return 500, {"error": "mcp_invoke_error", "detail": str(e)}
        except Exception as e:
            log.info(f"event=mcp.invoke.error run_id={run_id} err={type(e).__name__} msg={e}")
            return 500, {"error": "mcp_invoke_exception", "detail": f"{type(e).__name__}: {e}"}

    # -----------------------------
    # Wrappers con eventos
    # -----------------------------
    def validate_step_with_emit(step: "PlanStep") -> Optional[str]:
        err = validate_step(step)
        if err:
            emit("step_err", "?", step.title or "Paso", err)
        return err

    def _invoke_command_with_auto_retry(
        step_path: str,
        title: str,
        mcp_id: str,
        method: str,
        path: str,
        query: Optional[Dict[str, Any]],
        body: Any,
    ) -> Tuple[int, Any]:
        attempt = 1
        current_body = body

        def get_cmd(b: Any) -> str:
            if isinstance(b, dict):
                return str(b.get("cmd") or "")
            return ""

        while True:
            sc, res = invoke_step(mcp_id, method, path, query, current_body)

            if not (200 <= int(sc) < 300):
                log.info(f"event=cmd.http_fail run_id={run_id} status_code={sc} res={_short(res)}")
                return sc, res

            failed, reason = _command_failed(res)

            # Log extra para entender bloqueos
            if isinstance(res, dict):
                out_len = len(str(res.get("stdout") or ""))
                err_len = len(str(res.get("stderr") or ""))
                log.info(
                    f"event=cmd.result run_id={run_id} failed={failed} stdout_len={out_len} stderr_len={err_len} "
                    f"status={res.get('status')} exit_code={res.get('exit_code')}"
                )

            if not failed:
                return sc, res

            if attempt >= MAX_ATTEMPTS_PER_COMMAND_STEP:
                return sc, res

            cmd = get_cmd(current_body)

            fixed = _fix_windows_del_command(cmd, reason)
            if fixed and fixed.strip() != cmd.strip() and not _looks_dangerous(fixed):
                attempt += 1
                emit(
                    "step_retry",
                    step_path,
                    title,
                    f"El comando falló. Intento {attempt}/{MAX_ATTEMPTS_PER_COMMAND_STEP}\nAntes: {cmd}\nAhora:  {fixed}",
                )
                current_body = dict(current_body) if isinstance(current_body, dict) else {}
                current_body["cmd"] = fixed
                continue

            repaired = _repair_command_with_gpt(
                client=client,
                goal=plan.goal,
                cmd=cmd,
                error_text=reason,
                attempt=attempt + 1,
                max_attempts=MAX_ATTEMPTS_PER_COMMAND_STEP,
            )
            if not repaired:
                return sc, res

            new_cmd, why = repaired
            attempt += 1

            detail = f"Intento {attempt}/{MAX_ATTEMPTS_PER_COMMAND_STEP}\nAntes: {cmd}\nAhora:  {new_cmd}"
            if why:
                detail += f"\nMotivo: {why}"

            emit("step_retry", step_path, title, detail)

            current_body = dict(current_body) if isinstance(current_body, dict) else {}
            current_body["cmd"] = new_cmd

    def invoke_mcp_with_emit(mcp_id: str, method: str, path: str, query: Optional[Dict[str, Any]], body: Any):
        step_path = path or "step"
        title = f"{method} {path}"

        emit("step_start", step_path, title)

        if _is_command_call(method, path):
            sc, res = _invoke_command_with_auto_retry(step_path, title, mcp_id, method, path, query, body)

            if 200 <= int(sc) < 300:
                failed, reason = _command_failed(res)
                if not failed:
                    emit("step_ok", step_path, title, _short(res))
                else:
                    emit("step_err", step_path, title, reason)
            else:
                emit("step_err", step_path, title, _short(res))

            return sc, res

        sc, res = invoke_step(mcp_id, method, path, query, body)

        if 200 <= int(sc) < 300:
            emit("step_ok", step_path, title, _short(res))
        else:
            emit("step_err", step_path, title, _short(res))

        return sc, res

    # -----------------------------
    # Ejecutar plan
    # -----------------------------
    try:
        run_store.update(run_id, last_event="execute_plan_run_start")

        final_plan: "PlanRun" = await asyncio.wait_for(
            asyncio.to_thread(
                execute_plan_run,
                plan=plan,
                trace_id=trace_id,
                invoke_mcp_call=invoke_mcp_with_emit,
                validate_step=validate_step_with_emit,
            ),
            timeout=PLAN_TIMEOUT_S,
        )

        final_status = "error" if getattr(final_plan, "status", "") == "error" else "done"

        run_store.update(
            run_id,
            status=final_status,
            plan=asdict(final_plan),
            last_event="run_done",
        )

        safe_add_message(
            "assistant",
            f"Plan '{final_plan.goal}' finalizado con estado: {final_plan.status}. "
            f"Último paso: {final_plan.current_step_path or '-'}",
        )

        log.info(f"event=runner.finish run_id={run_id} status={final_status}")

    except asyncio.TimeoutError:
        run_store.update(run_id, status="error", error="TimeoutError: plan_timeout", last_event="run_timeout")
        safe_add_message("assistant", f"⏱️ El plan excedió el tiempo máximo y fue detenido.\n(run_id={run_id})")
        log.info(f"event=runner.timeout run_id={run_id}")

    except asyncio.CancelledError:
        run_store.update(run_id, status="error", error="CancelledError: runner_cancelled", last_event="runner_cancelled")
        safe_add_message("assistant", f"⚠️ Ejecución cancelada (probable recarga del servidor).\n(run_id={run_id})")
        log.info(f"event=runner.cancelled run_id={run_id}")
        raise

    except Exception as e:
        run_store.update(run_id, status="error", error=f"{type(e).__name__}: {e}", last_event="run_error")
        safe_add_message("assistant", f"❌ Error ejecutando plan\n{type(e).__name__}: {e}")
        log.info(f"event=runner.exception run_id={run_id} err={type(e).__name__} msg={e}")

    finally:
        # CRÍTICO: limpiar estado del chat para que la 3era petición no quede “pegada”
        try:
            _safe_call(store.set_state, chat_id, active_run_id=None)
            log.info(f"event=chat.state.clear_active run_id={run_id} chat_id={chat_id}")
        except Exception as e:
            log.info(f"event=chat.state.clear_active.error run_id={run_id} err={type(e).__name__} msg={e}")

        safe_chat_preview_title()

        r = run_store.get(run_id)
        log.info(f"event=plan.bg.done run_id={run_id} status={r.status if r else 'missing'}")
