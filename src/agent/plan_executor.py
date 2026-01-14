from __future__ import annotations

from typing import Any, Dict, Optional
from src.agent.plan_models import PlanRun, PlanStep
from src.observability.logger import get_logger
import time


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _flatten_steps(step: PlanStep, prefix: str = ""):
    """
    Genera (path, step) en orden DFS: 1, 1.1, 1.2, 2, 2.1 ...
    """
    path = prefix or ""
    yield (path, step)

    if step.type == "subplan" and step.substeps:
        for i, s in enumerate(step.substeps, start=1):
            subpath = f"{path}.{i}" if path else str(i)
            yield from _flatten_steps(s, subpath)


def _is_command_call(method: Optional[str], path: Optional[str]) -> bool:
    return (method or "").upper() == "POST" and (path or "") == "/command"


def _command_failed(result: Any) -> tuple[bool, str]:
    """
    Devuelve (failed, reason_text) basado en el payload del MCP /command.
    """
    if not isinstance(result, dict):
        return True, "Resultado /command inválido (no es JSON objeto)."

    status = str(result.get("status") or "").lower().strip()
    exit_code = result.get("exit_code")

    # stdout/stderr: en tu MCP a veces stdout trae el error
    stdout = result.get("stdout")
    stderr = result.get("stderr")

    # normalizamos salida
    msg = ""
    if isinstance(stderr, str) and stderr.strip():
        msg = stderr.strip()
    elif isinstance(stdout, str) and stdout.strip():
        msg = stdout.strip()

    # exit_code puede venir como str/int
    try:
        ec = int(exit_code) if exit_code is not None else 0
    except Exception:
        ec = 0

    if status == "ok" and ec == 0:
        return False, msg or "OK"

    reason = msg or f"/command falló (status={status or 'unknown'}, exit_code={exit_code})"
    return True, reason


def execute_plan_run(
    *,
    plan: PlanRun,
    trace_id: str,
    invoke_mcp_call,  # callable(mcp_id, method, path, query, body) -> (status_code, result)
    validate_step,    # callable(step) -> Optional[str]  (error string si inválido)
) -> PlanRun:
    """
    Ejecuta el plan de forma determinista.
    - Valida cada paso antes de ejecutar.
    - Para /command: NO confundir HTTP 200 con éxito. Se valida exit_code/status del payload.
    """
    log = get_logger(trace_id)

    plan.status = "running"
    log.info(f"event=plan.start plan_id={plan.id} steps={len(plan.steps)}")

    try:
        for i, top in enumerate(plan.steps, start=1):
            path = str(i)

            for subpath, step in _flatten_steps(top, path):
                plan.current_step_path = subpath or path

                if step.status in ("done", "skipped"):
                    continue

                err = validate_step(step)
                if err:
                    step.status = "error"
                    step.error = err
                    step.ended_ts = _now_ms()
                    log.info(f"event=plan.step.error plan_id={plan.id} step={plan.current_step_path} err={err}")
                    plan.status = "error"
                    plan.ended_ts = _now_ms()   # FIX: nunca None
                    return plan

                step.status = "running"
                step.started_ts = step.started_ts or _now_ms()
                log.info(
                    f"event=plan.step.start plan_id={plan.id} step={plan.current_step_path} "
                    f"type={step.type} title={step.title}"
                )

                if step.type == "note":
                    step.status = "done"
                    step.ended_ts = _now_ms()
                    step.result_summary = step.title or "OK"
                    log.info(f"event=plan.step.done plan_id={plan.id} step={plan.current_step_path} type=note")
                    continue

                if step.type == "mcp_call":
                    status_code, result = invoke_mcp_call(
                        step.mcp_id, step.method, step.path, step.query, step.body
                    )
                    step.result_raw = result

                    # FIX: éxito real para /command
                    if _is_command_call(step.method, step.path):
                        failed, reason = _command_failed(result)
                        step.result_summary = reason
                        step.status = "error" if failed else "done"
                    else:
                        step.result_summary = f"status_code={status_code}"
                        step.status = "done" if 200 <= int(status_code) < 300 else "error"

                    step.ended_ts = _now_ms()

                    log.info(
                        f"event=plan.step.done plan_id={plan.id} step={plan.current_step_path} "
                        f"mcp_id={step.mcp_id} method={step.method} path={step.path} "
                        f"status_code={status_code} final_status={step.status}"
                    )

                    if step.status == "error":
                        # guardar razón humana en step.error si aún no está
                        if not getattr(step, "error", None):
                            step.error = step.result_summary or "Step failed"
                        plan.status = "error"
                        plan.ended_ts = _now_ms()
                        return plan

        plan.status = "done"
        plan.ended_ts = _now_ms()
        log.info(f"event=plan.done plan_id={plan.id} status=done")
        return plan

    except Exception as e:
        plan.status = "error"
        plan.ended_ts = _now_ms()
        log.info(f"event=plan.fatal plan_id={plan.id} err={type(e).__name__}")
        return plan
