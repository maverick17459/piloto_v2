from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
from src.agent.plan_models import PlanRun, PlanStep
from src.observability.logger import get_logger

# Estas dependencias ya existen en tu /api/send actual:
# - invoke_mcp_sync
# - MCPCall
# - mcp_store / store / proj checks (se hacen afuera o aquí)
# Aquí lo dejamos parametrizable.

def _flatten_steps(step: PlanStep, prefix: str = ""):
    """
    Genera (path, step) en orden DFS: 1, 1.1, 1.2, 2, 2.1 ...
    """
    if not prefix:
        path = ""
    else:
        path = prefix

    yield (path, step)

    if step.type == "subplan" and step.substeps:
        for i, s in enumerate(step.substeps, start=1):
            subpath = f"{path}.{i}" if path else str(i)
            yield from _flatten_steps(s, subpath)

def execute_plan_run(
    *,
    plan: PlanRun,
    trace_id: str,
    invoke_mcp_call,   # callable(mcp_id, method, path, query, body) -> (status_code, result)
    validate_step,     # callable(step) -> Optional[str]  (error string si inválido)
) -> PlanRun:
    """
    Ejecuta el plan de forma determinista.
    - NO vuelve a preguntarle al LLM para cada paso (evita “olvidos”).
    - Valida cada paso antes de ejecutar.
    """
    log = get_logger(trace_id)

    plan.status = "running"
    log.info(f"event=plan.start plan_id={plan.id} steps={len(plan.steps)}")

    try:
        # recorremos steps top-level en orden
        for i, top in enumerate(plan.steps, start=1):
            path = str(i)

            # si es subplan, igual lo marcamos y bajamos
            for subpath, step in _flatten_steps(top, path):
                plan.current_step_path = subpath or path

                if step.status in ("done", "skipped"):
                    continue

                err = validate_step(step)
                if err:
                    step.status = "error"
                    step.error = err
                    log.info(f"event=plan.step.error plan_id={plan.id} step={plan.current_step_path} err={err}")
                    plan.status = "error"
                    plan.ended_ts = None
                    return plan

                step.status = "running"
                step.started_ts = step.started_ts or __import__("time").time_ns() // 1_000_000
                log.info(f"event=plan.step.start plan_id={plan.id} step={plan.current_step_path} type={step.type} title={step.title}")

                if step.type == "note":
                    step.status = "done"
                    step.ended_ts = __import__("time").time_ns() // 1_000_000
                    step.result_summary = step.title or "OK"
                    log.info(f"event=plan.step.done plan_id={plan.id} step={plan.current_step_path} type=note")
                    continue

                if step.type == "mcp_call":
                    status_code, result = invoke_mcp_call(step.mcp_id, step.method, step.path, step.query, step.body)
                    step.result_raw = result
                    step.result_summary = f"status_code={status_code}"
                    step.status = "done" if 200 <= int(status_code) < 300 else "error"
                    step.ended_ts = __import__("time").time_ns() // 1_000_000

                    log.info(
                        f"event=plan.step.done plan_id={plan.id} step={plan.current_step_path} "
                        f"mcp_id={step.mcp_id} method={step.method} path={step.path} status_code={status_code} final_status={step.status}"
                    )

                    if step.status == "error":
                        plan.status = "error"
                        plan.ended_ts = __import__("time").time_ns() // 1_000_000
                        return plan

        plan.status = "done"
        plan.ended_ts = __import__("time").time_ns() // 1_000_000
        log.info(f"event=plan.done plan_id={plan.id} status=done")
        return plan

    except Exception as e:
        plan.status = "error"
        plan.ended_ts = __import__("time").time_ns() // 1_000_000
        log.info(f"event=plan.fatal plan_id={plan.id} err={type(e).__name__}")
        return plan
