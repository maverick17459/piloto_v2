from __future__ import annotations

from typing import Optional, Any
from src.agent.plan_models import PlanStep


def reason_about_command_failure(
    *,
    client,
    goal: str,
    step: PlanStep,
    stdout: str,
    stderr: str,
    attempt: int,
    max_attempts: int,
) -> Optional[PlanStep]:
    """
    Decide si se puede reintentar un comando fallido.
    Devuelve un NUEVO PlanStep o None.
    """

    system = (
        "Eres un agente experto en ejecución de comandos.\n"
        "Analiza el fallo y decide si puede corregirse.\n\n"
        "REGLAS:\n"
        "- Si puedes corregirlo, propone UN nuevo comando.\n"
        "- Mantén la intención original.\n"
        "- NO repitas el mismo comando.\n"
        "- NO propongas comandos destructivos.\n"
        "- Si no es posible, responde con give_up y explica en 'why'.\n"
    )

    # step.body normalmente es {"cmd": "..."}; lo normalizamos para mostrarlo bien
    prev_cmd = ""
    if isinstance(step.body, dict):
        prev_cmd = str(step.body.get("cmd") or "")
    else:
        prev_cmd = str(step.body or "")

    user = (
        f"OBJETIVO:\n{goal}\n\n"
        f"INTENTO:\n{attempt}/{max_attempts}\n\n"
        f"COMANDO ANTERIOR:\n{prev_cmd}\n\n"
        f"STDOUT:\n{stdout or '(vacío)'}\n\n"
        f"STDERR:\n{stderr or '(vacío)'}\n"
    )

    #  OJO: el wrapper espera messages como primer arg POSICIONAL (no messages=)
    msg = client.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "propose_fix",
                    "description": "Propose a corrected command or give up",
                    "parameters": {
                        "type": "object",
                        "required": ["action"],
                        "properties": {
                            "action": {"type": "string", "enum": ["retry", "give_up"]},
                            "cmd": {"type": "string"},
                            "why": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                },
            }
        ],
        tool_choice="auto",
        temperature=0,
        mode="message",
    )

    tool_calls = msg.get("tool_calls")

    # Si no hay tool_call, el modelo decidió explicar en texto (o no supo)
    if not tool_calls:
        return None

    call0 = tool_calls[0]
    args: Any = (call0.get("function") or {}).get("arguments")

    # Tu api_client.py ya intenta parsear arguments; igual dejamos defensa extra
    if not isinstance(args, dict):
        return None

    if args.get("action") != "retry":
        return None

    new_cmd = (args.get("cmd") or "").strip()
    if not new_cmd:
        return None

    # Regla: no repetir mismo comando
    if new_cmd.strip() == prev_cmd.strip():
        return None

    return PlanStep(
        title=step.title,
        type="mcp_call",
        mcp_id=step.mcp_id,
        method=step.method,
        path=step.path,
        query=step.query,
        body={"cmd": new_cmd},
    )
