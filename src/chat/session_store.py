from typing import Dict, List
from src.chat.prompts import DEFAULT_SYSTEM_PROMPT

# Demo: memoria en RAM (se pierde al reiniciar).
# Producción: usar Redis/DB.
_SESSIONS: Dict[str, List[dict]] = {}

def get_or_create_session(session_id: str) -> List[dict]:
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
    return _SESSIONS[session_id]

def reset_session(session_id: str) -> None:
    _SESSIONS[session_id] = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
