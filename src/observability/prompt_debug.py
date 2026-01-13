from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List

LOG_PROMPT_MAX_CHARS = int(os.getenv("LOG_PROMPT_MAX_CHARS", "12000"))
# Si está en 1, loguea solo resumen (sin contenido completo) incluso en piloto_prompts.log
LOG_PROMPTS_SAFE = os.getenv("LOG_PROMPTS_SAFE", "0") == "1"

REDACT_PATTERNS = [
    # Headers comunes
    (re.compile(r"(?im)^(x-api-key\s*:\s*)(.+)$"), r"\1[REDACTED]"),
    (re.compile(r"(?im)^(authorization\s*:\s*bearer\s+)(\S+)$"), r"\1[REDACTED]"),
    (re.compile(r"(?im)^(cookie\s*:\s*)(.+)$"), r"\1[REDACTED]"),
    (re.compile(r"(?im)^(set-cookie\s*:\s*)(.+)$"), r"\1[REDACTED]"),

    # OpenAI-like keys (ej sk-...)
    (re.compile(r"(?i)\b(sk|rk)-[A-Za-z0-9]{10,}\b"), "[REDACTED]"),

    # Env style: KEY=VALUE
    (re.compile(r"(?i)\b(openai_api_key\s*=\s*)(\S+)"), r"\1[REDACTED]"),
    (re.compile(r"(?i)\b(aws_access_key_id\s*=\s*)(\S+)"), r"\1[REDACTED]"),
    (re.compile(r"(?i)\b(aws_secret_access_key\s*=\s*)(\S+)"), r"\1[REDACTED]"),
    (re.compile(r"(?i)\b(aws_session_token\s*=\s*)(\S+)"), r"\1[REDACTED]"),

    # JSON keys genéricas (apiKey, token, secret, password...)
    (re.compile(r'(?i)("?(api[_-]?key|apikey|access[_-]?token|refresh[_-]?token|token|secret|client[_-]?secret|password|passwd)"?\s*:\s*")([^"]+)(")'),
     r'\1[REDACTED]\4'),

    # Query-string style
    (re.compile(r"(?i)\b(token|access_token|refresh_token|api_key|apikey|secret|password)\s*=\s*([^\s&]+)"),
     r"\1=[REDACTED]"),
]

def _redact_text(s: str) -> str:
    if not s:
        return s
    out = s
    for rx, repl in REDACT_PATTERNS:
        out = rx.sub(repl, out)
    return out

def summarize_messages(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Resumen seguro:
    - roles
    - chars por mensaje
    - hash por mensaje (sobre contenido REDACTED para evitar huellas de secretos)
    """
    roles: List[str] = []
    lens: List[int] = []
    hashes: List[str] = []

    for m in messages:
        role = str(m.get("role", "?"))
        content = _redact_text(str(m.get("content", "") or ""))
        roles.append(role)
        lens.append(len(content))
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        hashes.append(h)

    return {
        "count": len(messages),
        "roles": roles,
        "chars": lens,
        "hash12": hashes,
        "total_chars": sum(lens),
    }

def serialize_messages_for_promptlog(messages: List[Dict[str, Any]]) -> str:
    """
    Contenido completo (sanitizado) para prompt log.
    Si LOG_PROMPTS_SAFE=1, retorna un resumen (sin contenido completo).
    """
    if LOG_PROMPTS_SAFE:
        summ = summarize_messages(messages)
        return json.dumps({"safe_mode": True, "summary": summ}, ensure_ascii=False, indent=2)

    safe_msgs = []
    for m in messages:
        role = str(m.get("role", "?"))
        content = _redact_text(str(m.get("content", "") or ""))

        # recorte por mensaje para evitar que un solo blob mate todo
        if len(content) > LOG_PROMPT_MAX_CHARS:
            content = content[:LOG_PROMPT_MAX_CHARS] + "\n...[TRUNCATED]..."

        safe_msgs.append({"role": role, "content": content})

    raw = json.dumps(safe_msgs, ensure_ascii=False, indent=2)

    # recorte total por si el array es gigantesco
    if len(raw) > LOG_PROMPT_MAX_CHARS:
        raw = raw[:LOG_PROMPT_MAX_CHARS] + "\n...[TRUNCATED]..."

    return raw

def serialize_text_for_promptlog(label: str, text: str) -> str:
    """
    Serializa un texto suelto (system prompt, etc.) sanitizado.
    Si LOG_PROMPTS_SAFE=1, solo loguea longitud y hash.
    """
    t = _redact_text(text or "")

    if LOG_PROMPTS_SAFE:
        h = hashlib.sha256(t.encode("utf-8")).hexdigest()[:12]
        return f"{label}:\n[safe_mode] chars={len(t)} hash12={h}"

    if len(t) > LOG_PROMPT_MAX_CHARS:
        t = t[:LOG_PROMPT_MAX_CHARS] + "\n...[TRUNCATED]..."
    return f"{label}:\n{t}"
