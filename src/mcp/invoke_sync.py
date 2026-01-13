# src/mcp/invoke_sync.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import httpx


@dataclass(frozen=True)
class MCPCall:
    mcp_id: str
    method: str
    path: str
    query: Optional[Dict[str, Any]] = None
    body: Optional[Any] = None


class MCPInvokeError(RuntimeError):
    pass


def _endpoint_allowed(mcp, method: str, path: str) -> bool:
    method = (method or "").upper().strip()
    path = (path or "").strip()

    for e in (mcp.endpoints or []):
        if e.method.upper() == method and e.path == path:
            return True
    return False


def invoke_mcp_sync(
    *,
    mcp,
    call: MCPCall,
    timeout_s: float = 15.0,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, Any]:
    """
    Ejecuta una llamada HTTP real a un MCP (sync).
    - No hardcodea endpoints.
    - Permite solo endpoints existentes en mcp.endpoints (descubiertos OpenAPI).
    - Retorna (status_code, response_json|text).
    """
    if not _endpoint_allowed(mcp, call.method, call.path):
        raise MCPInvokeError(f"Endpoint no permitido: {call.method} {call.path}")

    base = (mcp.base_url or "").rstrip("/")
    url = f"{base}{call.path}"

    headers = {"Accept": "application/json"}
    if extra_headers:
        headers.update(extra_headers)

    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        r = client.request(
            call.method.upper(),
            url,
            params=call.query or None,
            json=call.body,
            headers=headers,
        )

    try:
        data = r.json()
    except Exception:
        data = r.text

    return r.status_code, data
