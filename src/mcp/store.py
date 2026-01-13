from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time
import uuid




def _id() -> str:
    return uuid.uuid4().hex


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class MCPEndpoint:
    """
    Endpoint normalizado extraído desde un OpenAPI (paths/methods).

    Guardamos lo mínimo para:
    - listar en UI
    - elegir qué operación usar
    - debuggear
    """
    path: str
    method: str  # GET/POST/PUT/PATCH/DELETE...
    operation_id: Optional[str] = None
    summary: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class MCP:
    """
    Representa una Consola/MCP registrada en nuestra aplicación.
    """
    id: str
    name: str
    base_url: str              # ej: http://192.168.1.50:9090
    docs_url: Optional[str]    # ej: http://.../docs (informativo)
    openapi_url: Optional[str] # ej: http://.../openapi.json (real)
    is_active: bool
    endpoints: List[MCPEndpoint] = field(default_factory=list)

    created_ts: int = field(default_factory=_now_ms)
    updated_ts: int = field(default_factory=_now_ms)

    # Opcional: guardar el spec completo (puede ser pesado). Aquí lo dejamos opcional.
    openapi_raw: Optional[Dict[str, Any]] = None


class MCPStore:
    """
    Store en memoria (RAM) para MCPs.
    Sigue el espíritu de MemoryStore del proyecto: simple, rápido, sin persistencia aún.
    """

    def __init__(self) -> None:
        self._mcps: Dict[str, MCP] = {}

    # -------- CRUD --------

    def list_mcps(self) -> List[MCP]:
        return list(self._mcps.values())

    def get_mcp(self, mcp_id: str) -> Optional[MCP]:
        return self._mcps.get(mcp_id)

    def create_mcp(
        self,
        *,
        base_url: str,
        name: Optional[str] = None,
        docs_url: Optional[str] = None,
    ) -> MCP:
        mcp_id = _id()
        mcp = MCP(
            id=mcp_id,
            name=(name.strip() if name and name.strip() else base_url),
            base_url=base_url,
            docs_url=docs_url,
            openapi_url=None,
            is_active=True,
            endpoints=[],
        )
        self._mcps[mcp_id] = mcp
        return mcp

    def update_mcp(
        self,
        mcp_id: str,
        *,
        name: Optional[str] = None,
        base_url: Optional[str] = None,
        docs_url: Optional[str] = None,
    ) -> bool:
        mcp = self._mcps.get(mcp_id)
        if not mcp:
            return False

        if name is not None:
            name = name.strip()
            if name:
                mcp.name = name

        if base_url is not None:
            base_url = base_url.strip()
            if base_url:
                mcp.base_url = base_url

        if docs_url is not None:
            mcp.docs_url = docs_url.strip() or None

        mcp.updated_ts = _now_ms()
        return True

    def delete_mcp(self, mcp_id: str) -> bool:
        if mcp_id not in self._mcps:
            return False
        del self._mcps[mcp_id]
        return True

    # -------- Estado --------

    def set_active(self, mcp_id: str, active: bool) -> bool:
        mcp = self._mcps.get(mcp_id)
        if not mcp:
            return False
        mcp.is_active = bool(active)
        mcp.updated_ts = _now_ms()
        return True

    # -------- Discovery results --------

    def save_discovery(
        self,
        mcp_id: str,
        *,
        openapi_url: str,
        endpoints: List[MCPEndpoint],
        openapi_raw: Optional[Dict[str, Any]] = None,
    ) -> bool:
        mcp = self._mcps.get(mcp_id)
        if not mcp:
            return False
        mcp.openapi_url = openapi_url
        mcp.endpoints = endpoints
        mcp.openapi_raw = openapi_raw
        mcp.updated_ts = _now_ms()
        return True

    # -------- Util --------

    def find_by_base_url(self, base_url: str) -> Optional[MCP]:
        """
        Útil para evitar duplicados: si ya existe un MCP con ese base_url,
        podemos decidir si lo reutilizamos o lo actualizamos.
        """
        for m in self._mcps.values():
            if m.base_url == base_url:
                return m
        return None
