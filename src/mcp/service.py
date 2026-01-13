from __future__ import annotations

from typing import Optional

from src.mcp.discovery import discover_openapi, _normalize_base_url
from src.mcp.store import MCPStore, MCP


class MCPService:
    """
    Capa de negocio para gestionar MCPs:
    - crear MCP desde ip:puerto
    - descubrir OpenAPI
    - refrescar endpoints
    - activar/desactivar
    """

    def __init__(self, store: MCPStore) -> None:
        self.store = store


    async def register(
        self,
        *,
        address: str,
        name: Optional[str] = None,
        docs_url: Optional[str] = None,
        save_openapi_raw: bool = False,
        allow_offline: bool = True,   # NUEVO
    ) -> MCP:
        base_url = _normalize_base_url(address)

        existing = self.store.find_by_base_url(base_url)
        if existing:
            # refresh best-effort
            try:
                await self.refresh(existing.id, save_openapi_raw=save_openapi_raw)
            except Exception:
                pass
            self.store.update_mcp(existing.id, name=name, docs_url=docs_url)
            return self.store.get_mcp(existing.id) or existing

        mcp = self.store.create_mcp(base_url=base_url, name=name, docs_url=docs_url)

        try:
            await self.refresh(mcp.id, save_openapi_raw=save_openapi_raw)
        except Exception:
            if not allow_offline:
                # si no se permite offline, revertimos
                self.store.delete_mcp(mcp.id)
                raise

        return self.store.get_mcp(mcp.id) or mcp


    async def refresh(self, mcp_id: str, *, save_openapi_raw: bool = False) -> MCP:
        """
        Relee OpenAPI y actualiza endpoints guardados.
        """
        mcp = self.store.get_mcp(mcp_id)
        if not mcp:
            raise ValueError("MCP no encontrado")

        result = await discover_openapi(mcp.base_url)
        self.store.save_discovery(
            mcp_id,
            openapi_url=result.openapi_url,
            endpoints=result.endpoints,
            openapi_raw=(result.spec if save_openapi_raw else None),
        )

        updated = self.store.get_mcp(mcp_id)
        if not updated:
            raise ValueError("MCP no encontrado tras refresh (estado inconsistente)")
        return updated

    def set_active(self, mcp_id: str, active: bool) -> MCP:
        ok = self.store.set_active(mcp_id, active)
        if not ok:
            raise ValueError("MCP no encontrado")
        m = self.store.get_mcp(mcp_id)
        if not m:
            raise ValueError("MCP no encontrado tras set_active (estado inconsistente)")
        return m

    def delete(self, mcp_id: str) -> None:
        ok = self.store.delete_mcp(mcp_id)
        if not ok:
            raise ValueError("MCP no encontrado")
