from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re
import time

import httpx

from src.observability.logger import get_logger
from src.mcp.store import MCPEndpoint


@dataclass
class OpenAPIDiscoveryResult:
    """
    Resultado del discovery de OpenAPI:
    - openapi_url: URL exacta que funcionó
    - spec: JSON completo OpenAPI (dict)
    - endpoints: endpoints normalizados para UI/selección
    """
    openapi_url: str
    spec: Dict[str, Any]
    endpoints: List[MCPEndpoint]


def _normalize_base_url(address_or_url: str) -> str:
    """
    Normaliza entradas como:
      - "192.168.1.50:9090"
      - "http://192.168.1.50:9090"
      - "https://host"
      - "http://host/docs" (si alguien pega /docs)
    a una base_url con esquema, sin trailing slash, y sin /docs.
    """
    s = (address_or_url or "").strip()
    if not s:
        raise ValueError("address/base_url vacío")

    # Si no tiene esquema, asumimos http://
    if not re.match(r"^https?://", s, flags=re.IGNORECASE):
        s = "http://" + s

    # Si nos pasan /docs, /redoc, etc. lo recortamos al host.
    s = re.sub(r"/(docs|redoc)(/.*)?$", "", s, flags=re.IGNORECASE)

    return s.rstrip("/")


def _candidate_openapi_paths() -> List[str]:
    """
    Rutas comunes de OpenAPI/Swagger JSON (server-side).

    Nota: /docs suele ser HTML; preferimos siempre el JSON.
    """
    return [
        "/openapi.json",  # FastAPI default
        "/api/openapi.json",
        "/swagger.json",
        "/v1/openapi.json",
        "/openapi",
        "/api-docs",
    ]


def _extract_endpoints_from_openapi(spec: Dict[str, Any]) -> List[MCPEndpoint]:
    """
    Extrae endpoints desde spec OpenAPI:
      spec["paths"][path][method] -> operation

    Devolvemos una lista “plana” para UI y selección.
    """
    endpoints: List[MCPEndpoint] = []
    paths = spec.get("paths") or {}
    if not isinstance(paths, dict):
        return endpoints

    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue

        for method, operation in methods.items():
            m = str(method).upper()
            if m not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
                continue
            if not isinstance(operation, dict):
                continue

            endpoints.append(
                MCPEndpoint(
                    path=str(path),
                    method=m,
                    operation_id=operation.get("operationId"),
                    summary=operation.get("summary") or operation.get("description"),
                    tags=list(operation.get("tags") or []),
                )
            )

    endpoints.sort(key=lambda e: (e.path, e.method))
    return endpoints


def _looks_like_openapi(data: Any) -> bool:
    """
    Heurística mínima para aceptar un JSON como OpenAPI.
    (Admite OpenAPI 3.x y Swagger 2.0 mientras exista paths no vacío.)
    """
    if not isinstance(data, dict):
        return False
    paths = data.get("paths")
    if not isinstance(paths, dict) or len(paths) == 0:
        return False
    # opcional: pista de versión
    if "openapi" in data and isinstance(data.get("openapi"), str):
        return True
    if "swagger" in data and isinstance(data.get("swagger"), str):
        return True
    # si tiene paths pero no versión, igual lo aceptamos
    return True


async def discover_openapi(
    address_or_url: str,
    *,
    timeout_s: float = 6.0,
    trace_id: str = "-",
) -> OpenAPIDiscoveryResult:
    """
    Descubre un OpenAPI JSON a partir de una IP:PUERTO o base_url.

    Estrategia:
    - Normaliza base_url
    - Prueba candidatos típicos (openapi.json, swagger.json, etc.)
    - Retorna spec + endpoints extraídos

    Logging (auditoría):
    - start/try/success/fail + latencias
    - detalles de errores por URL (status, timeout, parse JSON, etc.)

    Errores:
    - Distingue timeouts, connect error, HTTP error
    - Devuelve ValueError con mensaje final entendible
    """
    log = get_logger(trace_id)
    t_total = time.time()

    base_url = _normalize_base_url(address_or_url)
    log.info(f"event=openapi.discover.start base_url={base_url} timeout_s={timeout_s}")

    # Timeouts separados: connect y read.
    timeout = httpx.Timeout(timeout=timeout_s, connect=timeout_s, read=timeout_s)

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        last_error: Optional[str] = None

        for p in _candidate_openapi_paths():
            url = f"{base_url}{p}"
            t_try = time.time()
            log.info(f"event=openapi.discover.try url={url}")

            try:
                r = await client.get(url, headers={"Accept": "application/json"})
                try_ms = int((time.time() - t_try) * 1000)

                # 404/401/403/etc: no abortamos, probamos la siguiente ruta.
                if r.status_code >= 400:
                    last_error = f"{url} -> HTTP {r.status_code}"
                    log.info(
                        f"event=openapi.discover.http_error url={url} status={r.status_code} duration_ms={try_ms}"
                    )
                    continue

                # Algunos servidores devuelven OpenAPI como text/plain.
                # httpx igual permite .json(); si falla, lo capturamos.
                try:
                    data = r.json()
                except Exception:
                    last_error = f"{url} -> respuesta no era JSON válido"
                    log.info(
                        f"event=openapi.discover.json_error url={url} duration_ms={try_ms}"
                    )
                    continue

                if not _looks_like_openapi(data):
                    last_error = f"{url} -> no parece OpenAPI (sin 'paths')"
                    log.info(
                        f"event=openapi.discover.not_openapi url={url} duration_ms={try_ms}"
                    )
                    continue

                endpoints = _extract_endpoints_from_openapi(data)

                total_ms = int((time.time() - t_total) * 1000)
                log.info(
                    f"event=openapi.discover.success openapi_url={url} endpoints={len(endpoints)} "
                    f"try_duration_ms={try_ms} total_duration_ms={total_ms}"
                )

                # (opcional) resumen corto de endpoints para debug (sin volcar el spec completo)
                # muestra solo los primeros 5, y si hay más, lo indica
                try:
                    sample = [f"{e.method} {e.path}" for e in endpoints[:5]]
                    more = len(endpoints) - len(sample)
                    sample_str = ";".join(sample) + (f";+{more}_more" if more > 0 else "")
                    log.info(f"event=openapi.discover.sample endpoints_sample={sample_str}")
                except Exception:
                    log.info("event=openapi.discover.sample_error")

                return OpenAPIDiscoveryResult(openapi_url=url, spec=data, endpoints=endpoints)

            except httpx.ConnectTimeout:
                try_ms = int((time.time() - t_try) * 1000)
                last_error = f"{url} -> ConnectTimeout (host/puerto no responde)"
                log.info(
                    f"event=openapi.discover.connect_timeout url={url} duration_ms={try_ms}"
                )

            except httpx.ReadTimeout:
                try_ms = int((time.time() - t_try) * 1000)
                last_error = f"{url} -> ReadTimeout (respondió lento)"
                log.info(
                    f"event=openapi.discover.read_timeout url={url} duration_ms={try_ms}"
                )

            except httpx.ConnectError as e:
                try_ms = int((time.time() - t_try) * 1000)
                last_error = f"{url} -> ConnectError ({e})"
                log.info(
                    f"event=openapi.discover.connect_error url={url} duration_ms={try_ms} err={type(e).__name__}"
                )

            except httpx.HTTPError as e:
                try_ms = int((time.time() - t_try) * 1000)
                last_error = f"{url} -> HTTPError ({e})"
                log.info(
                    f"event=openapi.discover.httpx_error url={url} duration_ms={try_ms} err={type(e).__name__}"
                )

            except Exception as e:
                try_ms = int((time.time() - t_try) * 1000)
                last_error = f"{url} -> {type(e).__name__}: {e}"
                log.info(
                    f"event=openapi.discover.unknown_error url={url} duration_ms={try_ms} err={type(e).__name__}"
                )

        total_ms = int((time.time() - t_total) * 1000)
        log.info(
            f"event=openapi.discover.fail base_url={base_url} total_duration_ms={total_ms} last_error={last_error}"
        )
        raise ValueError(
            "No se pudo descubrir OpenAPI JSON en el host. "
            "Revisa conectividad (IP/puerto/firewall) y que el servicio exponga OpenAPI. "
            f"Último error: {last_error}"
        )
