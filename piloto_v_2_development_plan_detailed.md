# Piloto v2 – Plan de Desarrollo Paso a Paso (Estimaciones Realistas)


## Fase 0 – Preparación y diseño (8–12 h)

### Entregables
- Definición del flujo: chat vs tool-call.
- Decisión de confirmación explícita.
- Modelo mental de estados: `pending_run_id` / `active_run_id`.
- Boceto de endpoints API.

### Tareas típicas
- Documento breve de arquitectura.
- Primer prompt system base.

⏱ 1–1.5 días

---

## Fase 1 – Base FastAPI + UI mínima (12–18 h)

### Entregables
- FastAPI corriendo con CORS.
- Página `index.html` (chat simple).
- `GET /` sirve la UI.

### Tareas
- Setup de proyecto.
- Middleware de logs/trace.
- UI: caja de texto + lista de mensajes.

⏱ 1.5–2.5 días

---

## Fase 2 – Modelo de dominio: Projects/Chats/Messages (16–24 h)

### Entregables
- `MemoryStore` con:
  - Projects
  - Chats
  - Messages
- Endpoints básicos:
  - crear proyecto
  - crear chat
  - listar chats
  - traer mensajes

### Riesgos
- Diseño del store (evitar acoplarse demasiado al UI).

⏱ 2–3 días

---

## Fase 3 – Integración LLM + envío `/api/send` (16–22 h)

### Entregables
- `OpenAIChatClient` wrapper.
- `/api/send` que:
  - arma payload con system + historial
  - guarda mensajes user/assistant

### Tareas
- Manejar errores del modelo.
- Logging de prompts (opcional).

⏱ 2–3 días

---

## Fase 4 – Tool calling: `mcp_request` (12–18 h)

### Entregables
- Definir schema tool `mcp_request`.
- Parse robusto `tool_calls.arguments` (JSON string).
- Rama en `/api/send` para detectar tool_call.

### Tareas
- Validación mínima de campos.
- Mensaje “Plan propuesto… confirma/cancela”.

⏱ 1.5–2.5 días

---

## Fase 5 – Sistema de planes (PlanRun) (20–32 h)

### Entregables
- `PlanRun` y `PlanStep`.
- `PlanRunStore` (primero en RAM, luego persistente).
- Estados:
  - `draft` (pendiente)
  - `queued` (confirmado)
  - `running` (ejecutando)
  - `done` / `error`

### Tareas
- Serializar `plan_json`.
- Guardar `pending_run_id` en estado de chat.

⏱ 3–4 días

---

## Fase 6 – Confirmación robusta e idempotencia (16–26 h)

### Entregables
- Detección de confirmación/cancelación.
- Guard de timing:
  - si `active_run_id` → “ya ejecutándose”
  - si no hay pending → “no hay plan pendiente”
- CAS `draft → queued` para evitar doble confirmación.

### Tareas
- Tests manuales de race conditions (doble click, doble enter).

⏱ 2–3.5 días

---

## Fase 7 – MCP: registro + discovery OpenAPI (20–32 h)

### Entregables
- `MCPStore` + `MCPService`.
- `discover_openapi()`:
  - normaliza base_url
  - prueba `/openapi.json`, etc.
  - extrae endpoints allowlist

### Tareas
- UI: listar MCPs, alta/baja, refresh.
- Manejar offline (`allow_offline`).

⏱ 3–4 días

---

## Fase 8 – Invocación MCP segura (12–18 h)

### Entregables
- `invoke_mcp_sync()`:
  - solo endpoints permitidos por OpenAPI
  - devuelve `(status_code, json|text)`

### Tareas
- Timeout.
- Logging de request/response resumidos.

⏱ 1.5–2.5 días

---

## Fase 9 – Runner en background (20–30 h)

### Entregables
- `run_plan_in_background()`:
  - ejecuta steps
  - emite mensajes al chat
  - actualiza estado en PlanRunStore

### Tareas
- `asyncio.create_task` + callback de error.
- Finalización limpia (set last_run_*).

⏱ 3–4 días

---

## Fase 10 – Retry + reasoner para `/command` (12–22 h)

### Entregables
- Heurística `_command_failed`.
- Retry N intentos.
- `agent_reasoner` (un paso):
  - propone cmd corregido al último intento.
  - bloquea cmd peligroso.

### Riesgos
- Balance entre “ser útil” y “no ejecutar cosas riesgosas”.

⏱ 2–3 días

---

## Fase 11 – Persistencia real (SQLite) + recovery (12–20 h)

### Entregables
- `SqliteChatStateRepo`.
- `SqlitePlanRunStore`.
- `recover_stale_runs` en startup.

### Tareas
- Migrar estado crítico desde RAM.
- Probar reload con runs activos.

⏱ 1.5–2.5 días

---

## Fase 12 – UI completa (18–28 h)

### Entregables
- Sidebar Projects/Chats/MCPs.
- Modales (crear/eliminar/renombrar).
- Polling de runs.

### Tareas
- UX: estados “Pensando… / Ejecutando…”.
- Errores amigables.

⏱ 2.5–4 días

---

## Total realista (con IA)

- **170–260 horas**
- **22–34 días hábiles**

> Con IA se reduce mucho el boilerplate, pero la parte “dura” (estado, concurrencia, recovery, seguridad) sigue llevando tiempo real.

---

## Total aproximado (sin IA)

- **260–420 horas**
- **34–55 días hábiles**

---

## Qué acelera más con IA

- UI HTML/JS (componentes, estilos, modales).
- CRUD endpoints y modelos.
- Refactors grandes (mover lógica, renombrar, ordenar módulos).
- Logs y mensajes de error consistentes.

## Qué NO conviene delegar a IA (o solo parcialmente)

- Políticas de seguridad (allowlists, permisos).
- Diseño de estados / recovery.
- Edge cases de ejecución (retry, cancelación, idempotencia).

---

## Roadmap inmediato recomendado (v2 → v3)

1) Persistir mensajes (hoy están en RAM).
2) Streaming (SSE/websocket) en vez de polling.
3) Permisos por MCP y por endpoint.
4) Auditoría: log estructurado por step + hash de `plan_json`.
5) Sandbox para comandos (allowlist real + límites por proyecto).

