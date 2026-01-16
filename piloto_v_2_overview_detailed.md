# Piloto v2 – Documentación Técnica (Detallada)

> Este documento describe **qué hace Piloto v2**, **cómo está implementado**, y **cómo fluye el estado** entre Frontend ⇄ Backend ⇄ LLM ⇄ MCP.

---

## 1) Qué es Piloto v2

Piloto v2 es un **chat estilo ChatGPT** con capacidad de ejecutar acciones externas mediante **planes confirmables**.

El sistema soporta dos modos:
- **Chat normal**: el modelo responde con texto.
- **Acciones (MCP)**: el modelo devuelve un `tool_call` (`mcp_request`) y el backend lo convierte en un **PlanRun** que el usuario debe **confirmar**.

La decisión arquitectónica central es separar:
- **Intención / razonamiento (LLM)**: produce tool-call estandarizada.
- **Ejecución / efectos (Backend)**: valida, persiste y ejecuta en background.

---

## 2) Arquitectura a alto nivel

### 2.1 Flujo general
1. UI envía mensaje → `POST /api/send`.
2. Backend consulta estado del chat (SQLite).
3. Backend llama al LLM con `tools=[mcp_request]`.
4. Si el LLM responde texto → se guarda como mensaje assistant.
5. Si responde `mcp_request` → se crea `PlanRun` en estado `draft` + `pending_run_id`.
6. Usuario confirma (“confirmo”).
7. Backend marca `draft → queued` (CAS) y ejecuta runner en background.
8. Runner invoca MCPs, agrega mensajes, finaliza run.

### 2.2 Principios y garantías
- Nada se ejecuta sin confirmación.
- Endpoints MCP permitidos = solo los descubiertos por OpenAPI.
- Confirmación idempotente: doble confirmación no duplica ejecución.
- Si el server se recarga, los runs activos se marcan error (recovery).

---

## 3) Backend (FastAPI) – módulos y responsabilidades

### 3.1 `src/main.py`
- Crea la app FastAPI.
- Middleware de trazas:
  - Genera/propaga `X-Trace-Id`.
  - Loguea `request.start` y `request.end` con duración.
- Carga router principal.

### 3.2 `src/app_state.py`
Inicializa singletons globales:
- **Config**: lee `OPENAI_API_KEY` y `OPENAI_MODEL`.
- **LLM**: instancia `OpenAIChatClient`.
- **SQLite**:
  - `SqliteChatStateRepo` (estado por chat)
  - `SqlitePlanRunStore` (runs)
- **Stores**:
  - `MemoryStore` (proyectos/chats/mensajes en RAM, estado en SQLite)
  - `MCPStore` (MCPs en RAM)
- **Servicios**:
  - `MCPService` (register/refresh/etc.)

### 3.3 `src/web/routes.py`
Define:
- UI: `GET /` devuelve `index.html` (Jinja2)
- APIs:
  - Projects
  - Chats y Messages
  - MCP (register, refresh, active, delete)
  - Runs
  - **`POST /api/send`** (router central)

Además contiene helpers:
- `_looks_like_plan_text(text)` → anti-plan fantasma.
- `_looks_like_confirmation_prompt(text)` → detectar pedidos de confirmación.
- `_extract_command(text)` → parse simple “ejecuta …”.

---

## 4) LLM y Tool Calling (contrato exacto)

### 4.1 Tool schema: `mcp_request`
Parámetros:
- `mcp_id: string` (requerido)
- `method: enum` (GET/POST/PUT/PATCH/DELETE)
- `path: string` (requerido)
- `query: object` (opcional)
- `body: object|string|null` (opcional)

Restricción importante:
- `additionalProperties: false` (si el modelo inventa keys, se puede invalidar).

### 4.2 Parse robusto de tool args
En Chat Completions, los argumentos suelen venir como **string JSON**.
Por eso el wrapper del cliente intenta `json.loads(arguments)`.

---

## 5) Persistencia: qué se guarda y por qué

### 5.1 `chat_state` (SQLite)
Se guarda **solo estado crítico** (no mensajes):
- `pending_run_id`
- `active_run_id`
- `last_run_id`
- `last_run_status`
- `last_run_ts`

**Detalle clave**: `get_state()` retorna solo keys con valores != None (evita “pending_run_id=None” fantasma).

### 5.2 `plan_runs` (SQLite)
Se guarda por run:
- Identidad: `run_id`, `chat_id`, `plan_id`
- `goal`
- `status` (`draft|queued|running|done|error`)
- `plan_json` (plan serializado)
- `current_step_path`, `last_event`
- `error`

---

## 6) `POST /api/send` – implementación y ramas

### 6.1 Entrada
`{ chat_id, message }`

### 6.2 Lecturas iniciales
- Estado del chat (pending/active).
- Historial de mensajes para armar payload del modelo.

### 6.3 Rama 1: confirmación/cancelación
El sistema detecta confirmación/cancelación por patrones (confirmo/ok/dale vs cancela/no/detener).

**Reglas de guard (robustez):**
- Si el usuario confirma/cancela y no existe `pending_run_id`: responde “No hay plan pendiente”.
- Si existe `active_run_id`: responde “Ya está ejecutándose”.

**Confirmación:**
1. Recupera run draft.
2. Reconstruye el plan desde `plan_json`.
3. Transición idempotente `draft → queued` con CAS (`try_mark_queued`).
   - Si falla: 409 (otro request ganó o ya no estaba en draft).
4. Agrega mensaje al chat: “Confirmado. Ejecutando plan…”.
5. Actualiza estado del chat:
   - `pending_run_id = None`
   - `active_run_id = run_id`
6. Lanza `run_plan_in_background()`.

### 6.4 Rama 2: mensaje normal
1. Construye payload del modelo:
   - System prompt base + contexto del proyecto.
   - Historial de mensajes del chat.
2. Llama al modelo con `tools=[mcp_request]`.
3. Si no hay tool_call:
   - Agrega respuesta a chat.
4. Si hay tool_call:
   - Crea `PlanRunState` status `draft`.
   - Guarda `plan_json`.
   - Setea `pending_run_id` en chat.
   - Responde con mensaje “Plan propuesto… responde confirmo/cancela”.

---

## 7) Anti-plan fantasma (detallado)

### 7.1 Qué detecta
Cuando **el modelo** escribe planes tipo:
- “Plan propuesto”
- “Paso 1 / Paso 2 …”
- “Responde confirmo para ejecutar”

pero **sin tool_call real**.

### 7.2 Cómo lo detecta
`_looks_like_plan_text`:
- Busca patrones como `plan`, `paso`, `step`, `propuesto`, `running`, `action`.
- Si encuentra 2+ coincidencias → lo considera plan en texto.

Uso típico:
- Loguear y forzar reintento para que el modelo devuelva tool_call.

---

## 8) Runner: `run_plan_in_background` (ejecución paso a paso)

### 8.1 Objetivos del runner
- Ejecutar el plan de forma determinista.
- Emitir mensajes al chat mientras corre.
- Reintentar solo cuando corresponde.
- Cerrar el run actualizando estado y persistencia.

### 8.2 Emisiones
Mensajes típicos:
- Inicio: `⏳ Iniciando plan ...`
- Por step:
  - `⏳ step_start`
  - `✅ step_ok` / `❌ step_err`
  - `⚠️ step_retry`
- Fin: `Plan ... finalizado con estado ...`

### 8.3 Validación y permisos
Antes de invocar:
- MCP existe.
- MCP activo.
- Si hay proyecto: MCP debe estar habilitado en `proj.mcp_ids`.

### 8.4 Ejecución segura contra MCP
`invoke_mcp_sync` verifica allowlist:
- Solo se permite `(method, path)` si existe en `mcp.endpoints`.

### 8.5 Reintentos y reasoner (solo `/command`)
Para `/command`:
- Si HTTP no es 2xx → error inmediato.
- Si el comando falla (exit_code/status):
  - reintenta hasta N.
  - en el último intento, llama `reason_about_command_failure`.
  - si el reasoner propone un comando NO peligroso → nuevo intento.

`agent_reasoner`:
- Evita repetir el comando.
- Evita comandos destructivos.
- Mantiene intención.

---

## 9) MCP: registro, discovery y refresh

### 9.1 Normalización de URL
`_normalize_base_url`:
- Si falta esquema: agrega `http://`.
- Si viene `/docs` o `/redoc`: lo recorta.
- Remueve trailing slash.

### 9.2 Discovery
`discover_openapi`:
- Prueba rutas comunes: `/openapi.json`, `/swagger.json`, etc.
- Extrae endpoints desde `paths`.
- Loguea errores (timeouts/connect/http).

### 9.3 Register offline
`MCPService.register(... allow_offline=True)`:
- Registra el MCP aunque el host esté offline.
- Permite conectar luego y hacer refresh.

---

## 10) Frontend (index.html) – comportamiento exacto

### 10.1 Guard local de confirmación
`sendMessage()` mantiene sets:
- `CONFIRM`: confirmo/ok/si/dale/ejecuta...
- `CANCEL`: cancela/no/detener/para...

Si el usuario confirma/cancela y `State.pendingRunId` es null:
- No llama backend.
- Muestra: “No hay ningún plan pendiente…”.

### 10.2 Envío y polling
Luego de `/api/send`:
- Recarga mensajes.
- Recarga lista de chats.
- Si devuelve `queued` o `running`, invoca `pollRun(runId)`.

---

## 11) Recovery en reload

En startup se ejecuta `recover_stale_runs`:
- Detecta runs `queued/running` tras recarga.
- Los marca `error`.
- Agrega mensaje al chat avisando.

---

## 12) Checklist de invariantes

1. Un plan ejecutable debe existir como `plan_runs.plan_json`.
2. Confirmación solo si hay `pending_run_id`.
3. Doble confirmación bloqueada por CAS `draft→queued` + guard `active_run_id`.
4. Endpoints solo por allowlist OpenAPI.
5. Reasoner solo para `/command`.

