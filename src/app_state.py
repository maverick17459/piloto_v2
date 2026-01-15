# src/app_state.py
import os

from src.config import get_settings
from src.llm.openai_client import OpenAIChatClient
from src.chat.store import MemoryStore
from src.chat.prompts import DEFAULT_SYSTEM_PROMPT
from src.mcp.store import MCPStore
from src.mcp.service import MCPService
from src.persistence.sqlite_repos import SqliteChatStateRepo, SqlitePlanRunStore

settings = get_settings()

# -----------------------
# SQLite path
# -----------------------
DB_PATH = os.getenv("PILOTO_DB_PATH", "data/piloto.db")
db_dir = os.path.dirname(DB_PATH)
if db_dir:
    os.makedirs(db_dir, exist_ok=True)

# -----------------------
# LLM client (global)
# -----------------------
client = OpenAIChatClient(api_key=settings.api_key, model=settings.model)

# -----------------------
# Persistence (SQLite)
# -----------------------
state_repo = SqliteChatStateRepo(DB_PATH)
plan_run_store = SqlitePlanRunStore(DB_PATH)

# -----------------------
# Stores / Services
# -----------------------
store = MemoryStore(base_system_prompt=DEFAULT_SYSTEM_PROMPT, state_repo=state_repo)

mcp_store = MCPStore()
mcp_service = MCPService(mcp_store)
