# src/app_state.py
from src.config import get_settings
from src.llm.openai_client import OpenAIChatClient
from src.chat.store import MemoryStore
from src.chat.prompts import DEFAULT_SYSTEM_PROMPT
from src.mcp.store import MCPStore
from src.mcp.service import MCPService
from src.agent.plan_run_store import PlanRunStore

settings = get_settings()

# LLM client (global)
client = OpenAIChatClient(api_key=settings.api_key, model=settings.model)

# In-memory stores (se reinician al reiniciar)
store = MemoryStore(base_system_prompt=DEFAULT_SYSTEM_PROMPT)

mcp_store = MCPStore()
mcp_service = MCPService(mcp_store)

plan_run_store = PlanRunStore()



