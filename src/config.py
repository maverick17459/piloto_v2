import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str
    temperature: float = 0.7

def get_settings() -> Settings:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

    if not api_key:
        raise RuntimeError("Falta OPENAI_API_KEY. Crea un .env basado en .env.example.")

    return Settings(api_key=api_key, model=model)



