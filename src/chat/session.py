from typing import List, Dict
from src.chat.prompts import DEFAULT_SYSTEM_PROMPT

class ChatSession:
    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self.system_prompt = system_prompt
        self.reset()

    def reset(self) -> None:
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages
