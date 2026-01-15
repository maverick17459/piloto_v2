from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union, Literal, overload
from openai import OpenAI


ReturnMode = Literal["text", "message", "raw"]


class OpenAIChatClient:
    """
    Wrapper compatible con tu implementación actual, pero con soporte real de tool-calling.

    - mode="text"   -> str (compatibilidad hacia atrás)
    - mode="message"-> dict estable con: role/content/tool_calls (si los hay) + debug útil
    - mode="raw"    -> objeto/response completo del SDK (por si necesitas más campos)
    """

    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    @overload
    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        mode: Literal["text"] = "text",
    ) -> str: ...

    @overload
    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        mode: Literal["message"] = "message",
    ) -> Dict[str, Any]: ...

    @overload
    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        mode: Literal["raw"] = "raw",
    ) -> Any: ...

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        mode: ReturnMode = "text",
    ) -> Union[str, Dict[str, Any], Any]:
        """
        Ejecuta Chat Completions.

        - Si pasas tools/tool_choice, el modelo puede devolver tool_calls.
        - Para leer tool_calls: usa mode="message" o mode="raw".
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        if mode == "raw":
            return resp

        # dict estable + debug útil
        msg_dict: Dict[str, Any] = {
            "role": msg.role,
            "content": msg.content or "",
        }

        # debug opcional (solo informativo; no rompe nada)
        try:
            msg_dict["finish_reason"] = getattr(resp.choices[0], "finish_reason", None)
        except Exception:
            msg_dict["finish_reason"] = None
        msg_dict["usage"] = getattr(resp, "usage", None)

        tool_calls = getattr(msg, "tool_calls", None)
        msg_dict["has_tool_calls"] = bool(tool_calls)

        if tool_calls:
            msg_dict["tool_calls"] = []
            for tc in tool_calls:
                fn = getattr(tc, "function", None)
                raw_args = getattr(fn, "arguments", None)

                # PARSE ROBUSTO: arguments suele venir como string JSON
                parsed_args: Any = raw_args
                if isinstance(raw_args, str):
                    try:
                        parsed_args = json.loads(raw_args)
                    except Exception:
                        parsed_args = raw_args  # fallback: queda string

                msg_dict["tool_calls"].append(
                    {
                        "id": getattr(tc, "id", None),
                        "type": getattr(tc, "type", "function"),
                        "function": {
                            "name": getattr(fn, "name", None),
                            "arguments": parsed_args,
                        },
                    }
                )

        if mode == "message":
            return msg_dict

        return msg_dict.get("content", "") or ""
