"""OpenRouter chat completions.

The request matches the call in v1: POST /chat/completions with a model,
messages, and function tools. The key is read from the environment.
"""

import json
import os
from collections.abc import Callable
from typing import Any

import requests

from agent_runtime.models import ToolCall
from agent_runtime.providers.base import ModelDecision, ProviderError

DEFAULT_MODEL = "openrouter/free"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterProvider:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60,
        post: Callable[..., Any] | None = None,
    ) -> None:
        if not api_key:
            raise ProviderError("OpenRouter API key is empty")
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._api_key = api_key
        self._post = post or requests.post

    @classmethod
    def from_env(cls) -> "OpenRouterProvider":
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ProviderError("OPENROUTER_API_KEY is not set")
        model = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
        return cls(api_key=api_key, model=model)

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelDecision:
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = [_function_tool(tool) for tool in tools]
        try:
            response = self._post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ProviderError(f"OpenRouter request failed: {exc}") from exc
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise ProviderError(f"OpenRouter request failed: {exc}") from exc
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError("OpenRouter response was not JSON") from exc
        return _parse_decision(body)


def _function_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        },
    }


def _parse_decision(body: dict[str, Any]) -> ModelDecision:
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError("OpenRouter response did not include a message") from exc
    if not isinstance(message, dict):
        raise ProviderError("OpenRouter message was not an object")

    tool_calls: list[ToolCall] = []
    for call in message.get("tool_calls") or []:
        tool_calls.append(_parse_tool_call(call))

    content = message.get("content")
    if not isinstance(content, str):
        content = None
    if not tool_calls and content is None:
        content = ""
    return ModelDecision(content=content, tool_calls=tool_calls, assistant_message=message)


def _parse_tool_call(call: Any) -> ToolCall:
    if not isinstance(call, dict):
        raise ProviderError("tool call was not an object")
    function = call.get("function") or {}
    if not isinstance(function, dict):
        raise ProviderError("tool call is missing a function")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ProviderError("tool call is missing a function name")
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id:
        raise ProviderError(f"tool call {name} is missing an id")
    return ToolCall(id=call_id, name=name, arguments=_parse_arguments(function.get("arguments")))


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"tool arguments are not valid JSON: {raw}") from exc
    elif isinstance(raw, dict):
        parsed = raw
    else:
        raise ProviderError("tool arguments must be a JSON object")
    if not isinstance(parsed, dict):
        raise ProviderError("tool arguments must be a JSON object")
    return parsed
