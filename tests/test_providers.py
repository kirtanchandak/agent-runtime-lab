import json

import pytest
import requests

from agent_runtime import ProviderError, ToolCall
from agent_runtime.providers.openrouter import DEFAULT_BASE_URL, OpenRouterProvider


class _Response:
    def __init__(self, payload: dict, status_code: int = 200, text: str | None = None) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload) if text is None else text

    def json(self) -> dict:
        if self.text == "not-json":
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} {self.text}")


def _provider(payload: dict, status_code: int = 200, text: str | None = None):
    seen: dict = {}

    def post(url, headers, json, timeout):  # noqa: A002
        seen["url"] = url
        seen["headers"] = headers
        seen["json"] = json
        seen["timeout"] = timeout
        return _Response(payload, status_code=status_code, text=text)

    provider = OpenRouterProvider(api_key="test-key", post=post)
    return provider, seen


def test_text_answer_uses_the_v1_request_shape():
    provider, seen = _provider(
        {"choices": [{"message": {"role": "assistant", "content": "hello"}}]}
    )

    decision = provider.complete(
        [{"role": "user", "content": "hi"}],
        [
            {
                "name": "add",
                "description": "Add two integers.",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    )

    assert decision.content == "hello"
    assert decision.tool_calls == []
    assert decision.assistant_message["content"] == "hello"
    assert seen["url"] == DEFAULT_BASE_URL
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert seen["json"]["model"] == "openrouter/free"
    assert seen["json"]["messages"] == [{"role": "user", "content": "hi"}]
    assert seen["json"]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two integers.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    assert seen["timeout"] == 60


def test_tool_calls_are_parsed_into_tool_call_objects():
    provider, _ = _provider(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
                            }
                        ],
                    }
                }
            ]
        }
    )

    decision = provider.complete([{"role": "user", "content": "add"}], [])

    assert decision.content is None
    assert decision.tool_calls == [ToolCall(id="call_1", name="add", arguments={"a": 2, "b": 3})]


def test_invalid_tool_arguments_raise():
    provider, _ = _provider(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "add", "arguments": "{"},
                            }
                        ],
                    }
                }
            ]
        }
    )

    with pytest.raises(ProviderError, match="not valid JSON"):
        provider.complete([{"role": "user", "content": "add"}], [])


def test_http_error_and_unusable_body_raise():
    failed, _ = _provider({"error": "nope"}, status_code=401)
    with pytest.raises(ProviderError, match="OpenRouter request failed"):
        failed.complete([{"role": "user", "content": "hi"}], [])

    empty, _ = _provider({"choices": []})
    with pytest.raises(ProviderError, match="did not include a message"):
        empty.complete([{"role": "user", "content": "hi"}], [])

    broken, _ = _provider({}, text="not-json")
    with pytest.raises(ProviderError, match="not JSON"):
        broken.complete([{"role": "user", "content": "hi"}], [])


def test_from_env_reads_the_key_and_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "some/model")

    provider = OpenRouterProvider.from_env()

    assert provider.model == "some/model"


def test_from_env_requires_a_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ProviderError, match="OPENROUTER_API_KEY"):
        OpenRouterProvider.from_env()
