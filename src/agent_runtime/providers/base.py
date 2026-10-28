"""What the agent loop needs from any model.

`complete` takes the conversation and the tool schemas, and returns a
decision. The HTTP client, URL, and response shape stay inside one
provider.
"""

from typing import Any, Protocol

from pydantic import BaseModel, Field

from agent_runtime.models import ToolCall


class ProviderError(Exception):
    """The model request failed, or the response was not usable."""


class ModelDecision(BaseModel):
    """One model turn.

    `tool_calls` means the run is not finished. `content` is the final
    answer when `tool_calls` is empty. `assistant_message` is the raw
    message the provider must send back on the next request, unchanged.
    """

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    assistant_message: dict[str, Any]


class ModelProvider(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelDecision:
        """Return the model's next decision for this conversation."""
