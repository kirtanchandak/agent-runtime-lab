"""Named functions the model is allowed to call.

A tool is a schema plus a function. The registry checks the model's
arguments against that schema, then calls the function with the parsed
object. The function never sees the raw dict.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from agent_runtime.models import ToolCall


class ToolArgs(BaseModel):
    """Base class for every tool's argument model.

    Extra keys are rejected. The model must send only the fields this
    tool declared.
    """

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[ToolArgs]
    fn: Callable[[Any], str]

    def schema(self) -> dict[str, Any]:
        parameters = self.args_model.model_json_schema()
        parameters.pop("title", None)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": parameters,
        }


@dataclass(frozen=True)
class ToolResult:
    """What one execution produced.

    A bad call is still a result: `error` is set and `output` is empty.
    The agent loop records that on a tool step and can show it to the model.
    """

    output: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("tool name is required")
        if not issubclass(tool.args_model, ToolArgs):
            raise TypeError(f"{tool.name} args_model must subclass ToolArgs")
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(error=f"unknown tool: {call.name}")
        try:
            parsed = tool.args_model.model_validate(call.arguments)
        except ValidationError as exc:
            return ToolResult(error=_format_validation_error(exc))
        try:
            output = tool.fn(parsed)
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            return ToolResult(error=message)
        if not isinstance(output, str):
            return ToolResult(error=f"{tool.name} returned {type(output).__name__}, expected str")
        return ToolResult(output=output)


def _format_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(item) for item in err["loc"]) or "arguments"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)
