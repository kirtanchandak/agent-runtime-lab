import pytest
from pydantic import BaseModel

from agent_runtime import Tool, ToolArgs, ToolCall, ToolRegistry


class AddArgs(ToolArgs):
    a: int
    b: int


class NoArgs(ToolArgs):
    pass


def _add(args: AddArgs) -> str:
    return str(args.a + args.b)


def _registry() -> tuple[ToolRegistry, list[AddArgs]]:
    seen: list[AddArgs] = []

    def add(args: AddArgs) -> str:
        seen.append(args)
        return _add(args)

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="add",
            description="Add two integers.",
            args_model=AddArgs,
            fn=add,
        )
    )
    return registry, seen


def test_valid_call_reaches_the_function_as_a_parsed_model():
    registry, seen = _registry()

    result = registry.execute(ToolCall(id="call_1", name="add", arguments={"a": 2, "b": 3}))

    assert result.ok
    assert result.output == "5"
    assert result.error is None
    assert seen == [AddArgs(a=2, b=3)]


def test_invalid_arguments_never_call_the_function():
    registry, seen = _registry()

    missing = registry.execute(ToolCall(id="call_1", name="add", arguments={"a": 2}))
    wrong_type = registry.execute(ToolCall(id="call_2", name="add", arguments={"a": "nope", "b": 3}))
    extra = registry.execute(
        ToolCall(id="call_3", name="add", arguments={"a": 2, "b": 3, "c": 1})
    )

    assert seen == []
    assert missing.ok is False
    assert missing.output is None
    assert "b:" in (missing.error or "")
    assert wrong_type.ok is False
    assert "a:" in (wrong_type.error or "")
    assert extra.ok is False
    assert "c:" in (extra.error or "")


def test_unknown_tool_is_a_result():
    registry, seen = _registry()

    result = registry.execute(ToolCall(id="call_1", name="read_file", arguments={}))

    assert seen == []
    assert result.ok is False
    assert result.error == "unknown tool: read_file"


def test_tool_exception_is_a_result():
    registry = ToolRegistry()

    def boom(_: NoArgs) -> str:
        raise RuntimeError("disk full")

    registry.register(Tool(name="boom", description="Always fails.", args_model=NoArgs, fn=boom))

    result = registry.execute(ToolCall(id="call_1", name="boom"))

    assert result.ok is False
    assert result.error == "disk full"


def test_tool_must_return_a_string():
    registry = ToolRegistry()

    def bad(_: NoArgs) -> str:
        return 5  # type: ignore[return-value]

    registry.register(Tool(name="bad", description="Returns an int.", args_model=NoArgs, fn=bad))

    result = registry.execute(ToolCall(id="call_1", name="bad"))

    assert result.error == "bad returned int, expected str"


def test_empty_arguments_are_valid_when_the_schema_has_no_fields():
    registry = ToolRegistry()
    registry.register(
        Tool(name="ping", description="Reply pong.", args_model=NoArgs, fn=lambda _: "pong")
    )

    result = registry.execute(ToolCall(id="call_1", name="ping"))

    assert result.output == "pong"


def test_schema_describes_the_argument_model():
    registry, _ = _registry()

    schema = registry.schemas()[0]

    assert schema["name"] == "add"
    assert schema["description"] == "Add two integers."
    assert schema["parameters"]["properties"]["a"]["type"] == "integer"
    assert set(schema["parameters"]["required"]) == {"a", "b"}


def test_duplicate_registration_is_rejected():
    registry, _ = _registry()

    with pytest.raises(ValueError, match="already registered"):
        registry.register(
            Tool(name="add", description="Again.", args_model=AddArgs, fn=_add)
        )


def test_args_model_must_subclass_tool_args():
    class Plain(BaseModel):
        a: int

    registry = ToolRegistry()

    with pytest.raises(TypeError, match="ToolArgs"):
        registry.register(Tool(name="plain", description="Nope.", args_model=Plain, fn=lambda args: "x"))
