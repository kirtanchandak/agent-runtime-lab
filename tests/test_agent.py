import json
import threading

import pytest

from agent_runtime import (
    Agent,
    ModelDecision,
    ProviderError,
    RunStatus,
    StepKind,
    Tool,
    ToolArgs,
    ToolCall,
    ToolRegistry,
    format_trace,
)


class AddArgs(ToolArgs):
    a: int
    b: int


class NoArgs(ToolArgs):
    pass


class ScriptedModel:
    def __init__(self, decisions: list[ModelDecision]) -> None:
        self._decisions = list(decisions)
        self.calls: list[dict] = []

    def complete(self, messages, tools):
        self.calls.append({"messages": [dict(message) for message in messages], "tools": tools})
        if not self._decisions:
            raise ProviderError("scripted model has no decision left")
        return self._decisions.pop(0)


def _add_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="add",
            description="Add two integers.",
            args_model=AddArgs,
            fn=lambda args: str(args.a + args.b),
        )
    )
    return registry


def _decision(content: str | None, calls: list[ToolCall], message: dict | None = None) -> ModelDecision:
    if message is None:
        message = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in calls
            ]
    return ModelDecision(content=content, tool_calls=calls, assistant_message=message)


def test_text_answer_completes_the_run():
    provider = ScriptedModel([_decision("done", [])])
    agent = Agent(provider, _add_registry(), system_prompt="be brief")

    run = agent.run("say done")

    assert run.status is RunStatus.COMPLETED
    assert run.result == "done"
    assert [(step.kind, step.content) for step in run.steps] == [(StepKind.MODEL, "done")]
    assert provider.calls[0]["messages"][0] == {"role": "system", "content": "be brief"}
    assert provider.calls[0]["messages"][1] == {"role": "user", "content": "say done"}
    assert provider.calls[0]["tools"][0]["name"] == "add"


def test_several_tools_then_an_answer():
    first = [
        ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3}),
        ToolCall(id="c2", name="add", arguments={"a": 10, "b": 1}),
    ]
    provider = ScriptedModel([_decision(None, first), _decision("13", [])])
    agent = Agent(provider, _add_registry())

    run = agent.run("add some numbers")

    assert run.status is RunStatus.COMPLETED
    assert run.result == "13"
    assert [step.kind for step in run.steps] == [
        StepKind.MODEL,
        StepKind.TOOL,
        StepKind.TOOL,
        StepKind.MODEL,
    ]
    assert run.steps[1].output == "5"
    assert run.steps[2].output == "11"
    sent = provider.calls[1]["messages"]
    assert sent[1]["role"] == "assistant"
    assert sent[2] == {"role": "tool", "tool_call_id": "c1", "content": "5"}
    assert sent[3] == {"role": "tool", "tool_call_id": "c2", "content": "11"}
    trace = json.loads(format_trace(run))
    assert trace["status"] == "completed"
    assert trace["result"] == "13"
    assert [step["kind"] for step in trace["steps"]] == ["model", "tool", "tool", "model"]


def test_tool_error_is_sent_back_to_the_model():
    provider = ScriptedModel(
        [
            _decision(None, [ToolCall(id="c1", name="add", arguments={"a": 1})]),
            _decision("the call was missing b", []),
        ]
    )
    agent = Agent(provider, _add_registry())

    run = agent.run("add")

    assert run.status is RunStatus.COMPLETED
    assert run.steps[1].error
    assert provider.calls[1]["messages"][-1]["content"].startswith("error:")


def test_max_steps_stops_before_another_model_call():
    provider = ScriptedModel(
        [
            _decision(None, [ToolCall(id="c1", name="add", arguments={"a": 1, "b": 1})]),
            _decision("should not be asked", []),
        ]
    )
    agent = Agent(provider, _add_registry(), max_steps=2)

    run = agent.run("add")

    assert run.status is RunStatus.FAILED
    assert run.error == "exceeded max steps (2)"
    assert len(provider.calls) == 1
    assert [step.kind for step in run.steps] == [StepKind.MODEL, StepKind.TOOL]
    assert run.result is None


def test_max_steps_skips_tool_calls_that_do_not_fit():
    calls = [
        ToolCall(id="c1", name="add", arguments={"a": 1, "b": 1}),
        ToolCall(id="c2", name="add", arguments={"a": 2, "b": 2}),
    ]
    provider = ScriptedModel([_decision(None, calls)])
    agent = Agent(provider, _add_registry(), max_steps=2)

    run = agent.run("add twice")

    assert run.status is RunStatus.FAILED
    assert [step.tool_name for step in run.steps if step.kind is StepKind.TOOL] == ["add"]
    assert run.steps[-1].tool_call_id == "c1"


def test_provider_error_fails_the_run():
    class Broken:
        def complete(self, messages, tools):
            raise ProviderError("OpenRouter request failed: 401")

    run = Agent(Broken(), _add_registry()).run("hi")

    assert run.status is RunStatus.FAILED
    assert run.error == "OpenRouter request failed: 401"
    assert run.steps == []


def test_tool_timeout_is_recorded(monkeypatch):
    release = threading.Event()

    def hang(_: NoArgs) -> str:
        release.wait(5)
        return "late"

    registry = ToolRegistry()
    registry.register(Tool(name="hang", description="Blocks.", args_model=NoArgs, fn=hang))
    provider = ScriptedModel(
        [
            _decision(None, [ToolCall(id="c1", name="hang")]),
            _decision("stopped waiting", []),
        ]
    )
    agent = Agent(provider, registry, tool_timeout=0.05)

    try:
        run = agent.run("hang")
    finally:
        release.set()

    assert run.status is RunStatus.COMPLETED
    assert run.steps[1].error == "tool timed out after 0.05s"
    assert provider.calls[1]["messages"][-1]["content"] == "error: tool timed out after 0.05s"


def test_limits_must_be_positive():
    provider = ScriptedModel([])
    with pytest.raises(ValueError):
        Agent(provider, _add_registry(), max_steps=0)
    with pytest.raises(ValueError):
        Agent(provider, _add_registry(), tool_timeout=0)
