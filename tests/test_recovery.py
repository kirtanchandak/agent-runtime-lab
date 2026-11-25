import pytest

from agent_runtime import (
    Agent,
    AttemptStatus,
    ModelDecision,
    RunStatus,
    RunStore,
    Tool,
    ToolArgs,
    ToolCall,
    ToolRegistry,
    sqlite_url,
)


class WriteArgs(ToolArgs):
    path: str
    content: str


class OutcomeNotRecorded(Exception):
    pass


class KilledBeforeNextDecision(Exception):
    pass


class DropOutcome:
    """Store whose outcome commit never lands. The intent commit still does."""

    def __init__(self, store: RunStore) -> None:
        self._store = store

    def __getattr__(self, name: str):
        return getattr(self._store, name)

    def finish_tool(self, *args, **kwargs):
        raise OutcomeNotRecorded()


def _registry(fn) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool(name="write_file", description="Write a file.", args_model=WriteArgs, fn=fn))
    return registry


def _write_decision() -> ModelDecision:
    call = ToolCall(id="c1", name="write_file", arguments={"path": "note.txt", "content": "hello"})
    return ModelDecision(
        content=None,
        tool_calls=[call],
        assistant_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": '{"path": "note.txt", "content": "hello"}'},
                }
            ],
        },
    )


def _answer(text: str) -> ModelDecision:
    return ModelDecision(
        content=text,
        tool_calls=[],
        assistant_message={"role": "assistant", "content": text},
    )


def test_resume_does_not_repeat_a_recorded_tool(tmp_path):
    database = tmp_path / "runs.sqlite3"
    calls: list[str] = []

    def write(args: WriteArgs) -> str:
        calls.append(args.content)
        (tmp_path / args.path).write_text(args.content)
        return "wrote"

    class FirstModel:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages, tools):
            self.calls += 1
            if self.calls == 2:
                raise KilledBeforeNextDecision()
            return _write_decision()

    with pytest.raises(KilledBeforeNextDecision):
        Agent(FirstModel(), _registry(write), store=RunStore(sqlite_url(database))).run("write a note")

    assert calls == ["hello"]
    assert (tmp_path / "note.txt").read_text() == "hello"

    class SecondModel:
        def __init__(self) -> None:
            self.messages = []

        def complete(self, messages, tools):
            self.messages.append(messages)
            return _answer("saved")

    def fail_if_called(_: WriteArgs) -> str:
        raise AssertionError("recorded tool ran again")

    store = RunStore(sqlite_url(database))
    second = SecondModel()
    run = Agent(second, _registry(fail_if_called), store=store).resume(store.ids()[0])

    assert run.status is RunStatus.COMPLETED
    assert run.result == "saved"
    assert calls == ["hello"]
    assert second.messages[0][-1] == {"role": "tool", "tool_call_id": "c1", "content": "wrote"}
    assert store.get_attempt(run.id, "c1").status is AttemptStatus.COMPLETED


def test_unrecorded_outcome_is_not_retried(tmp_path):
    database = tmp_path / "runs.sqlite3"
    note = tmp_path / "note.txt"

    def write(args: WriteArgs) -> str:
        note.write_text(args.content)
        return "wrote"

    class OneDecision:
        def complete(self, messages, tools):
            return _write_decision()

    store = RunStore(sqlite_url(database))
    with pytest.raises(OutcomeNotRecorded):
        Agent(OneDecision(), _registry(write), store=DropOutcome(store)).run("write a note")

    reopened = RunStore(sqlite_url(database))
    run_id = reopened.ids()[0]
    attempt = reopened.get_attempt(run_id, "c1")
    assert attempt is not None
    assert attempt.status is AttemptStatus.STARTED
    assert attempt.idempotency_key == f"{run_id}:c1"
    assert note.read_text() == "hello"

    class UnusedModel:
        def complete(self, messages, tools):
            raise AssertionError("model was called")

    def fail_if_called(_: WriteArgs) -> str:
        raise AssertionError("tool ran again")

    resumed = Agent(UnusedModel(), _registry(fail_if_called), store=reopened).resume(run_id)

    assert resumed.status is RunStatus.FAILED
    assert resumed.error == (
        "outcome unknown for write_file; it may have already run. Reconcile before retrying."
    )
    assert reopened.get_attempt(run_id, "c1").status is AttemptStatus.UNKNOWN
    assert note.read_text() == "hello"


def test_listener_runs_after_the_attempt_is_committed(tmp_path):
    database = tmp_path / "runs.sqlite3"
    store = RunStore(sqlite_url(database))
    other = RunStore(sqlite_url(database))
    seen: list[AttemptStatus] = []

    def listener(event: dict) -> None:
        if event["source"] != "begin_tool":
            return
        attempt = other.get_attempt(event["run_id"], "c1")
        assert attempt is not None
        seen.append(attempt.status)

    store.listen(listener)
    run = Agent(
        type("Model", (), {"complete": lambda self, messages, tools: _answer("done")})(),
        ToolRegistry(),
        store=store,
    ).run("say done")
    store.begin_tool(run.id, ToolCall(id="c1", name="write_file", arguments={"path": "note.txt"}))

    assert seen == [AttemptStatus.STARTED]


def test_finished_run_stays_finished(tmp_path):
    database = tmp_path / "runs.sqlite3"

    class Answer:
        def complete(self, messages, tools):
            return _answer("done")

    store = RunStore(sqlite_url(database))
    first = Agent(Answer(), ToolRegistry(), store=store).run("say done")

    class UnusedModel:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages, tools):
            self.calls += 1
            raise AssertionError("model was called")

    model = UnusedModel()
    again = Agent(model, ToolRegistry(), store=RunStore(sqlite_url(database))).resume(first.id)

    assert again.status is RunStatus.COMPLETED
    assert again.result == "done"
    assert model.calls == 0
