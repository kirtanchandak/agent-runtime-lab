import pytest

from agent_runtime import InvalidTransition, Run, RunStatus, Step, StepKind, ToolCall


def test_new_run_is_created_and_empty():
    run = Run(task="read the file")

    assert run.status is RunStatus.CREATED
    assert run.steps == []
    assert run.result is None
    assert run.error is None
    assert run.id


def test_happy_path_created_running_completed():
    run = Run(task="read the file")
    run.start()
    run.add_step(Step(index=0, kind=StepKind.MODEL, content="the file says hello"))
    run.complete("the file says hello")

    assert run.status is RunStatus.COMPLETED
    assert run.result == "the file says hello"
    assert [step.index for step in run.steps] == [0]


def test_running_run_can_fail():
    run = Run(task="read the file")
    run.start()
    run.fail("tool timed out")

    assert run.status is RunStatus.FAILED
    assert run.error == "tool timed out"
    assert run.result is None


def test_created_run_cannot_complete_or_record_steps():
    run = Run(task="read the file")

    with pytest.raises(InvalidTransition) as completed:
        run.complete("done")
    assert completed.value.current is RunStatus.CREATED
    assert completed.value.target is RunStatus.COMPLETED

    with pytest.raises(RuntimeError):
        run.add_step(Step(index=0, kind=StepKind.MODEL, content="too early"))

    assert run.status is RunStatus.CREATED
    assert run.steps == []


def test_finished_run_cannot_start_again():
    run = Run(task="read the file")
    run.start()
    run.complete("done")

    with pytest.raises(InvalidTransition):
        run.start()

    assert run.status is RunStatus.COMPLETED


def test_step_indexes_must_be_sequential():
    run = Run(task="read the file")
    run.start()

    with pytest.raises(ValueError):
        run.add_step(Step(index=1, kind=StepKind.MODEL, content="skipped zero"))

    assert run.steps == []


def test_tool_call_arguments_default_to_an_empty_dict():
    call = ToolCall(id="call_1", name="list_files")

    assert call.arguments == {}
