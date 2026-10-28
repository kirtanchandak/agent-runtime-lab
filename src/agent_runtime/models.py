"""The nouns of one agent execution.

Nothing here calls a model or a tool. A run is a state machine, and
later modules only append steps and move it between these states.
"""

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# Terminal states have an empty set: once a run ends, it stays ended.
_TRANSITIONS: dict[RunStatus, set[RunStatus]] = {
    RunStatus.CREATED: {RunStatus.RUNNING},
    RunStatus.RUNNING: {RunStatus.COMPLETED, RunStatus.FAILED},
    RunStatus.COMPLETED: set(),
    RunStatus.FAILED: set(),
}


class InvalidTransition(Exception):
    def __init__(self, current: RunStatus, target: RunStatus) -> None:
        super().__init__(f"cannot move from {current.value} to {target.value}")
        self.current = current
        self.target = target


class ToolCall(BaseModel):
    """One request from the model to run a named tool.

    `arguments` is an untrusted dict. The tool registry validates it
    later. The model is allowed to be wrong; the registry rejects it.
    """

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class StepKind(str, Enum):
    MODEL = "model"
    TOOL = "tool"


class Step(BaseModel):
    """One recorded event, in order.

    A model step is the model's decision: a final answer, tool calls, or both.
    A tool step is what happened when one of those calls was executed.
    """

    index: int
    kind: StepKind
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None
    output: str | None = None
    error: str | None = None


class Run(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    task: str
    status: RunStatus = RunStatus.CREATED
    steps: list[Step] = Field(default_factory=list)
    result: str | None = None
    error: str | None = None

    def start(self) -> None:
        self._transition(RunStatus.RUNNING)

    def complete(self, result: str) -> None:
        self._transition(RunStatus.COMPLETED)
        self.result = result

    def fail(self, error: str) -> None:
        self._transition(RunStatus.FAILED)
        self.error = error

    def add_step(self, step: Step) -> None:
        if self.status is not RunStatus.RUNNING:
            raise RuntimeError(f"cannot record a step while status is {self.status.value}")
        if step.index != len(self.steps):
            raise ValueError(f"expected step index {len(self.steps)}, got {step.index}")
        self.steps.append(step)

    def _transition(self, target: RunStatus) -> None:
        if target not in _TRANSITIONS[self.status]:
            raise InvalidTransition(self.status, target)
        self.status = target
