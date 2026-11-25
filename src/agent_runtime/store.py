"""Checkpoints for a run.

A checkpoint is one committed transaction. The tool itself is not inside
that transaction. The intent is committed first, the tool runs, and the
outcome is committed after. A crash in the gap leaves the attempt
`started`: the side effect may already have happened, and it may not.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import ForeignKey, Integer, String, Text, create_engine, event, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from agent_runtime.models import Run, RunStatus, Step, StepKind, ToolCall


class AttemptStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ToolAttempt:
    id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    status: AttemptStatus
    output: str | None
    error: str | None
    idempotency_key: str


@dataclass(frozen=True)
class StoredRun:
    run: Run
    system_prompt: str | None
    max_steps: int
    tool_timeout: float
    attempts: dict[str, ToolAttempt]


class Base(DeclarativeBase):
    pass


class RunRow(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    task: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String)
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    max_steps: Mapped[int] = mapped_column(Integer)
    tool_timeout: Mapped[float] = mapped_column()


class StepRow(Base):
    __tablename__ = "steps"

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    index: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_calls_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    assistant_message_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String, nullable=True)
    tool_name: Mapped[str | None] = mapped_column(String, nullable=True)
    output: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class AttemptRow(Base):
    __tablename__ = "tool_attempts"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    tool_call_id: Mapped[str] = mapped_column(String)
    tool_name: Mapped[str] = mapped_column(String)
    arguments_json: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String)
    output: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    idempotency_key: Mapped[str] = mapped_column(String, unique=True)


class RunStore:
    def __init__(self, url: str) -> None:
        self.url = url
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, connect_args=connect_args)
        if self.engine.dialect.name == "sqlite":

            @event.listens_for(self.engine, "connect")
            def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        Base.metadata.create_all(self.engine)

    def listen(self, listener: Callable[[dict[str, Any]], None]) -> None:
        self._listeners.append(listener)

    def create(
        self,
        run: Run,
        *,
        system_prompt: str | None,
        max_steps: int,
        tool_timeout: float,
    ) -> None:
        with Session(self.engine) as session:
            with session.begin():
                session.add(
                    RunRow(
                        id=run.id,
                        task=run.task,
                        status=run.status.value,
                        result=run.result,
                        error=run.error,
                        system_prompt=system_prompt,
                        max_steps=max_steps,
                        tool_timeout=tool_timeout,
                    )
                )
        self._emit("create", run.id)

    def save(self, run: Run) -> None:
        with Session(self.engine) as session:
            with session.begin():
                self._sync_run(session, run)
        self._emit("save", run.id)

    def begin_tool(self, run_id: str, call: ToolCall) -> ToolAttempt:
        """Commit the intent to run this call, before the tool function runs."""

        key = f"{run_id}:{call.id}"
        with Session(self.engine) as session:
            with session.begin():
                existing = session.scalar(select(AttemptRow).where(AttemptRow.idempotency_key == key))
                if existing is not None:
                    attempt = _attempt(existing)
                else:
                    row = AttemptRow(
                        id=uuid4().hex,
                        run_id=run_id,
                        tool_call_id=call.id,
                        tool_name=call.name,
                        arguments_json=json.dumps(call.arguments),
                        status=AttemptStatus.STARTED.value,
                        idempotency_key=key,
                    )
                    session.add(row)
                    session.flush()
                    attempt = _attempt(row)
        self._emit("begin_tool", run_id)
        return attempt

    def finish_tool(
        self,
        run: Run,
        attempt_id: str,
        status: AttemptStatus,
        output: str | None,
        error: str | None,
    ) -> None:
        """Commit the tool step and the outcome together."""

        with Session(self.engine) as session:
            with session.begin():
                self._sync_run(session, run)
                row = session.get(AttemptRow, attempt_id)
                if row is None:
                    raise KeyError(attempt_id)
                row.status = status.value
                row.output = output
                row.error = error
        self._emit("finish_tool", run.id)

    def mark_unknown(self, run: Run, attempt_id: str) -> None:
        with Session(self.engine) as session:
            with session.begin():
                self._sync_run(session, run)
                row = session.get(AttemptRow, attempt_id)
                if row is None:
                    raise KeyError(attempt_id)
                row.status = AttemptStatus.UNKNOWN.value
                row.error = run.error
        self._emit("mark_unknown", run.id)

    def get_attempt(self, run_id: str, tool_call_id: str) -> ToolAttempt | None:
        with Session(self.engine) as session:
            row = session.scalar(
                select(AttemptRow).where(
                    AttemptRow.run_id == run_id,
                    AttemptRow.tool_call_id == tool_call_id,
                )
            )
            if row is None:
                return None
            return _attempt(row)

    def ids(self) -> list[str]:
        with Session(self.engine) as session:
            return list(session.scalars(select(RunRow.id)).all())

    def load(self, run_id: str) -> StoredRun:
        with Session(self.engine) as session:
            row = session.get(RunRow, run_id)
            if row is None:
                raise KeyError(run_id)
            steps = session.scalars(
                select(StepRow).where(StepRow.run_id == run_id).order_by(StepRow.index)
            ).all()
            attempts = session.scalars(select(AttemptRow).where(AttemptRow.run_id == run_id)).all()
            run = Run(
                id=row.id,
                task=row.task,
                status=RunStatus(row.status),
                steps=[_step(step) for step in steps],
                result=row.result,
                error=row.error,
            )
            return StoredRun(
                run=run,
                system_prompt=row.system_prompt,
                max_steps=row.max_steps,
                tool_timeout=row.tool_timeout,
                attempts={item.tool_call_id: _attempt(item) for item in attempts},
            )

    def _sync_run(self, session: Session, run: Run) -> None:
        row = session.get(RunRow, run.id)
        if row is None:
            raise KeyError(run.id)
        row.status = run.status.value
        row.result = run.result
        row.error = run.error
        existing = set(session.scalars(select(StepRow.index).where(StepRow.run_id == run.id)).all())
        for step in run.steps:
            if step.index not in existing:
                session.add(_step_row(run.id, step))

    def snapshot(self) -> dict[str, Any]:
        with Session(self.engine) as session:
            runs = session.scalars(select(RunRow)).all()
            steps = session.scalars(select(StepRow).order_by(StepRow.index)).all()
            attempts = session.scalars(select(AttemptRow)).all()
            return {
                "runs": [_run_public(row) for row in runs],
                "steps": [_step_public(row) for row in steps],
                "attempts": [_attempt_public(row) for row in attempts],
            }

    def _emit(self, source: str, run_id: str) -> None:
        if not self._listeners:
            return
        event = {"type": "checkpoint", "source": source, "run_id": run_id, **self.snapshot()}
        for listener in list(self._listeners):
            listener(event)


def sqlite_url(path: str | Path) -> str:
    return "sqlite:///" + Path(path).resolve().as_posix()


def _step_row(run_id: str, step: Step) -> StepRow:
    return StepRow(
        run_id=run_id,
        index=step.index,
        kind=step.kind.value,
        content=step.content,
        tool_calls_json=json.dumps([call.model_dump() for call in step.tool_calls]) if step.tool_calls else None,
        assistant_message_json=json.dumps(step.assistant_message) if step.assistant_message is not None else None,
        tool_call_id=step.tool_call_id,
        tool_name=step.tool_name,
        output=step.output,
        error=step.error,
    )


def _step(row: StepRow) -> Step:
    raw_calls = json.loads(row.tool_calls_json) if row.tool_calls_json else []
    return Step(
        index=row.index,
        kind=StepKind(row.kind),
        content=row.content,
        tool_calls=[ToolCall.model_validate(call) for call in raw_calls],
        assistant_message=json.loads(row.assistant_message_json) if row.assistant_message_json else None,
        tool_call_id=row.tool_call_id,
        tool_name=row.tool_name,
        output=row.output,
        error=row.error,
    )


def _attempt(row: AttemptRow) -> ToolAttempt:
    return ToolAttempt(
        id=row.id,
        tool_call_id=row.tool_call_id,
        tool_name=row.tool_name,
        arguments=json.loads(row.arguments_json),
        status=AttemptStatus(row.status),
        output=row.output,
        error=row.error,
        idempotency_key=row.idempotency_key,
    )


def _run_public(row: RunRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "task": row.task,
        "status": row.status,
        "result": row.result,
        "error": row.error,
    }


def _step_public(row: StepRow) -> dict[str, Any]:
    calls = json.loads(row.tool_calls_json) if row.tool_calls_json else []
    names = ", ".join(call.get("name", "") for call in calls if call.get("name"))
    return {
        "run_id": row.run_id,
        "index": row.index,
        "kind": row.kind,
        "content": row.content,
        "tool_name": row.tool_name or names or None,
        "tool_call_id": row.tool_call_id,
        "output": row.output,
        "error": row.error,
    }


def _attempt_public(row: AttemptRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "run_id": row.run_id,
        "tool_call_id": row.tool_call_id,
        "tool_name": row.tool_name,
        "arguments": json.loads(row.arguments_json),
        "status": row.status,
        "output": row.output,
        "error": row.error,
        "idempotency_key": row.idempotency_key,
    }

