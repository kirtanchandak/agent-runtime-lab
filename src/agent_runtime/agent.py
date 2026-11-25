"""Connect a provider and a tool registry into one run.

Each turn asks the model what to do. A text answer completes the run.
Tool calls are executed, recorded, and sent back. The loop stops when
the model answers, the provider fails, or the step limit is reached.

With a store, each checkpoint is committed on its own. Resume loads the
last commit. A tool whose outcome was saved is not run again. A tool
that was started but whose outcome was not saved is not run again either:
the run fails and leaves the attempt for reconciliation.
"""

import threading
from typing import Any

from agent_runtime.models import Run, RunStatus, Step, StepKind, ToolCall
from agent_runtime.providers.base import ModelProvider, ProviderError
from agent_runtime.store import AttemptStatus, RunStore, ToolAttempt
from agent_runtime.tools import ToolRegistry, ToolResult


class Agent:
    def __init__(
        self,
        provider: ModelProvider,
        tools: ToolRegistry,
        *,
        max_steps: int = 8,
        tool_timeout: float = 30,
        system_prompt: str | None = None,
        store: RunStore | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if tool_timeout <= 0:
            raise ValueError("tool_timeout must be positive")
        self._provider = provider
        self._tools = tools
        self._max_steps = max_steps
        self._tool_timeout = tool_timeout
        self._system_prompt = system_prompt
        self._store = store

    def run(self, task: str) -> Run:
        run = Run(task=task)
        run.start()
        if self._store is not None:
            self._store.create(
                run,
                system_prompt=self._system_prompt,
                max_steps=self._max_steps,
                tool_timeout=self._tool_timeout,
            )
        return self._loop(run, self._messages_from(run, self._system_prompt))

    def resume(self, run_id: str) -> Run:
        if self._store is None:
            raise RuntimeError("resume requires a store")
        stored = self._store.load(run_id)
        self._max_steps = stored.max_steps
        self._tool_timeout = stored.tool_timeout
        if stored.run.status is not RunStatus.RUNNING:
            return stored.run
        return self._loop(stored.run, self._messages_from(stored.run, stored.system_prompt))

    def _loop(self, run: Run, messages: list[dict[str, Any]]) -> Run:
        while run.status is RunStatus.RUNNING:
            pending = self._pending_tool_calls(run)
            if pending:
                if self._out_of_steps(run):
                    return run
                self._resolve_tool(run, messages, pending[0])
                continue
            if self._out_of_steps(run):
                return run
            try:
                decision = self._provider.complete(messages, self._tools.schemas())
            except ProviderError as exc:
                run.fail(str(exc))
                self._save(run)
                return run

            step = Step(
                index=len(run.steps),
                kind=StepKind.MODEL,
                content=decision.content,
                tool_calls=list(decision.tool_calls),
                assistant_message=decision.assistant_message,
            )
            run.add_step(step)
            if not decision.tool_calls:
                run.complete(decision.content or "")
            self._save(run)
            if not decision.tool_calls:
                return run
            messages.append(decision.assistant_message)
        return run

    def _resolve_tool(self, run: Run, messages: list[dict[str, Any]], call: ToolCall) -> None:
        attempt: ToolAttempt | None = None
        if self._store is not None:
            attempt = self._store.get_attempt(run.id, call.id)
            if attempt is not None and attempt.status is AttemptStatus.STARTED:
                self._fail_unknown(run, call, attempt)
                return
            if attempt is not None and attempt.status in (
                AttemptStatus.COMPLETED,
                AttemptStatus.FAILED,
                AttemptStatus.UNKNOWN,
            ):
                self._record_tool(
                    run,
                    messages,
                    call,
                    ToolResult(output=attempt.output, error=attempt.error),
                    attempt,
                    replayed=True,
                )
                return
            attempt = self._store.begin_tool(run.id, call)

        result = _execute_with_timeout(self._tools, call, self._tool_timeout)
        self._record_tool(run, messages, call, result, attempt, replayed=False)

    def _record_tool(
        self,
        run: Run,
        messages: list[dict[str, Any]],
        call: ToolCall,
        result: ToolResult,
        attempt: ToolAttempt | None,
        *,
        replayed: bool,
    ) -> None:
        step = Step(
            index=len(run.steps),
            kind=StepKind.TOOL,
            tool_call_id=call.id,
            tool_name=call.name,
            output=result.output,
            error=result.error,
        )
        run.add_step(step)
        messages.append(_tool_message(call, result))
        if self._store is None or attempt is None or replayed:
            if self._store is not None and replayed:
                self._save(run)
            return
        status = AttemptStatus.COMPLETED if result.ok else AttemptStatus.FAILED
        self._store.finish_tool(run, attempt.id, status, result.output, result.error)

    def _fail_unknown(self, run: Run, call: ToolCall, attempt: ToolAttempt) -> None:
        message = (
            f"outcome unknown for {call.name}; it may have already run. "
            "Reconcile before retrying."
        )
        run.add_step(
            Step(
                index=len(run.steps),
                kind=StepKind.TOOL,
                tool_call_id=call.id,
                tool_name=call.name,
                error=message,
            )
        )
        run.fail(message)
        if self._store is not None:
            self._store.mark_unknown(run, attempt.id)

    def _pending_tool_calls(self, run: Run) -> list[ToolCall]:
        last_model_index: int | None = None
        for step in run.steps:
            if step.kind is StepKind.MODEL:
                last_model_index = step.index
        if last_model_index is None:
            return []
        model_step = run.steps[last_model_index]
        done = {
            step.tool_call_id
            for step in run.steps
            if step.kind is StepKind.TOOL and step.index > last_model_index
        }
        return [call for call in model_step.tool_calls if call.id not in done]

    def _messages_from(self, run: Run, system_prompt: str | None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": run.task})
        index = 0
        while index < len(run.steps):
            step = run.steps[index]
            if step.kind is StepKind.MODEL and step.tool_calls:
                if step.assistant_message is None:
                    raise RuntimeError(f"model step {step.index} is missing assistant_message")
                messages.append(step.assistant_message)
                index += 1
                while index < len(run.steps) and run.steps[index].kind is StepKind.TOOL:
                    tool_step = run.steps[index]
                    content = tool_step.output if tool_step.error is None else f"error: {tool_step.error}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_step.tool_call_id or "",
                            "content": content or "",
                        }
                    )
                    index += 1
                continue
            index += 1
        return messages

    def _out_of_steps(self, run: Run) -> bool:
        if len(run.steps) < self._max_steps:
            return False
        run.fail(f"exceeded max steps ({self._max_steps})")
        self._save(run)
        return True

    def _save(self, run: Run) -> None:
        if self._store is not None:
            self._store.save(run)


def _tool_message(call: ToolCall, result: ToolResult) -> dict[str, str]:
    content = result.output if result.ok else f"error: {result.error}"
    return {"role": "tool", "tool_call_id": call.id, "content": content or ""}


def _execute_with_timeout(registry: ToolRegistry, call: ToolCall, timeout: float) -> ToolResult:
    """Wait up to `timeout` seconds for one tool call.

    On timeout the thread is left running. Python cannot force a thread
    to stop, so a slow tool may finish after the run has moved on.
    """

    outcome: dict[str, ToolResult] = {}

    def target() -> None:
        outcome["result"] = registry.execute(call)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive() or "result" not in outcome:
        return ToolResult(error=f"tool timed out after {timeout}s")
    return outcome["result"]
