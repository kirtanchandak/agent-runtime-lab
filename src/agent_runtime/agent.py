"""Connect a provider and a tool registry into one run.

Each turn asks the model what to do. A text answer completes the run.
Tool calls are executed, recorded, and sent back. The loop stops when
the model answers, the provider fails, or the step limit is reached.
"""

import threading
from typing import Any

from agent_runtime.models import Run, Step, StepKind, ToolCall
from agent_runtime.providers.base import ModelProvider, ProviderError
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

    def run(self, task: str) -> Run:
        run = Run(task=task)
        run.start()
        messages: list[dict[str, Any]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": task})

        while True:
            if self._out_of_steps(run):
                return run
            try:
                decision = self._provider.complete(messages, self._tools.schemas())
            except ProviderError as exc:
                run.fail(str(exc))
                return run

            run.add_step(
                Step(
                    index=len(run.steps),
                    kind=StepKind.MODEL,
                    content=decision.content,
                    tool_calls=list(decision.tool_calls),
                )
            )
            if not decision.tool_calls:
                run.complete(decision.content or "")
                return run

            messages.append(decision.assistant_message)
            for call in decision.tool_calls:
                if self._out_of_steps(run):
                    return run
                result = _execute_with_timeout(self._tools, call, self._tool_timeout)
                run.add_step(
                    Step(
                        index=len(run.steps),
                        kind=StepKind.TOOL,
                        tool_call_id=call.id,
                        tool_name=call.name,
                        output=result.output,
                        error=result.error,
                    )
                )
                messages.append(_tool_message(call, result))

    def _out_of_steps(self, run: Run) -> bool:
        if len(run.steps) < self._max_steps:
            return False
        run.fail(f"exceeded max steps ({self._max_steps})")
        return True


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
