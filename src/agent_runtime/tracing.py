"""Render the in-memory history stored on a run."""

import json
from typing import Any

from agent_runtime.models import Run, Step, StepKind


def format_trace(run: Run) -> str:
    payload = {
        "id": run.id,
        "task": run.task,
        "status": run.status.value,
        "result": run.result,
        "error": run.error,
        "steps": [_step_record(step) for step in run.steps],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _step_record(step: Step) -> dict[str, Any]:
    if step.kind is StepKind.MODEL:
        return {
            "index": step.index,
            "kind": step.kind.value,
            "content": step.content,
            "tool_calls": [
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in step.tool_calls
            ],
        }
    return {
        "index": step.index,
        "kind": step.kind.value,
        "tool_call_id": step.tool_call_id,
        "name": step.tool_name,
        "output": step.output,
        "error": step.error,
    }
