"""File agent: list, read, and write files inside one workspace directory.

Usage:
    uv run python examples/file_agent.py "list the files and summarize them"
"""

import json
import os
import sys
from pathlib import Path

from agent_runtime import (
    Agent,
    OpenRouterProvider,
    ProviderError,
    RunStatus,
    Tool,
    ToolArgs,
    ToolRegistry,
    format_trace,
)

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "workspace"
SYSTEM_PROMPT = (
    "You are a file assistant. Use tools to inspect and change files in the workspace. "
    "Paths are relative to that directory. When you know the answer, reply with it and stop calling tools."
)


class NoArgs(ToolArgs):
    pass


class ReadArgs(ToolArgs):
    path: str


class WriteArgs(ToolArgs):
    path: str
    content: str


def build_file_tools(workspace: Path) -> ToolRegistry:
    root = workspace.resolve()
    root.mkdir(parents=True, exist_ok=True)
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="list_files",
            description="List files in the workspace. Paths are relative to the workspace root.",
            args_model=NoArgs,
            fn=lambda _: _list_files(root),
        )
    )
    registry.register(
        Tool(
            name="read_file",
            description="Read a text file in the workspace.",
            args_model=ReadArgs,
            fn=lambda args: _read_file(root, args.path),
        )
    )
    registry.register(
        Tool(
            name="write_file",
            description="Create or replace a text file in the workspace.",
            args_model=WriteArgs,
            fn=lambda args: _write_file(root, args.path, args.content),
        )
    )
    return registry


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: uv run python examples/file_agent.py <task>", file=sys.stderr)
        return 2
    _load_env(ROOT / ".env")
    try:
        provider = OpenRouterProvider.from_env()
    except ProviderError as exc:
        print(exc, file=sys.stderr)
        return 1
    agent = Agent(
        provider=provider,
        tools=build_file_tools(WORKSPACE),
        system_prompt=SYSTEM_PROMPT,
    )
    run = agent.run(" ".join(args))
    print(format_trace(run))
    return 0 if run.status is RunStatus.COMPLETED else 1


def _list_files(root: Path) -> str:
    files = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    return json.dumps(files)


def _read_file(root: Path, path: str) -> str:
    target = _inside(root, path)
    if not target.is_file():
        raise FileNotFoundError(path)
    return target.read_text()


def _write_file(root: Path, path: str, content: str) -> str:
    target = _inside(root, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"wrote {target.relative_to(root).as_posix()}"


def _inside(root: Path, raw: str) -> Path:
    if not raw.strip():
        raise ValueError("path is required")
    root = root.resolve()
    target = (root / raw).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError(f"path escapes the workspace: {raw}")
    return target


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


if __name__ == "__main__":
    raise SystemExit(main())
