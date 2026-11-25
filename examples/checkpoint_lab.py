"""Chat on the left, database commits on the right.

Messages go to OpenRouter. A tool row appears only when the model calls a tool.

    uv run python examples/checkpoint_lab.py
"""

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent_runtime import Agent, OpenRouterProvider, ProviderError, RunStatus, RunStore, sqlite_url
from file_agent import SYSTEM_PROMPT, WORKSPACE, _load_env, build_file_tools

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "workspace" / "lab"
DATABASE = LAB / "lab.sqlite3"
PAGE = Path(__file__).with_name("checkpoint_lab.html")
HOST = "127.0.0.1"
PORT = 8765


class OutcomeNotRecorded(Exception):
    pass


class DropOutcome:
    """Intent commits. The outcome commit is the crash."""

    def __init__(self, store: RunStore) -> None:
        self._store = store

    def __getattr__(self, name: str):
        return getattr(self._store, name)

    def finish_tool(self, *args, **kwargs):
        raise OutcomeNotRecorded()


class Lab:
    def __init__(self) -> None:
        self.queues: list[queue.Queue] = []
        self.clients = threading.Lock()
        self.busy = threading.Lock()
        self.last_run_id: str | None = None
        self._seen_steps: set[tuple[str, int]] = set()
        self._seen_failures: set[str] = set()
        self.store = self._open_store()

    def _open_store(self) -> RunStore:
        LAB.mkdir(parents=True, exist_ok=True)
        store = RunStore(sqlite_url(DATABASE))
        store.listen(self._on_checkpoint)
        for step in store.snapshot()["steps"]:
            self._seen_steps.add((step["run_id"], step["index"]))
        return store

    def publish(self, event: dict) -> None:
        with self.clients:
            clients = list(self.queues)
        for client in clients:
            client.put(event)

    def _on_checkpoint(self, event: dict) -> None:
        self.last_run_id = event["run_id"]
        for step in event.get("steps") or []:
            if step["run_id"] != event["run_id"]:
                continue
            key = (step["run_id"], step["index"])
            if key in self._seen_steps:
                continue
            self._seen_steps.add(key)
            self._narrate(step)
        run = next((row for row in event.get("runs") or [] if row["id"] == event["run_id"]), None)
        if run and run["status"] == "failed" and run.get("error") and event["run_id"] not in self._seen_failures:
            self._seen_failures.add(event["run_id"])
            already = any(
                step.get("error") == run["error"]
                for step in event.get("steps") or []
                if step["run_id"] == event["run_id"]
            )
            if not already:
                self.publish(
                    {"type": "chat", "role": "assistant", "text": run["error"], "run_id": event["run_id"]}
                )
        event["files"] = _workspace_text()
        self.publish(event)

    def _narrate(self, step: dict) -> None:
        run_id = step["run_id"]
        if step["kind"] == "model":
            if step.get("content"):
                self.publish({"type": "chat", "role": "assistant", "text": step["content"], "run_id": run_id})
            if step.get("tool_name"):
                self.publish(
                    {
                        "type": "chat",
                        "role": "assistant",
                        "text": f"Calling {step['tool_name']}.",
                        "run_id": run_id,
                    }
                )
        elif step["kind"] == "tool":
            detail = step.get("error") or step.get("output") or ""
            self.publish(
                {
                    "type": "chat",
                    "role": "system",
                    "text": f"{step.get('tool_name')}: {detail}",
                    "run_id": run_id,
                }
            )

    def subscribe(self) -> queue.Queue:
        client: queue.Queue = queue.Queue()
        snapshot = {
            "type": "checkpoint",
            "source": "load",
            "run_id": self.last_run_id,
            "files": _workspace_text(),
            **self.store.snapshot(),
        }
        with self.clients:
            self.queues.append(client)
        client.put(snapshot)
        return client

    def unsubscribe(self, client: queue.Queue) -> None:
        with self.clients:
            if client in self.queues:
                self.queues.remove(client)

    def start_chat(self, text: str, crash: bool) -> bool:
        if not self.busy.acquire(blocking=False):
            return False
        threading.Thread(target=self._chat, args=(text, crash), daemon=True).start()
        return True

    def start_resume(self) -> bool:
        if not self.busy.acquire(blocking=False):
            return False
        threading.Thread(target=self._resume, daemon=True).start()
        return True

    def reset(self) -> None:
        if not self.busy.acquire(blocking=False):
            raise RuntimeError("busy")
        try:
            self.store.engine.dispose()
            for path in LAB.glob("lab.sqlite3*"):
                path.unlink(missing_ok=True)
            self.last_run_id = None
            self._seen_steps.clear()
            self._seen_failures.clear()
            self.store = self._open_store()
            self.publish(
                {
                    "type": "reset",
                    "files": _workspace_text(),
                    "runs": [],
                    "steps": [],
                    "attempts": [],
                }
            )
        finally:
            self.busy.release()

    def _chat(self, text: str, crash: bool) -> None:
        try:
            self.publish({"type": "chat", "role": "user", "text": text, "run_id": None})
            try:
                provider = _provider()
            except ProviderError as exc:
                self.publish({"type": "chat", "role": "system", "text": str(exc)})
                return
            store = DropOutcome(self.store) if crash else self.store
            agent = Agent(
                provider,
                _WatchingTools(build_file_tools(WORKSPACE), self._publish_files),
                system_prompt=SYSTEM_PROMPT,
                store=store,
            )
            try:
                agent.run(text)
            except OutcomeNotRecorded:
                self.publish(
                    {
                        "type": "chat",
                        "role": "system",
                        "text": (
                            "Process killed after the tool ran. "
                            "The attempt row is still started. Resume does not run that tool again."
                        ),
                    }
                )
        finally:
            self.publish({"type": "idle"})
            self.busy.release()

    def _resume(self) -> None:
        try:
            if self.last_run_id is None:
                self.publish({"type": "chat", "role": "system", "text": "Nothing to resume yet."})
                return
            stored = self.store.load(self.last_run_id)
            if stored.run.status is not RunStatus.RUNNING:
                self.publish(
                    {
                        "type": "chat",
                        "role": "system",
                        "text": "This run already finished. Its recorded tools are not run again.",
                    }
                )
                return
            try:
                provider = _provider()
            except ProviderError as exc:
                self.publish({"type": "chat", "role": "system", "text": str(exc)})
                return
            Agent(
                provider,
                _WatchingTools(build_file_tools(WORKSPACE), self._publish_files),
                system_prompt=SYSTEM_PROMPT,
                store=self.store,
            ).resume(self.last_run_id)
        finally:
            self.publish({"type": "idle"})
            self.busy.release()

    def _publish_files(self) -> None:
        self.publish({"type": "file", "files": _workspace_text()})


class _WatchingTools:
    def __init__(self, registry, after) -> None:
        self._registry = registry
        self._after = after

    def schemas(self):
        return self._registry.schemas()

    def execute(self, call):
        result = self._registry.execute(call)
        self._after()
        return result


def _provider() -> OpenRouterProvider:
    _load_env(ROOT / ".env")
    return OpenRouterProvider.from_env()


def _workspace_text() -> str:
    root = WORKSPACE.resolve()
    if not root.is_dir():
        return "empty"
    chunks: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "sqlite3" in path.name:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            text = "(binary)"
        chunks.append(f"{path.relative_to(root).as_posix()}\n{text[:400]}")
    return "\n\n".join(chunks) if chunks else "empty"


LAB_APP = Lab()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/" or self.path.startswith("/chat/"):
            body = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/events":
            self._events()
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "bad json"})
            return
        if self.path == "/api/chat":
            text = str(payload.get("text", "")).strip()
            if not text:
                self._json(400, {"error": "message required"})
                return
            if not LAB_APP.start_chat(text, bool(payload.get("crash"))):
                self._json(409, {"error": "busy"})
                return
            self._json(202, {"ok": True})
            return
        if self.path == "/api/resume":
            if not LAB_APP.start_resume():
                self._json(409, {"error": "busy"})
                return
            self._json(202, {"ok": True})
            return
        if self.path == "/api/reset":
            try:
                LAB_APP.reset()
            except RuntimeError:
                self._json(409, {"error": "busy"})
                return
            self._json(200, {"ok": True})
            return
        self.send_error(404)

    def _events(self) -> None:
        client = LAB_APP.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while True:
                event = client.get()
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            LAB_APP.unsubscribe(client)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Checkpoint lab at http://{HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
