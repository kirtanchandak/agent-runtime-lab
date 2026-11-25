# Agent runtime lab

v0.1 runs one agent task in a single process.

A run moves `created → running → completed` or `failed`. Each turn asks the model what to do. Tool calls go through a registry that checks their arguments, then the results are sent back. The run stops when the model answers, the provider fails, or the step limit is reached. The history of that run is printed as JSON.

There is no queue or UI in this version.

## Run the file agent

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`.

```bash
uv run python examples/file_agent.py "list the files in the workspace"
```

Tools can list, read, and write files inside `workspace/`. A path outside that directory is rejected. The command prints the trace and exits `0` when the run completes.

Runs are written to `runtime.sqlite3`. The trace includes the run id. If the process dies, resume that id:

```bash
uv run python examples/file_agent.py --resume <run-id>
```

A tool whose result was committed is not executed again. If the process dies after the tool has started and before its result is committed, resume does not run it either. The run fails with `outcome unknown`. The file may already have been written. Look at that effect yourself before starting a new run. A later model call with a new tool-call id can still write the file again.

## Watch the checkpoints

```bash
uv run python examples/checkpoint_lab.py
```

Open http://127.0.0.1:8765. The left side is a real OpenRouter chat. The right side is the database. A green flash is a commit. A greeting does not write a file. A tool row appears when the model calls a tool. The checkbox kills the process after that tool runs and before its result is saved. **Resume** then leaves the attempt `unknown`.

## Tests

```bash
uv run pytest
```

The loop tests script model decisions, so they do not call OpenRouter.

## Next

The checkpoint rule above is running on SQLite. v0.2 still needs the same tables on PostgreSQL, and tools that can accept the stored idempotency key so a supported write can be retried safely. Unsupported tools stay on manual reconciliation.
