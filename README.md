# Agent runtime lab

v0.1 runs one agent task in a single process.

A run moves `created → running → completed` or `failed`. Each turn asks the model what to do. Tool calls go through a registry that checks their arguments, then the results are sent back. The run stops when the model answers, the provider fails, or the step limit is reached. The history of that run is printed as JSON.

There is no database, queue, or UI in this version. Those start at v0.2.

## Run the file agent

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`.

```bash
uv run python examples/file_agent.py "list the files in the workspace"
```

Tools can list, read, and write files inside `workspace/`. A path outside that directory is rejected. The command prints the trace and exits `0` when the run completes.

## Tests

```bash
uv run pytest
```

The loop tests script model decisions, so they do not call OpenRouter.

## Next

v0.2 stores runs and steps in PostgreSQL and resumes a run after the process is killed. A recorded tool result is not executed again. External side effects that finished but were not recorded can still happen twice.
