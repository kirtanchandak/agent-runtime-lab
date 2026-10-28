# Agent Runtime Lab — Development Roadmap

## Vision

Build an open-source, developer-focused platform for **running, recovering, observing, and evaluating AI agents**. This is a learning and portfolio project, not an attempt to replace Temporal or LangGraph.

**Final user experience:** A developer registers an agent and tools, starts a run via CLI/API/SDK, watches execution in a dashboard, inspects tool calls and costs, approves sensitive actions, resumes interrupted runs, and compares agent versions on evaluation datasets.

**Example application:** A GitHub issue investigation agent that reads repository files and issues, runs approved read-only checks, and proposes a fix. Use the same example throughout the versions to expose real engineering problems.

## Stack

- **Runtime:** Python 3.12+, Pydantic, asyncio, pytest
- **API:** FastAPI
- **Persistence:** PostgreSQL, SQLAlchemy, Alembic
- **Workers:** PostgreSQL-backed queue initially; consider Redis only when justified
- **Observability:** Structured JSON logs, OpenTelemetry
- **Frontend:** Next.js, TypeScript, Tailwind/shadcn
- **Packaging:** uv, Docker Compose, GitHub Actions
- **Models:** Provider-neutral adapter; start with one provider

## Version plan

### v0.1 — Core agent runtime

**Build**
- CLI to submit a task and print a final result
- Agent loop: model decision → tool invocation → tool result → next decision
- Typed tool registry with argument validation
- Provider-neutral model interface
- Run states (`CREATED`, `RUNNING`, `COMPLETED`, `FAILED`)
- In-memory step history, max-step limit, tool timeout
- Fake model and fake tools for deterministic tests

**Learn:** State machines, async Python, typed interfaces, dependency injection, testability.

**Done when:** One task executes several tools, exits predictably, prints a structured trace, and passes unit tests.

**Scope:** No database, UI, queues, or multi-agent features. Since the basics are familiar, aim to finish quickly.

### v0.2 — Durable execution and crash recovery

**Build**
- PostgreSQL tables for runs, steps, tool attempts, and checkpoints
- Transactional state transitions and recovery on restart
- Durable record of tool intents and outcomes
- Explicit handling for the ambiguous window where a tool succeeds but the result is not recorded
- Idempotency keys for supported external operations; manual reconciliation for unsupported ones
- Failure-injection tests at checkpoint boundaries

**Learn:** Transactions, consistency, idempotency, recovery, execution guarantees.

**Done when:** Kill the process midway, restart, and resume without redoing *recorded* successful steps. Clearly document where external side effects may still be duplicated.

### v0.3 — Background workers and concurrency

**Build**
- FastAPI endpoints to create, inspect, and list runs
- PostgreSQL-backed job queue with atomic claiming
- Worker leases, heartbeats, expiry, and fencing/version checks
- Multiple workers and per-agent concurrency limits
- Recovery when a worker disappears
- Concurrency and race-condition integration tests

**Learn:** Distributed coordination, locks, leases, at-least-once processing, backpressure.

**Done when:** Multiple workers process jobs safely; expired jobs are reclaimed; stale workers cannot commit updates after losing ownership.

### v0.4 — Reliability, controls, and human approval

**Build**
- Retry policies with exponential backoff and jitter
- Retryable vs non-retryable error classification
- Cancellation and timeout handling
- Tool allowlists, permission scopes, approval gates
- Persisted `WAITING_FOR_APPROVAL` state with approve/reject API
- Redaction of secrets and sensitive tool outputs

**Learn:** Fault tolerance, authorization, safe retries, compensation, threat modeling.

**Done when:** An approval-required tool cannot execute without authorization; retries are bounded; cancellation prevents new work from being scheduled.

### v0.5 — Tracing and observability

**Build**
- Structured run/step/tool logs with correlation IDs
- OpenTelemetry traces and spans
- Token usage, approximate cost, duration, error categories
- Queryable execution timeline and failure summaries
- Basic operational metrics (queue depth, retries, latency)

**Learn:** Instrumentation, distributed tracing, debugging, operational metrics.

**Done when:** A failed run can be diagnosed from its trace, including where time and tokens were spent.

### v0.6 — Evaluation engine

**Build**
- Versioned evaluation datasets and expected outcomes
- Deterministic tool mocks and replayable fixtures
- Task success and tool-selection metrics
- Comparison of prompts/models/runtime versions
- Regression checks in CI with explicit thresholds
- Reports covering success rate, latency, cost, and failure modes

**Learn:** Experimental design, AI evaluation, reproducibility, regression analysis.

**Done when:** Two versions run on the same dataset and generate a reproducible comparison report. Start with 20–30 high-quality cases; grow toward 100+.

### v0.7 — Developer dashboard

**Build**
- Next.js UI for creating runs and viewing status
- Live progress using SSE (WebSockets only if needed)
- Step-by-step trace viewer with inputs/outputs and errors
- Approval/rejection interface
- Evaluation results and comparison views

**Learn:** Full-stack design, streaming, API contracts, UI state management, developer experience.

**Done when:** A developer can start, monitor, approve, and debug a run without using the terminal.

### v0.8 — SDK, MCP, and extensibility

**Build**
- Installable Python SDK with documented API
- Tool/plugin interface and MCP tool adapter
- Authentication for API access
- Configurable model providers
- Versioned API contracts, examples, and integration tests

**Learn:** Library design, packaging, compatibility, extensibility, API security.

**Done when:** Someone else can install the SDK, register their tools, and run an example agent using the documentation.

### v0.9 — Performance and security hardening

**Build**
- Load tests for concurrent runs and worker throughput
- Profiling and bottleneck analysis
- Rate limits, resource quotas, and secret handling
- Failure injection: DB disconnect, worker kill, API timeout, duplicate delivery
- Documented operational limits and known failure modes

**Learn:** Capacity planning, profiling, resilience testing, secure operations.

**Done when:** You can publish reproducible performance results and a failure-mode matrix, not vague claims of scale.

### v1.0 — Public release

**Build**
- Reproducible Docker Compose setup and hosted demo if feasible
- CI/CD, linting, type checks, integration tests
- Architecture diagram, quickstart, example agent, API docs
- Benchmarks, design decisions, known limitations, contribution guide
- Technical write-up: trade-offs versus LangGraph and Temporal

**Learn:** Release engineering, deployment, documentation, open-source maintenance, technical communication.

**Done when:** A new developer can clone, start, test, and extend the project without your help.

## Suggested timeline (8–12 hours/week)

| Period | Versions | Primary focus |
|---|---|---|
| Month 1 | v0.1–v0.2 | Runtime and durability |
| Month 2 | v0.3–v0.4 | Workers and reliability |
| Month 3 | v0.5–v0.6 | Observability and evaluations |
| Month 4 | v0.7–v0.8 | Dashboard and SDK |
| Month 5 | v0.9–v1.0 | Hardening and public release |

These are estimates, not deadlines. Recovery and concurrency are likely to require iteration.

## Build principles

1. **Own the hard parts:** Implement runtime state, persistence, recovery, and worker coordination yourself. Use AI for boilerplate, test generation, review, and UI scaffolding.
2. **Build vertically:** Keep one real example agent working through every version.
3. **Test failure paths:** A successful demo is not enough. Include crash and duplicate-delivery tests.
4. **Avoid premature infrastructure:** Start with one worker and PostgreSQL; add Redis or orchestration services only if measurements justify them.
5. **Make guarantees explicit:** Durable execution is not automatically exactly-once execution of external side effects.
6. **Keep the README current:** Each version gets a short demo, tests, and a design-decision note.

## Recommended reference systems

- [LangGraph](https://github.com/langchain-ai/langgraph) — stateful agent orchestration, checkpoints, human-in-the-loop
- [Temporal workflows](https://docs.temporal.io/workflows) — durable execution, history replay, failure recovery
- [Temporal tasks](https://docs.temporal.io/tasks) — workers and task queues

## Immediate action: v0.1

- [x] Initialize Python project and pytest
- [x] Define run states and typed tool contracts
- [x] Implement provider interface and one adapter
- [x] Implement loop with limits and timeouts
- [x] Add fake model/tools and unit tests
- [x] Add one working example and structured trace output
- [x] Document what comes next
- [ ] Tag `v0.1.0` after this version is committed
