# kiln-agents

A native Python port of [kiln](https://github.com/abhishekvardanbotta/kiln) — build and orchestrate AI
agents. Same primitives as the TypeScript/npm package (`@abhishekvardanbotta/kiln`), reimplemented
natively in Python: no Node runtime involved.

```bash
pip install kiln-agents
```

## What's ported

- **Unified providers** — OpenAI, Groq, and Ollama via one shared `httpx`-based adapter (same design
  as the TS version: one class, three providers, swap a base URL + env var name). Gemini and Claude are
  declared but not implemented yet, matching the TS package's current status.
- **`ctx.ai.run()`** — the tool-calling loop: call the model, execute any requested tool calls, feed
  the results back, repeat, up to `max_steps`.
- **`ctx.ai.object()`** — self-correcting structured output. Takes a **pydantic** model as the schema
  (Python's equivalent of the TS version's zod schema), writes the schema instruction into the prompt,
  validates the response, and retries with the exact validation error fed back on failure.
- **`connect_mcp()`** — real Model Context Protocol support via the official `mcp` Python SDK. Any MCP
  server's tools become normal kiln `Tool` objects.
- **`orchestrate()`** — the DAG orchestrator: `depends_on`, per-step retries, a concurrency pool
  (`asyncio.Semaphore`-based), and cascading `"skipped"` status for steps whose dependency failed —
  ported line-for-line from the TS `Orchestrator`.
- **`Team` / `ctx.team.ask()`** — a named roster of agents that call each other directly at run time, on
  top of a shared blackboard (`ctx.team.get`/`set`), for collaboration whose shape isn't known ahead of
  time (unlike `orchestrate()`'s pre-wired DAG). See "Teams" below.
- **Full structured runtime traces** — every `ctx.ai.*` call (`LLMCallCompleted`: exact provider, model,
  messages sent, response or error, duration), every tool call and its real result or error
  (`ToolCallCompleted`), every `ai.object()` validation attempt (`ObjectAttemptSucceeded`/
  `ObjectAttemptFailed`, with the raw output that failed), and a real traceback on `AgentFailed` — not
  just the agent's final output. `TraceCapture` (see "Traces" below) collects one run's timeline; each
  `TeamAskRecord` carries the asked teammate's own trace too.
- Scoped `ctx.memory` — private per run by default (two concurrent runs never see each other's writes,
  even using the same key), shareable across runs on purpose via `session_id`.

## What's *not* ported (by design, this release)

The `kiln` **CLI** — `kiln init`/`pack`/`install`/`serve`/`logs`, the `.agent` package format, and
`~/.kiln` project/registry management — is TypeScript-only for now. This package is the **embeddable
SDK**: you `define_agent(...)` directly in your own Python code and run it in-process, the same pattern
the TS SDK uses when embedded in a Node backend (no CLI, no separate project folder). If you need a
long-running HTTP surface for a non-Python caller, run the TS package's `kiln serve` instead — this
package is for using kiln *from* Python, not for running the CLI.

Also TS-only for now: `mountAgentRoutes()` (mounting agents/teams/flows into your own HTTP server with a
live dashboard) and `kiln mcp` (the MCP server that turns a *running* kiln app into an AI-native
debugging interface for Claude Code). This release ports the runtime primitives those are built on —
`Team` and the full trace system — so a Python HTTP/MCP layer over them is a real, well-scoped next step,
not a redesign.

## Quickstart

```python
import asyncio
from pydantic import BaseModel
from kiln_agents import create_kiln, define_agent

class Forecast(BaseModel):
    city: str
    temperature_c: float
    summary: str

async def forecast_execute(ctx):
    city = ctx.input["city"]
    return await ctx.ai.object(
        messages=[{"role": "user", "content": f"Give a short weather forecast for {city}."}],
        schema=Forecast,
    )

forecaster = define_agent(
    name="forecaster",
    provider="groq",           # "openai" | "groq" | "ollama"
    model="openai/gpt-oss-120b",
    execute=forecast_execute,
)

async def main():
    kiln = create_kiln()
    result = await kiln.run_inline_agent(forecaster, {"city": "Tokyo"})
    print(result.output)

asyncio.run(main())
```

## Orchestrating multiple agents

```python
from kiln_agents import create_kiln, PipelineStep, PipelineRunContext

kiln = create_kiln()
result = await kiln.orchestrate([
    PipelineStep(id="forecaster", agent=forecaster, input={"city": "Paris"}),
    PipelineStep(
        id="advisor",
        agent=advisor,
        depends_on=["forecaster"],
        retries=2,
        input=lambda ctx: {"forecast": ctx.outputs["forecaster"]},
    ),
])

print(result.outputs["advisor"])
print([step.status for step in result.steps])  # "succeeded" | "failed" | "skipped"
```

## Teams — dynamic multi-agent collaboration

Use `orchestrate()` when the data flow is known ahead of time. Use a team when it isn't — a support
ticket that might need one specialist or three, decided at run time by whichever agent is running.

```python
from kiln_agents import create_kiln, define_agent
from pydantic import BaseModel

class Resolution(BaseModel):
    answer: str

kiln = create_kiln()

async def billing_execute(ctx):
    return await ctx.ai.object(messages=[{"role": "user", "content": ctx.input["question"]}], schema=Resolution)

billing = define_agent(name="billing-specialist", provider="groq", model="openai/gpt-oss-120b", execute=billing_execute)

async def coordinator_execute(ctx):
    # ctx.team.ask() returns the teammate's raw output, or raises if it failed.
    result = await ctx.team.ask("billing-specialist", {"question": ctx.input["question"]})
    return result

coordinator = define_agent(name="coordinator", provider="groq", model="openai/gpt-oss-120b", execute=coordinator_execute)

team = kiln.team([billing, coordinator])
result = await team.run("coordinator", {"question": "I was charged twice."})
# result.asks: [TeamAskRecord(from_agent="coordinator", to_agent="billing-specialist", output=..., trace=[...])]
```

`ctx.team` is also a shared blackboard (`await ctx.team.get(key)` / `await ctx.team.set(key, value)`),
visible to every member of the same run — separate from `ctx.memory`, which stays private per run.
`ctx.team.ask()` raises `KilnError` if called outside a `team.run()` (there's no roster to dispatch to).

## Traces — debugging what actually happened

Every run's structured timeline is one `TraceCapture` away — subscribe before the run starts (same
"subscribe first" rule the `Team` internals follow), then read it back once the run settles:

```python
from kiln_agents.trace import TraceCapture
import uuid

run_id = str(uuid.uuid4())
trace = TraceCapture(kiln.events, run_id)
result = await kiln.run_inline_agent(some_agent, {"city": "Tokyo"}, run_id=run_id)
for entry in trace.stop():
    print(entry.type, entry.data)
# AgentStarted {...}
# LLMCallCompleted {'kind': 'object', 'provider': 'groq', 'model': 'openai/gpt-oss-120b', 'messages': [...], 'response': {...}, 'duration_ms': 812}
# ObjectAttemptSucceeded {'attempt': 1, 'output': {...}}
# AgentCompleted {...}
```

Every `TeamAskRecord` (see "Teams" above) carries this same trace for that one call, so a team run's
`result.asks[i].trace` shows exactly what that teammate was sent and what it got back — the difference
between "a specialist gave a wrong answer" and "here's the exact prompt and model response that produced it".

## Tools

```python
from pydantic import BaseModel
from kiln_agents import define_tool

class GetForecastInput(BaseModel):
    city: str

async def get_forecast(input: GetForecastInput, ctx):
    return {"city": input.city, "temperature_c": 21.0, "summary": "Clear skies."}

get_forecast_tool = define_tool(
    name="get_forecast",
    description="Fetch a real weather forecast for a city.",
    parameters=GetForecastInput,
    execute=get_forecast,
)

# Pass tools=[get_forecast_tool] into define_agent(...), then either:
#   await ctx.tools["get_forecast"]({"city": "Tokyo"})   # call it directly
#   await ctx.ai.run(messages=[...])                      # or let the model decide to call it
```

## MCP

```python
from kiln_agents import connect_mcp

mcp = await connect_mcp("npx", args=["-y", "@some/mcp-server"])
# mcp.tools is list[Tool] — pass straight into define_agent(tools=[...])
...
await mcp.close()  # always close when done
```

## Provider setup

| provider | env var | notes |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | base URL `https://api.openai.com/v1` |
| `groq` | `GROQ_API_KEY` | base URL `https://api.groq.com/openai/v1` |
| `ollama` | `OLLAMA_HOST` (optional) | defaults to `localhost:11434`; give it bare `host:port` |
| `gemini`, `claude` | — | **stub only**, raises `KilnError("<name> is not implemented yet.")` |

## License

MIT
