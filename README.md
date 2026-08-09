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

## What's *not* ported (by design, this release)

The `kiln` **CLI** — `kiln init`/`pack`/`install`/`serve`/`logs`, the `.agent` package format, and
`~/.kiln` project/registry management — is TypeScript-only for now. This package is the **embeddable
SDK**: you `define_agent(...)` directly in your own Python code and run it in-process, the same pattern
the TS SDK uses when embedded in a Node backend (no CLI, no separate project folder). If you need a
long-running HTTP surface for a non-Python caller, run the TS package's `kiln serve` instead — this
package is for using kiln *from* Python, not for running the CLI.

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
    model="llama-3.3-70b-versatile",
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
