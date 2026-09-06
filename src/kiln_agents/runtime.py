import asyncio
import inspect
import time
import traceback
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Set

from pydantic import BaseModel, ValidationError

from ._errors import KilnError
from .agent import AgentContext, AgentDefinition, AgentMemory, BoundAI, TeamMemory
from .events import EventBus
from .providers import ProviderManager, ToolDefinitionForProvider
from .tool import Tool, ToolExecutionContext


@dataclass
class AgentRunResult:
    run_id: str
    status: str  # "succeeded" | "failed"
    output: Any = None
    error: Optional[str] = None
    duration_ms: int = 0
    agent_name: Optional[str] = None


# Wires ctx.team.ask() to a real roster — supplied by Team.run(), not meant to be built by hand.
TeamDispatch = Callable[[str, Any], Awaitable[Any]]


def _is_pydantic_model(value: Any) -> bool:
    return inspect.isclass(value) and issubclass(value, BaseModel)


class _ScopedToolManager:
    """Per-run tool registry + invocation — validates pydantic-schema tools locally before execute()
    runs; raw-JSON-Schema tools (e.g. MCP-sourced) pass through unvalidated, same distinction kiln's
    TS ToolManager makes."""

    def __init__(self, tools: list[Tool]) -> None:
        self._tools: Dict[str, Tool] = {tool.name: tool for tool in tools}

    async def invoke(self, name: str, tool_input: Any, ctx: ToolExecutionContext) -> Any:
        tool = self._tools.get(name)
        if not tool:
            raise KilnError(f'Tool "{name}" is not registered.')
        if _is_pydantic_model(tool.parameters):
            try:
                parsed = tool.parameters.model_validate(tool_input)
            except ValidationError as error:
                raise KilnError(f'Invalid input for tool "{name}": {error}') from error
        else:
            parsed = tool_input
        ctx.log("ToolCalled", {"tool": name, "input": parsed if not isinstance(parsed, BaseModel) else parsed.model_dump()})
        try:
            return await tool.execute(parsed, ctx)
        except Exception as error:
            ctx.log("ToolFailed", {"tool": name, "error": str(error)})
            raise


_tool_definitions_cache: Dict[int, list[ToolDefinitionForProvider]] = {}


def _tool_definitions_for(definition: AgentDefinition) -> list[ToolDefinitionForProvider]:
    """definition.tools -> provider-wire-format schema is a pure function of static data — cache it
    per definition instead of reconverting on every single run, same as kiln's TS WeakMap cache."""
    key = id(definition)
    cached = _tool_definitions_cache.get(key)
    if cached is not None:
        return cached
    computed = []
    for tool in definition.tools:
        if _is_pydantic_model(tool.parameters):
            parameters: Any = tool.parameters.model_json_schema()
        elif tool.parameters is not None:
            parameters = tool.parameters
        else:
            parameters = {"type": "object", "properties": {}}
        computed.append(ToolDefinitionForProvider(name=tool.name, description=tool.description, parameters=parameters))
    _tool_definitions_cache[key] = computed
    return computed


class AgentRuntime:
    """Runs an in-hand AgentDefinition — the Python port skips kiln's TS file-based package loading
    (there's no equivalent of `kiln init`/src/agents/*.ts dynamic import here), since embedding
    `defineAgent(...)` directly in your own code is the primary way to use kiln from Python."""

    def __init__(self, providers: ProviderManager, memory_store: Dict[str, Any], events: EventBus) -> None:
        self._providers = providers
        self._memory_store = memory_store
        self.events = events  # public: Team publishes TeamAskStarted/Completed through this same bus.

    async def run_inline_agent(
        self,
        definition: AgentDefinition,
        input: Any = None,
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
        team_id: Optional[str] = None,
        dispatch: Optional[TeamDispatch] = None,
    ) -> AgentRunResult:
        run_id = run_id or str(uuid.uuid4())
        started = time.monotonic()
        # A run's ctx.log()-triggered events (ToolCallCompleted, LLMCallCompleted, ...) are scheduled
        # as background tasks (publish_nowait), not awaited at the call site — matching kiln's TS `log()`
        # staying a synchronous void call. Unlike the TS EventBus (whose synchronous-handler dispatch
        # means a subscriber has already run by the time publish() "returns"), asyncio.create_task()
        # merely SCHEDULES the coroutine; without tracking these tasks and draining them before this
        # method returns, a trace captured around this call could silently be missing its last few
        # entries — the exact kind of bug a naive line-for-line port would ship.
        pending: Set[asyncio.Task] = set()
        await self.events.publish("AgentStarted", run_id, {"agent": definition.name, "input": input})
        try:
            ctx = self._create_context(definition, run_id, input, session_id, team_id, dispatch, pending)
            output = await definition.execute(ctx)
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            await self.events.publish("AgentCompleted", run_id, {"output": output})
            return AgentRunResult(run_id=run_id, status="succeeded", output=output, duration_ms=int((time.monotonic() - started) * 1000))
        except Exception as error:
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            message = str(error)
            # A traceback (when the failure is a real Python exception, which it always is here) is what
            # localizes "it failed" to a real line — the message alone often isn't enough in an agent
            # with several ctx.ai calls in its execute().
            await self.events.publish("AgentFailed", run_id, {"error": message, "stack": traceback.format_exc()})
            return AgentRunResult(run_id=run_id, status="failed", error=message, duration_ms=int((time.monotonic() - started) * 1000))

    def _create_context(
        self,
        definition: AgentDefinition,
        run_id: str,
        input: Any,
        session_id: Optional[str],
        team_id: Optional[str],
        dispatch: Optional[TeamDispatch],
        pending: Set[asyncio.Task],
    ) -> AgentContext:
        model = definition.model or "gpt-4o-mini"

        def log(event: str, data: Any = None) -> None:
            self.events.publish_nowait("AgentLog", run_id, {"event": event, "data": data}, pending=pending)

        scoped_tools = _ScopedToolManager(definition.tools)
        tool_ctx = ToolExecutionContext(run_id=run_id, log=log)

        async def invoke_tool(name: str, tool_input: Any) -> Any:
            return await scoped_tools.invoke(name, tool_input, tool_ctx)

        tools: Dict[str, Any] = {tool.name: (lambda tool_input, _name=tool.name: invoke_tool(_name, tool_input)) for tool in definition.tools}

        def resolve_provider():
            if not definition.provider:
                raise KilnError(f'Agent "{definition.name}" does not declare a provider.')
            provider = self._providers.resolve(definition.provider)
            if not provider:
                raise KilnError(f'Provider "{definition.provider}" is not registered.')
            return provider

        ai = BoundAI(
            resolve_provider=resolve_provider,
            provider_name=definition.provider or "",
            model=model,
            tool_definitions=_tool_definitions_for(definition),
            invoke_tool=invoke_tool,
            log=log,
        )

        # One backing dict, two independent key-prefix namespaces — mirrors kiln's TS fix for the exact
        # same bug: before this, every run and every concurrent request shared one unscoped memory map.
        # Private by default (scoped to this run's own run_id); a caller opts into sharing across runs
        # with session_id (ctx.memory) / team_id (ctx.team), same as TS's AgentRunOptions.
        memory = AgentMemory(self._memory_store, f"session:{session_id or run_id}")

        async def team_ask(agent_name: str, ask_input: Any) -> Any:
            if not dispatch:
                raise KilnError("ctx.team.ask() needs a real roster to dispatch to — this run has none. Only available inside a Team.run().")
            return await dispatch(agent_name, ask_input)

        team = TeamMemory(self._memory_store, f"team:{team_id or run_id}", team_ask)

        return AgentContext(input=input, run_id=run_id, ai=ai, tools=tools, memory=memory, team=team, log=log)
