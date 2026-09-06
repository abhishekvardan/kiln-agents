import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from ._errors import KilnError
from .providers import ChatMessage, ChatOptions, ChatResult, ProviderAdapter, ToolDefinitionForProvider
from .tool import Tool

T = TypeVar("T", bound=BaseModel)

MessageLike = Any  # ChatMessage or a plain {"role": ..., "content": ...} dict


def _normalize_messages(messages: List[MessageLike]) -> List[ChatMessage]:
    """Accepts plain dicts (`{"role": "user", "content": "..."}`, the natural way to write these in
    Python — same shape the OpenAI SDK itself uses) as well as ChatMessage instances."""
    normalized = []
    for message in messages:
        if isinstance(message, ChatMessage):
            normalized.append(message)
        else:
            normalized.append(
                ChatMessage(
                    role=message["role"],
                    content=message.get("content", ""),
                    name=message.get("name"),
                    tool_call_id=message.get("tool_call_id"),
                    tool_calls=message.get("tool_calls"),
                )
            )
    return normalized


def _messages_as_dicts(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """Trace entries carry plain dicts, not dataclass instances — so a captured trace is directly
    JSON-serializable (e.g. for an MCP debugging server built on top of this later) without every
    consumer needing to know about ChatMessage."""
    return [{"role": m.role, "content": m.content, **({"name": m.name} if m.name else {}), **({"tool_call_id": m.tool_call_id} if m.tool_call_id else {})} for m in messages]


@dataclass
class AgentDefinition:
    name: str
    execute: Callable[["AgentContext"], Awaitable[Any]]
    description: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    tools: List[Tool] = field(default_factory=list)
    connectors: List[Any] = field(default_factory=list)


def define_agent(
    *,
    name: str,
    execute: Callable[["AgentContext"], Awaitable[Any]],
    description: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    tools: Optional[List[Tool]] = None,
    connectors: Optional[List[Any]] = None,
) -> AgentDefinition:
    return AgentDefinition(
        name=name,
        execute=execute,
        description=description,
        provider=provider,
        model=model,
        tools=tools or [],
        connectors=connectors or [],
    )


class BoundAI:
    """The AI client handed to an agent's execute(ctx) — bound to the agent's configured
    provider/model/tools. This is a line-for-line port of kiln's TS `BoundAI`, including the exact
    schema-instruction wording in `object()` (a real bug fix: smaller models confuse "the schema" with
    "an instance of it" if you don't spell out the difference with a worked example).

    Every call below logs a structured "LLMCallCompleted" event — provider, model, the exact messages
    sent, the exact response (or error), and duration — regardless of success or failure and regardless
    of which method the agent's own code called. This, plus the tool-call and validation-attempt logging
    further down, is what a `TraceCapture` around a run actually collects — see `trace.py`.
    """

    def __init__(
        self,
        resolve_provider: Callable[[], ProviderAdapter],
        provider_name: str,
        model: str,
        tool_definitions: List[ToolDefinitionForProvider],
        invoke_tool: Callable[[str, Any], Awaitable[Any]],
        log: Callable[[str, Any], None],
    ) -> None:
        self._resolve_provider = resolve_provider
        self._provider_name = provider_name
        self._model = model
        self._tool_definitions = tool_definitions
        self._invoke_tool = invoke_tool
        self._log = log

    def _log_llm_call(self, kind: str, messages: List[ChatMessage], started: float, response: Any = None, error: Optional[str] = None) -> None:
        payload: Dict[str, Any] = {
            "kind": kind,
            "provider": self._provider_name,
            "model": self._model,
            "messages": _messages_as_dicts(messages),
            "duration_ms": int((time.monotonic() - started) * 1000),
        }
        payload["error" if error else "response"] = error if error else response
        self._log("LLMCallCompleted", payload)

    async def chat(self, messages: List[MessageLike], **kwargs: Any) -> ChatResult:
        provider = self._resolve_provider()
        normalized = _normalize_messages(messages)
        started = time.monotonic()
        try:
            result = await provider.chat(ChatOptions(model=self._model, messages=normalized, **kwargs))
            self._log_llm_call("chat", normalized, started, response=result)
            return result
        except Exception as error:
            self._log_llm_call("chat", normalized, started, error=str(error))
            raise

    async def stream(self, messages: List[MessageLike], on_chunk: Callable[[str, bool], None], **kwargs: Any) -> ChatResult:
        provider = self._resolve_provider()
        normalized = _normalize_messages(messages)
        started = time.monotonic()
        try:
            result = await provider.stream(ChatOptions(model=self._model, messages=normalized, **kwargs), on_chunk)
            self._log_llm_call("stream", normalized, started, response=result)
            return result
        except Exception as error:
            self._log_llm_call("stream", normalized, started, error=str(error))
            raise

    async def json(self, messages: List[MessageLike], **kwargs: Any) -> Any:
        provider = self._resolve_provider()
        normalized = _normalize_messages(messages)
        started = time.monotonic()
        try:
            result = await provider.json(ChatOptions(model=self._model, messages=normalized, **kwargs))
            self._log_llm_call("json", normalized, started, response=result)
            return result
        except Exception as error:
            self._log_llm_call("json", normalized, started, error=str(error))
            raise

    async def run(self, messages: List[MessageLike], max_steps: int = 4) -> ChatResult:
        """The tool-calling loop: call the model, execute any tool_calls, feed the results back,
        repeat — up to max_steps."""
        provider = self._resolve_provider()
        conversation: List[ChatMessage] = _normalize_messages(messages)
        last = ChatResult(content="")
        for _ in range(max_steps):
            step_started = time.monotonic()
            last = await provider.tool_call(ChatOptions(model=self._model, messages=conversation, tools=self._tool_definitions or None))
            self._log_llm_call("tool_call", conversation, step_started, response=last)
            if not last.tool_calls:
                return last
            conversation.append(ChatMessage(role="assistant", content=last.content, tool_calls=last.tool_calls))
            for call in last.tool_calls:
                self._log("ToolCallRequested", {"tool": call.name, "arguments": call.arguments})
                tool_started = time.monotonic()
                error: Optional[str] = None
                try:
                    result = await self._invoke_tool(call.name, call.arguments)
                except Exception as exc:  # noqa: BLE001 — mirrored: a failing tool becomes a normal {error} result, not a crash
                    error = str(exc)
                    result = {"error": error}
                completed: Dict[str, Any] = {"tool": call.name, "arguments": call.arguments, "duration_ms": int((time.monotonic() - tool_started) * 1000)}
                completed["error" if error else "result"] = error if error else result
                self._log("ToolCallCompleted", completed)
                conversation.append(ChatMessage(role="tool", name=call.name, tool_call_id=call.id, content=json.dumps(result)))
        return last

    async def object(self, messages: List[MessageLike], schema: Type[T], max_attempts: int = 3) -> T:
        """Structured output with a pydantic schema: writes the schema instruction into the prompt for
        you, validates the model's response against it, and — on failure — feeds the validation error
        back and retries, up to `max_attempts`. Returns fully typed, validated data; raises KilnError if
        it never validates."""
        provider = self._resolve_provider()
        messages = _normalize_messages(messages)
        schema_instruction = ChatMessage(
            role="system",
            content=(
                "You must respond with ONLY a JSON DATA OBJECT — an actual answer, never the schema itself, never keys named "
                '"type"/"properties"/"required"/"$schema". Here is the JSON Schema describing the shape your DATA OBJECT must '
                f"have (for reference only — do not copy it back): {json.dumps(schema.model_json_schema())}\n\n"
                'Example of the difference: if the schema says {"type":"object","properties":{"name":{"type":"string"}}}, a '
                'correct DATA answer looks like {"name":"Alice"} — real values, not type descriptions. '
                "No markdown, no code fences, no commentary outside the JSON."
            ),
        )
        last_error = ""
        for attempt in range(1, max_attempts + 1):
            conversation = (
                [schema_instruction, *messages]
                if attempt == 1
                else [
                    schema_instruction,
                    *messages,
                    ChatMessage(role="user", content=f"Your previous response was invalid: {last_error}. Respond again with ONLY corrected JSON matching the required schema."),
                ]
            )
            attempt_started = time.monotonic()
            try:
                raw = await provider.json(ChatOptions(model=self._model, messages=conversation))
                self._log_llm_call("object", conversation, attempt_started, response=raw)
            except Exception as error:  # noqa: BLE001
                last_error = str(error)
                self._log_llm_call("object", conversation, attempt_started, error=last_error)
                self._log("ObjectAttemptFailed", {"attempt": attempt, "error": last_error})
                continue
            try:
                validated = schema.model_validate(raw)
                self._log("ObjectAttemptSucceeded", {"attempt": attempt, "output": validated.model_dump()})
                return validated
            except ValidationError as error:
                last_error = "; ".join(f"{'.'.join(str(p) for p in issue['loc']) or '(root)'}: {issue['msg']}" for issue in error.errors())
                self._log("ObjectAttemptFailed", {"attempt": attempt, "error": last_error, "raw_output": raw})
        raise KilnError(f"ai.object() did not produce output matching the schema after {max_attempts} attempt(s): {last_error}")


class AgentMemory:
    """Private to one run by default — scoped by a `session:<id>` key prefix carved out of one shared
    backing dict, so two concurrent runs (even of the same agent, even using the same key) never see
    each other's writes. See `AgentRuntime.run_inline_agent`'s `session_id` for opting into sharing this
    across separate runs on purpose (e.g. turns of one chat thread)."""

    def __init__(self, store: Dict[str, Any], scope: str) -> None:
        self._store = store
        self._scope = scope

    async def get(self, key: str) -> Any:
        return self._store.get(f"{self._scope}:{key}")

    async def set(self, key: str, value: Any) -> None:
        self._store[f"{self._scope}:{key}"] = value


class TeamMemory(AgentMemory):
    """A shared blackboard (`get`/`set`, scoped by `team:<id>` — separate from each step's own private
    `memory`) plus `ask()`: directly invokes another named agent on this run's team, mid-execution, and
    returns its raw output. Every standalone run gets a team of one, scoped to its own run_id — get/set
    are always safe to call, never `None`; `ask()` is the one part that needs a real roster (only a run
    started via `Team.run()` supplies one; otherwise it raises `KilnError`)."""

    def __init__(self, store: Dict[str, Any], scope: str, ask: Callable[[str, Any], Awaitable[Any]]) -> None:
        super().__init__(store, scope)
        self.ask = ask


@dataclass
class AgentContext:
    input: Any
    run_id: str
    ai: BoundAI
    tools: Dict[str, Callable[[Any], Awaitable[Any]]]
    memory: AgentMemory
    team: TeamMemory
    log: Callable[[str, Any], None]
