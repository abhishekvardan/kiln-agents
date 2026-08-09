import json
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
    "an instance of it" if you don't spell out the difference with a worked example)."""

    def __init__(
        self,
        resolve_provider: Callable[[], ProviderAdapter],
        model: str,
        tool_definitions: List[ToolDefinitionForProvider],
        invoke_tool: Callable[[str, Any], Awaitable[Any]],
        log: Callable[[str, Any], None],
    ) -> None:
        self._resolve_provider = resolve_provider
        self._model = model
        self._tool_definitions = tool_definitions
        self._invoke_tool = invoke_tool
        self._log = log

    async def chat(self, messages: List[MessageLike], **kwargs: Any) -> ChatResult:
        provider = self._resolve_provider()
        return await provider.chat(ChatOptions(model=self._model, messages=_normalize_messages(messages), **kwargs))

    async def stream(self, messages: List[MessageLike], on_chunk: Callable[[str, bool], None], **kwargs: Any) -> ChatResult:
        provider = self._resolve_provider()
        return await provider.stream(ChatOptions(model=self._model, messages=_normalize_messages(messages), **kwargs), on_chunk)

    async def json(self, messages: List[MessageLike], **kwargs: Any) -> Any:
        provider = self._resolve_provider()
        return await provider.json(ChatOptions(model=self._model, messages=_normalize_messages(messages), **kwargs))

    async def run(self, messages: List[MessageLike], max_steps: int = 4) -> ChatResult:
        """The tool-calling loop: call the model, execute any tool_calls, feed the results back,
        repeat — up to max_steps."""
        provider = self._resolve_provider()
        conversation: List[ChatMessage] = _normalize_messages(messages)
        last = ChatResult(content="")
        for _ in range(max_steps):
            last = await provider.tool_call(ChatOptions(model=self._model, messages=conversation, tools=self._tool_definitions or None))
            if not last.tool_calls:
                return last
            conversation.append(ChatMessage(role="assistant", content=last.content, tool_calls=last.tool_calls))
            for call in last.tool_calls:
                self._log("ToolCallRequested", {"tool": call.name, "arguments": call.arguments})
                try:
                    result = await self._invoke_tool(call.name, call.arguments)
                except Exception as error:  # noqa: BLE001 — mirrored: a failing tool becomes a normal {error} result, not a crash
                    result = {"error": str(error)}
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
            try:
                raw = await provider.json(ChatOptions(model=self._model, messages=conversation))
            except Exception as error:  # noqa: BLE001
                last_error = str(error)
                self._log("ObjectAttemptFailed", {"attempt": attempt, "error": last_error})
                continue
            try:
                return schema.model_validate(raw)
            except ValidationError as error:
                last_error = "; ".join(f"{'.'.join(str(p) for p in issue['loc']) or '(root)'}: {issue['msg']}" for issue in error.errors())
                self._log("ObjectAttemptFailed", {"attempt": attempt, "error": last_error})
        raise KilnError(f"ai.object() did not produce output matching the schema after {max_attempts} attempt(s): {last_error}")


class AgentMemory:
    def __init__(self, store: Dict[str, Any]) -> None:
        self._store = store

    async def get(self, key: str) -> Any:
        return self._store.get(key)

    async def set(self, key: str, value: Any) -> None:
        self._store[key] = value


@dataclass
class AgentContext:
    input: Any
    run_id: str
    ai: BoundAI
    tools: Dict[str, Callable[[Any], Awaitable[Any]]]
    memory: AgentMemory
    log: Callable[[str, Any], None]
