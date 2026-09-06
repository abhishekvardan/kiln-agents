"""Shared test doubles — a scripted ProviderAdapter that returns queued responses instead of making a
real network call, the same pattern kiln's TS test suite uses."""

from typing import Any, Callable, List, Optional

from kiln_agents.providers import ChatMessage, ChatOptions, ChatResult, ProviderManager, ToolCallRequest
from kiln_agents.events import EventBus
from kiln_agents.runtime import AgentRuntime


class ScriptedProvider:
    """`json_responses` are returned in order from `json()`; `tool_call_responses` are returned in
    order from `tool_call()`. A queued `Exception` instance is raised instead of returned."""

    def __init__(self, json_responses: Optional[List[Any]] = None, tool_call_responses: Optional[List[ChatResult]] = None) -> None:
        self.name = "scripted"
        self._json_responses = json_responses or []
        self._json_index = 0
        self._tool_call_responses = tool_call_responses or []
        self._tool_call_index = 0

    async def json(self, options: ChatOptions) -> Any:
        value = self._json_responses[min(self._json_index, len(self._json_responses) - 1)]
        self._json_index += 1
        if isinstance(value, Exception):
            raise value
        return value

    async def tool_call(self, options: ChatOptions) -> ChatResult:
        value = self._tool_call_responses[min(self._tool_call_index, len(self._tool_call_responses) - 1)]
        self._tool_call_index += 1
        return value

    async def chat(self, options: ChatOptions) -> ChatResult:
        return ChatResult(content="")

    async def stream(self, options: ChatOptions, on_chunk: Callable[[str, bool], None]) -> ChatResult:
        return ChatResult(content="")


def runtime_with(provider) -> tuple[AgentRuntime, EventBus]:
    providers = ProviderManager()
    providers.register(provider)
    events = EventBus()
    return AgentRuntime(providers, {}, events), events
