import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

import httpx

from ._errors import KilnError


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: Any


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCallRequest]] = None


@dataclass
class ToolDefinitionForProvider:
    name: str
    description: str
    parameters: Any


@dataclass
class ChatResult:
    content: str
    tool_calls: Optional[List[ToolCallRequest]] = None
    raw: Any = None


@dataclass
class ChatOptions:
    model: str
    messages: List[ChatMessage]
    tools: Optional[List[ToolDefinitionForProvider]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ProviderAdapter(Protocol):
    name: str

    async def chat(self, options: ChatOptions) -> ChatResult: ...
    async def stream(self, options: ChatOptions, on_chunk: Callable[[str, bool], None]) -> ChatResult: ...
    async def tool_call(self, options: ChatOptions) -> ChatResult: ...
    async def json(self, options: ChatOptions) -> Any: ...


class StubProvider:
    """Matches kiln's TS behavior: gemini/claude are declared but not implemented yet."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def _unavailable(self) -> Any:
        raise KilnError(f"{self.name} is not implemented yet.")

    async def chat(self, options: ChatOptions) -> ChatResult:
        return await self._unavailable()

    async def stream(self, options: ChatOptions, on_chunk: Callable[[str, bool], None]) -> ChatResult:
        return await self._unavailable()

    async def tool_call(self, options: ChatOptions) -> ChatResult:
        return await self._unavailable()

    async def json(self, options: ChatOptions) -> Any:
        return await self._unavailable()


class OpenAICompatibleProvider:
    """Real httpx-based ProviderAdapter for any OpenAI-compatible /chat/completions endpoint
    (OpenAI, Groq, Ollama) — the same one adapter reused across all three, exactly like kiln's TS
    OpenAICompatibleProvider."""

    def __init__(self, name: str, base_url: str, api_key_env_var: Optional[str] = None) -> None:
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._api_key_env_var = api_key_env_var

    async def chat(self, options: ChatOptions) -> ChatResult:
        raw = await self._request("/chat/completions", self._to_request_body(options))
        return self._to_chat_result(raw)

    async def tool_call(self, options: ChatOptions) -> ChatResult:
        return await self.chat(options)

    async def json(self, options: ChatOptions) -> Any:
        body = {**self._to_request_body(options), "response_format": {"type": "json_object"}}
        raw = await self._request("/chat/completions", body)
        result = self._to_chat_result(raw)
        try:
            return json.loads(result.content)
        except json.JSONDecodeError as error:
            raise KilnError(f"{self.name} did not return valid JSON.") from error

    async def stream(self, options: ChatOptions, on_chunk: Callable[[str, bool], None]) -> ChatResult:
        body = {**self._to_request_body(options), "stream": True}
        content = ""
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", f"{self._base_url}/chat/completions", json=body, headers=self._headers()) as response:
                if response.status_code >= 400:
                    text = await response.aread()
                    raise KilnError(f"{self.name} request failed ({response.status_code}): {text.decode()}")
                async for line in response.aiter_lines():
                    trimmed = line.strip()
                    if not trimmed.startswith("data:"):
                        continue
                    payload = trimmed[len("data:"):].strip()
                    if payload == "[DONE]":
                        on_chunk("", True)
                        continue
                    try:
                        parsed = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    delta = (parsed.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
                    if delta:
                        content += delta
                        on_chunk(delta, False)
        return ChatResult(content=content)

    def _headers(self) -> Dict[str, str]:
        api_key = os.environ.get(self._api_key_env_var) if self._api_key_env_var else None
        if self._api_key_env_var and not api_key:
            raise KilnError(f"{self.name} requires the {self._api_key_env_var} environment variable to be set.")
        headers = {"content-type": "application/json"}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        return headers

    def _to_request_body(self, options: ChatOptions) -> Dict[str, Any]:
        messages = []
        for message in options.messages:
            entry: Dict[str, Any] = {"role": message.role, "content": message.content}
            if message.name:
                entry["name"] = message.name
            if message.tool_call_id:
                entry["tool_call_id"] = message.tool_call_id
            if message.tool_calls:
                entry["tool_calls"] = [
                    {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments)}}
                    for call in message.tool_calls
                ]
            messages.append(entry)

        body: Dict[str, Any] = {"model": options.model, "messages": messages}
        if options.temperature is not None:
            body["temperature"] = options.temperature
        if options.max_tokens is not None:
            body["max_tokens"] = options.max_tokens
        if options.tools:
            body["tools"] = [
                {"type": "function", "function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters}}
                for tool in options.tools
            ]
        return body

    def _to_chat_result(self, raw: Dict[str, Any]) -> ChatResult:
        message = (raw.get("choices") or [{}])[0].get("message") or {}
        raw_tool_calls = message.get("tool_calls")
        tool_calls = None
        if raw_tool_calls:
            tool_calls = [
                ToolCallRequest(id=call["id"], name=call["function"]["name"], arguments=self._parse_arguments(call["function"]["arguments"]))
                for call in raw_tool_calls
            ]
        return ChatResult(content=message.get("content") or "", tool_calls=tool_calls, raw=raw)

    @staticmethod
    def _parse_arguments(raw: str) -> Any:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    async def _request(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self._base_url}{path}", json=body, headers=self._headers())
        if response.status_code >= 400:
            raise KilnError(f"{self.name} request failed ({response.status_code}): {response.text}")
        return response.json()


def _resolve_ollama_base_url() -> str:
    """Ollama's own convention for OLLAMA_HOST is bare host:port (no scheme, no /v1 suffix), but kiln
    talks to Ollama's OpenAI-compatible endpoint, which lives under /v1. Normalize instead of requiring
    the caller to know kiln's internal shape — same fix as the TS provider."""
    raw = os.environ.get("OLLAMA_HOST")
    if not raw:
        return "http://localhost:11434/v1"
    with_scheme = raw if "://" in raw else f"http://{raw}"
    trimmed = with_scheme.rstrip("/")
    return trimmed if trimmed.endswith("/v1") else f"{trimmed}/v1"


class OpenAIProvider(OpenAICompatibleProvider):
    def __init__(self) -> None:
        super().__init__("openai", "https://api.openai.com/v1", "OPENAI_API_KEY")


class GroqProvider(OpenAICompatibleProvider):
    def __init__(self) -> None:
        super().__init__("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY")


class OllamaProvider(OpenAICompatibleProvider):
    def __init__(self) -> None:
        super().__init__("ollama", _resolve_ollama_base_url())


class GeminiProvider(StubProvider):
    def __init__(self) -> None:
        super().__init__("gemini")


class ClaudeProvider(StubProvider):
    def __init__(self) -> None:
        super().__init__("claude")


class ProviderManager:
    def __init__(self) -> None:
        self._providers: Dict[str, ProviderAdapter] = {}

    def register(self, provider: ProviderAdapter) -> None:
        self._providers[provider.name] = provider

    def resolve(self, name: str) -> Optional[ProviderAdapter]:
        return self._providers.get(name)
