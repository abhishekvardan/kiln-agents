"""kiln-agents — a native Python port of kiln: build and orchestrate AI agents.

    from kiln_agents import create_kiln, define_agent, define_tool
"""

from ._errors import KilnError
from .agent import AgentContext, AgentDefinition, AgentMemory, BoundAI, define_agent
from .concurrency import ConcurrencyPool
from .events import EventBus, RuntimeEvent
from .kiln import Kiln, create_kiln
from .mcp import MCPConnection, connect_mcp
from .orchestrator import (
    Orchestrator,
    PipelineOptions,
    PipelineResult,
    PipelineRunContext,
    PipelineStep,
    PipelineStepOutcome,
)
from .prompt import PromptOptions, combine_prompts, define_prompt
from .providers import (
    ChatMessage,
    ChatOptions,
    ChatResult,
    ClaudeProvider,
    GeminiProvider,
    GroqProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    ProviderAdapter,
    ProviderManager,
    ToolCallRequest,
    ToolDefinitionForProvider,
)
from .runtime import AgentRunResult, AgentRuntime
from .tool import Tool, ToolExecutionContext, define_tool

__version__ = "0.1.0"

__all__ = [
    "KilnError",
    "AgentContext",
    "AgentDefinition",
    "AgentMemory",
    "BoundAI",
    "define_agent",
    "ConcurrencyPool",
    "EventBus",
    "RuntimeEvent",
    "Kiln",
    "create_kiln",
    "MCPConnection",
    "connect_mcp",
    "Orchestrator",
    "PipelineOptions",
    "PipelineResult",
    "PipelineRunContext",
    "PipelineStep",
    "PipelineStepOutcome",
    "PromptOptions",
    "combine_prompts",
    "define_prompt",
    "ChatMessage",
    "ChatOptions",
    "ChatResult",
    "ClaudeProvider",
    "GeminiProvider",
    "GroqProvider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "OpenAIProvider",
    "ProviderAdapter",
    "ProviderManager",
    "ToolCallRequest",
    "ToolDefinitionForProvider",
    "AgentRunResult",
    "AgentRuntime",
    "Tool",
    "ToolExecutionContext",
    "define_tool",
]
