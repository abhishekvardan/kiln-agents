from typing import Any, Dict, List, Optional

from .agent import AgentDefinition
from .events import EventBus
from .orchestrator import Orchestrator, PipelineOptions, PipelineResult, PipelineStep
from .providers import ClaudeProvider, GeminiProvider, GroqProvider, OllamaProvider, OpenAIProvider, ProviderManager
from .runtime import AgentRunResult, AgentRuntime
from .team import Team


class Kiln:
    def __init__(self, events: EventBus, agent_runtime: AgentRuntime, orchestrator: Orchestrator) -> None:
        self.events = events
        self._agent_runtime = agent_runtime
        self._orchestrator = orchestrator

    async def run_inline_agent(self, definition: AgentDefinition, input: Any = None, run_id: Optional[str] = None) -> AgentRunResult:
        """Runs an already-in-hand AgentDefinition — no file, no directory, no `kiln init`. Define an
        agent with `define_agent(...)` in the same module/process and run it directly."""
        return await self._agent_runtime.run_inline_agent(definition, input, run_id)

    async def orchestrate(self, steps: List[PipelineStep], options: Optional[PipelineOptions] = None) -> PipelineResult:
        """Runs a DAG of agent steps with `depends_on`, retries, a concurrency pool, and cascading
        failure isolation. The actual multi-agent orchestration primitive."""
        return await self._orchestrator.run(steps, options)

    def team(self, agents: List[AgentDefinition]) -> Team:
        """A named roster of agents that can reach each other directly via `ctx.team.ask(name, input)`,
        on top of a shared `ctx.team` blackboard — dynamic collaboration where the running agent decides
        at runtime who else to consult, unlike `orchestrate()`'s pre-wired DAG. See `Team`."""
        return Team(self._agent_runtime, agents)


def create_kiln() -> Kiln:
    """Creates a ready-to-use kiln runtime (all built-in providers registered) — the embeddable
    runtime for a Python backend to import directly, no HTTP hop or CLI needed."""
    events = EventBus()
    providers = ProviderManager()
    for provider in [OpenAIProvider(), GeminiProvider(), ClaudeProvider(), GroqProvider(), OllamaProvider()]:
        providers.register(provider)
    memory_store: Dict[str, Any] = {}
    agent_runtime = AgentRuntime(providers, memory_store, events)
    orchestrator = Orchestrator(agent_runtime)
    return Kiln(events=events, agent_runtime=agent_runtime, orchestrator=orchestrator)
