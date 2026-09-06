import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ._errors import KilnError
from .agent import AgentDefinition
from .events import EventBus
from .runtime import AgentRuntime
from .trace import TraceCapture, TraceEvent


@dataclass
class TeamAskRecord:
    """One `ctx.team.ask()` call, in order — who asked whom, what went in, what came back (or failed).
    The literal record of how the team collaborated, not an inference from the outputs."""

    from_agent: str
    to_agent: str
    input: Any
    output: Any = None
    error: Optional[str] = None
    started_at: float = 0.0
    duration_ms: int = 0
    trace: List[TraceEvent] = field(default_factory=list)
    """The asked agent's own full structured trace for this one call — every LLM call, tool call, and
    validation attempt it made while answering. This is what turns "billing-specialist gave a wrong
    answer" into "here's the exact prompt it was sent and the exact model response" — the collaboration
    record (from/to/input/output) tells you *that* it happened; this tells you *why*."""


@dataclass
class TeamMember:
    """One roster member, as much as a caller needs to introspect a team without touching its agents
    directly."""

    name: str
    description: Optional[str] = None


@dataclass
class TeamRunResult:
    status: str  # "succeeded" | "failed"
    output: Any = None
    error: Optional[str] = None
    team_id: str = ""
    run_id: str = ""
    asks: List[TeamAskRecord] = field(default_factory=list)
    duration_ms: int = 0
    trace: List[TraceEvent] = field(default_factory=list)
    """The lead agent's own full structured trace — same shape as each TeamAskRecord.trace, for the one
    call in this run that isn't itself an ask."""


class Team:
    """A named roster of agents that can reach each other directly (`ctx.team.ask(name, input)`) on top
    of a shared blackboard (`ctx.team.get`/`set`) — dynamic collaboration, not a pre-wired DAG. Use
    `Orchestrator` when you can draw the pipeline ahead of time; use a team when the collaboration
    pattern can't be known until run time.

    ```python
    team = Team(agent_runtime, [billing_specialist, technical_specialist, coordinator])
    result = await team.run("coordinator", {"question": "My invoice is wrong and the app won't load"})
    # result.asks: [TeamAskRecord(from_agent="coordinator", to_agent="billing-specialist", ...), ...]
    ```

    Every ask also publishes live on `agent_runtime.events` (`TeamAskStarted`/`TeamAskCompleted`, keyed
    by the run's `run_id`) as it happens, not just in the final `asks` list.
    """

    def __init__(self, agent_runtime: AgentRuntime, agents: List[AgentDefinition]) -> None:
        self._agent_runtime = agent_runtime
        self._roster: Dict[str, AgentDefinition] = {}
        for agent in agents:
            if agent.name in self._roster:
                raise KilnError(f'Team: two agents both named "{agent.name}" — names must be unique on one team\'s roster.')
            self._roster[agent.name] = agent

    @property
    def events(self) -> EventBus:
        return self._agent_runtime.events

    def list_members(self) -> List[TeamMember]:
        return [TeamMember(name=agent.name, description=agent.description) for agent in self._roster.values()]

    def _roster_names(self) -> str:
        return ", ".join(self._roster.keys()) or "(empty roster)"

    async def run(
        self,
        lead_agent_name: str,
        input: Any,
        team_id: Optional[str] = None,
        run_id: Optional[str] = None,
        max_asks: int = 20,
    ) -> TeamRunResult:
        lead = self._roster.get(lead_agent_name)
        if not lead:
            raise KilnError(f'Team.run(): no agent named "{lead_agent_name}" on this team\'s roster (have: {self._roster_names()}).')

        team_id = team_id or str(uuid.uuid4())
        run_id = run_id or str(uuid.uuid4())
        asks: List[TeamAskRecord] = []
        remaining = {"n": max_asks}

        def dispatch_for(from_agent_name: str):
            async def ask(agent_name: str, ask_input: Any) -> Any:
                remaining["n"] -= 1
                if remaining["n"] < 0:
                    raise KilnError(f"Team: exceeded max_asks ({max_asks}) for one run — a runaway agent-to-agent loop? Raise Team.run()'s max_asks if this many asks is expected.")
                target = self._roster.get(agent_name)
                if not target:
                    raise KilnError(f'ctx.team.ask(): no agent named "{agent_name}" on this team\'s roster (have: {self._roster_names()}).')

                await self._agent_runtime.events.publish("TeamAskStarted", run_id, {"from": from_agent_name, "to": agent_name, "input": ask_input})

                ask_started = time.time()
                # A run_id of its own (not the team run's run_id) — this one call's trace has to be
                # capturable independently of every other ask happening in the same team run, including a
                # nested one to the very same teammate.
                ask_run_id = str(uuid.uuid4())
                trace_capture = TraceCapture(self._agent_runtime.events, ask_run_id)
                result = await self._agent_runtime.run_inline_agent(target, ask_input, run_id=ask_run_id, team_id=team_id, dispatch=dispatch_for(agent_name))
                duration_ms = int((time.time() - ask_started) * 1000)

                record = TeamAskRecord(
                    from_agent=from_agent_name,
                    to_agent=agent_name,
                    input=ask_input,
                    output=result.output if result.status == "succeeded" else None,
                    error=result.error if result.status == "failed" else None,
                    started_at=ask_started,
                    duration_ms=duration_ms,
                    trace=trace_capture.stop(),
                )
                asks.append(record)
                await self._agent_runtime.events.publish("TeamAskCompleted", run_id, record)

                if result.status == "failed":
                    raise KilnError(f'ctx.team.ask("{agent_name}", ...) failed: {result.error}')
                return result.output

            return ask

        await self._agent_runtime.events.publish("TeamRunStarted", run_id, {"lead_agent_name": lead_agent_name, "input": input})

        started = time.time()
        lead_run_id = str(uuid.uuid4())
        lead_trace = TraceCapture(self._agent_runtime.events, lead_run_id)
        result = await self._agent_runtime.run_inline_agent(lead, input, run_id=lead_run_id, team_id=team_id, dispatch=dispatch_for(lead_agent_name))
        final = TeamRunResult(
            status=result.status,
            output=result.output,
            error=result.error,
            team_id=team_id,
            run_id=run_id,
            asks=asks,
            duration_ms=int((time.time() - started) * 1000),
            trace=lead_trace.stop(),
        )
        await self._agent_runtime.events.publish("TeamRunCompleted", run_id, final)
        return final
