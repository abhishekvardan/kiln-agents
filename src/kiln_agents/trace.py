from dataclasses import dataclass, field
from typing import Any, Dict, List

from .events import EventBus, RuntimeEvent


@dataclass
class TraceEvent:
    """One entry in a run's structured trace, in the order it actually happened. `type` is either a
    lifecycle event ("AgentStarted" / "AgentCompleted" / "AgentFailed") or the name passed to
    `ctx.log(event, data)` — "LLMCallCompleted", "ToolCallCompleted", "ObjectAttemptFailed"/
    "ObjectAttemptSucceeded" are the ones `AgentRuntime` itself emits automatically for every run, so a
    trace exists whether or not the agent's own code ever calls `ctx.log()`."""

    type: str
    at: float
    data: Dict[str, Any] = field(default_factory=dict)


_LIFECYCLE_TYPES = {"AgentStarted", "AgentCompleted", "AgentFailed"}


class TraceCapture:
    """Subscribes to one run's full lifecycle + log stream and collects everything, in order. Construct
    it *before* starting the run (same "subscribe first" rule as kiln's TS `captureTrace`/`Team`) so
    nothing fires between "start listening" and "start running" gets missed, then call `stop()` once the
    run has settled to unsubscribe and get the collected list back.

    This is the one mechanism behind every trace surface kiln-agents exposes: `TeamAskRecord.trace` and
    `TeamRunResult.trace` are both just `TraceCapture(...).stop()` around the matching
    `run_inline_agent()` / `Team.run()` call.
    """

    def __init__(self, events: EventBus, run_id: str) -> None:
        self._trace: List[TraceEvent] = []

        async def on_event(event: RuntimeEvent) -> None:
            if event.execution_id != run_id:
                return
            if event.type in _LIFECYCLE_TYPES:
                self._push(event.type, event.at, event.payload)
            elif event.type == "AgentLog":
                # Unwrapped one level, same as the TS port — the trace shows the real event name
                # ("LLMCallCompleted", "ToolCallCompleted", ...) directly, not "AgentLog" for everything.
                payload = event.payload or {}
                self._push(payload.get("event", "Log"), event.at, payload.get("data"))

        self._unsubscribe = events.subscribe(on_event)

    def _push(self, event_type: str, at: float, data: Any) -> None:
        self._trace.append(TraceEvent(type=event_type, at=at, data=data if isinstance(data, dict) else {"value": data}))

    def stop(self) -> List[TraceEvent]:
        self._unsubscribe()
        return self._trace
