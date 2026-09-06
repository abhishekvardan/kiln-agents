import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Set


@dataclass
class RuntimeEvent:
    type: str
    execution_id: str
    at: float
    payload: Any = None


class EventBus:
    """A minimal typed pub/sub bus — subscribers are async callables invoked for every published
    event. `log()` inside an agent needs to stay a synchronous call site (matching kiln's TS
    `log(event, data?): void`), so `publish_nowait` schedules the publish as a background task instead
    of requiring every `ctx.log(...)` call to be awaited — see `AgentRuntime`'s `_pending` tracking for
    how those scheduled tasks are still guaranteed to finish before a run's result is handed back
    (`asyncio.create_task` does NOT run synchronously the way kiln's TS `void events.publish(...)` does
    — a naive port would drop trace entries scheduled right before a run returns)."""

    def __init__(self) -> None:
        self._subscribers: List[Callable[[RuntimeEvent], Awaitable[None]]] = []

    def subscribe(self, handler: Callable[[RuntimeEvent], Awaitable[None]]) -> Callable[[], None]:
        """Returns an unsubscribe callable — the Python equivalent of TS's `on(type, handler): () =>
        void`. Safe to call more than once (a second call is a no-op), matching the TS EventBus's own
        `Set.delete` semantics."""
        self._subscribers.append(handler)

        def unsubscribe() -> None:
            try:
                self._subscribers.remove(handler)
            except ValueError:
                pass

        return unsubscribe

    async def publish(self, event_type: str, execution_id: str, payload: Any = None) -> None:
        event = RuntimeEvent(type=event_type, execution_id=execution_id, at=time.time(), payload=payload)
        # A snapshot, not the live list — a handler that unsubscribes itself (or another handler) mid-
        # dispatch must not mutate the list this loop is iterating.
        for handler in list(self._subscribers):
            await handler(event)

    def publish_nowait(self, event_type: str, execution_id: str, payload: Any = None, pending: Set[asyncio.Task] = None) -> None:
        try:
            task = asyncio.get_running_loop().create_task(self.publish(event_type, execution_id, payload))
            if pending is not None:
                pending.add(task)
                task.add_done_callback(pending.discard)
        except RuntimeError:
            # No running loop (e.g. called outside async context) — drop rather than crash the caller;
            # ctx.log() is best-effort telemetry, not something that should ever break an agent.
            pass
