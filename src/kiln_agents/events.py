import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List


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
    of requiring every `ctx.log(...)` call to be awaited."""

    def __init__(self) -> None:
        self._subscribers: List[Callable[[RuntimeEvent], Awaitable[None]]] = []

    def subscribe(self, handler: Callable[[RuntimeEvent], Awaitable[None]]) -> None:
        self._subscribers.append(handler)

    async def publish(self, event_type: str, execution_id: str, payload: Any = None) -> None:
        event = RuntimeEvent(type=event_type, execution_id=execution_id, at=time.time(), payload=payload)
        for handler in self._subscribers:
            await handler(event)

    def publish_nowait(self, event_type: str, execution_id: str, payload: Any = None) -> None:
        try:
            asyncio.get_running_loop().create_task(self.publish(event_type, execution_id, payload))
        except RuntimeError:
            # No running loop (e.g. called outside async context) — drop rather than crash the caller;
            # ctx.log() is best-effort telemetry, not something that should ever break an agent.
            pass
