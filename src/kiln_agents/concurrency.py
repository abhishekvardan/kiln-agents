import asyncio
import os
from typing import Optional


class ConcurrencyPool:
    """Caps how many async jobs run at once; excess jobs queue and run as slots free up. Same default
    as kiln's TS ConcurrencyPool: KILN_CONCURRENCY env var, falling back to 20."""

    def __init__(self, limit: Optional[int] = None) -> None:
        resolved = limit or int(os.environ.get("KILN_CONCURRENCY", "20"))
        self._semaphore = asyncio.Semaphore(resolved)

    async def __aenter__(self) -> "ConcurrencyPool":
        await self._semaphore.acquire()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self._semaphore.release()
