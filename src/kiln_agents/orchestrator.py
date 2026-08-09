import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from ._errors import KilnError
from .agent import AgentDefinition
from .concurrency import ConcurrencyPool
from .runtime import AgentRunResult, AgentRuntime


@dataclass
class PipelineRunContext:
    """Each completed step's output, keyed by step id. Only deps declared in `depends_on` are safe to read."""

    outputs: Dict[str, Any]
    results: Dict[str, AgentRunResult]


InputResolver = Union[Any, Callable[[PipelineRunContext], Any]]


@dataclass
class PipelineStep:
    id: str
    agent: AgentDefinition
    input: InputResolver = None
    """Static input, or a function of prior steps' outputs — called once per attempt.
    `input=lambda ctx: {...}` (sync or async callables both work)."""
    depends_on: Optional[List[str]] = None
    """Which step ids must succeed before this one runs. Omit for a root step (no dependencies)."""
    retries: Optional[int] = None
    """Overrides PipelineOptions.retries for this step only."""
    is_valid: Optional[Callable[[Any], bool]] = None
    """Overrides the default validity check for this step only."""


@dataclass
class PipelineOptions:
    concurrency: Optional[int] = None
    """Max steps running at once — independent branches of the DAG run concurrently up to this cap."""
    retries: Optional[int] = None
    """Default extra attempts (beyond the first) for any step that doesn't set its own `retries`."""
    on_step: Optional[Callable[[str, AgentRunResult, int], None]] = None


@dataclass
class PipelineStepOutcome:
    id: str
    status: str  # "succeeded" | "failed" | "skipped"
    result: Optional[AgentRunResult] = None
    attempts: int = 0
    skipped_reason: Optional[str] = None


@dataclass
class PipelineResult:
    outputs: Dict[str, Any]
    results: Dict[str, AgentRunResult]
    steps: List[PipelineStepOutcome]
    duration_ms: int


def _default_is_valid(output: Any) -> bool:
    """`{"raw": "..."}` is the parse-failure fallback convention used throughout kiln's own example
    agents — treat it as invalid by default so a malformed model response triggers a retry instead of
    silently flowing downstream."""
    if isinstance(output, dict) and "raw" in output:
        return not output.get("raw")
    return True


def _detect_cycle(steps: List[PipelineStep], by_id: Dict[str, PipelineStep]) -> None:
    state: Dict[str, str] = {}

    def visit(step_id: str, path: List[str]) -> None:
        if state.get(step_id) == "done":
            return
        if state.get(step_id) == "visiting":
            raise KilnError(f"Pipeline has a dependency cycle: {' -> '.join([*path, step_id])}")
        state[step_id] = "visiting"
        for dep in by_id[step_id].depends_on or []:
            visit(dep, [*path, step_id])
        state[step_id] = "done"

    for step in steps:
        visit(step.id, [])


class Orchestrator:
    """Runs a DAG of agent steps: independent branches execute concurrently (capped by
    ConcurrencyPool), dependent steps wait for their declared `depends_on` to succeed, a step with a
    failed dependency is marked "skipped" rather than run with missing input (and its own dependents
    cascade-skip too), and each step gets its own retry budget against a pluggable validity check."""

    def __init__(self, agent_runtime: AgentRuntime) -> None:
        self._agent_runtime = agent_runtime

    async def run(self, steps: List[PipelineStep], options: Optional[PipelineOptions] = None) -> PipelineResult:
        options = options or PipelineOptions()
        started = time.monotonic()
        if not steps:
            return PipelineResult(outputs={}, results={}, steps=[], duration_ms=0)

        by_id = {step.id: step for step in steps}
        for step in steps:
            for dep in step.depends_on or []:
                if dep not in by_id:
                    raise KilnError(f'Step "{step.id}" depends on unknown step "{dep}".')
        _detect_cycle(steps, by_id)

        pool = ConcurrencyPool(options.concurrency)
        outputs: Dict[str, Any] = {}
        results: Dict[str, AgentRunResult] = {}
        outcomes: Dict[str, PipelineStepOutcome] = {}
        settled: set[str] = set()
        launched: set[str] = set()
        context = PipelineRunContext(outputs=outputs, results=results)

        def is_ready(step: PipelineStep) -> bool:
            return all(dep in settled for dep in (step.depends_on or []))

        def failed_deps(step: PipelineStep) -> List[str]:
            return [dep for dep in (step.depends_on or []) if outcomes.get(dep) is None or outcomes[dep].status != "succeeded"]

        async def resolve_input(step: PipelineStep) -> Any:
            if callable(step.input):
                result = step.input(context)
                return await result if asyncio.iscoroutine(result) else result
            return step.input

        async def run_one_step(step: PipelineStep) -> None:
            failed = failed_deps(step)
            if failed:
                outcomes[step.id] = PipelineStepOutcome(id=step.id, status="skipped", attempts=0, skipped_reason=f"dependency failed or was skipped: {', '.join(failed)}")
                settled.add(step.id)
                return

            is_valid = step.is_valid or _default_is_valid
            max_attempts = 1 + (step.retries if step.retries is not None else (options.retries or 0))
            last: Optional[AgentRunResult] = None
            for attempt in range(1, max_attempts + 1):
                resolved_input = await resolve_input(step)
                last = await self._agent_runtime.run_inline_agent(step.agent, resolved_input)
                if options.on_step:
                    options.on_step(step.id, last, attempt)
                if last.status == "succeeded" and is_valid(last.output):
                    outputs[step.id] = last.output
                    results[step.id] = last
                    outcomes[step.id] = PipelineStepOutcome(id=step.id, status="succeeded", result=last, attempts=attempt)
                    settled.add(step.id)
                    return
                if attempt == max_attempts:
                    results[step.id] = last
                    outcomes[step.id] = PipelineStepOutcome(id=step.id, status="failed", result=last, attempts=attempt)
                    settled.add(step.id)
                    return

        done = asyncio.Event()

        def try_launch() -> None:
            for step in steps:
                if step.id in launched or not is_ready(step):
                    continue
                launched.add(step.id)
                asyncio.create_task(_run_and_continue(step))

        async def _run_and_continue(step: PipelineStep) -> None:
            async with pool:
                await run_one_step(step)
            try_launch()
            if len(settled) == len(steps):
                done.set()

        try_launch()
        await done.wait()

        return PipelineResult(
            outputs=outputs,
            results=results,
            steps=[outcomes[step.id] for step in steps],
            duration_ms=int((time.monotonic() - started) * 1000),
        )
