"""ctx.memory used to be one unscoped dict shared by every run — a real concurrency bug (mirrors the
one found and fixed in kiln's TS runtime). These tests guard the fix: private by default, shareable
only when a caller opts in with session_id."""

import asyncio

import pytest

from kiln_agents import create_kiln, define_agent


def echo_memory_agent():
    async def execute(ctx):
        existing = await ctx.memory.get("k")
        await ctx.memory.set("k", (ctx.input or {}).get("value", existing))
        return await ctx.memory.get("k")

    return define_agent(name="echo", provider="none", model="m", execute=execute)


@pytest.mark.asyncio
async def test_memory_is_private_per_run_even_with_the_same_key():
    kiln = create_kiln()
    agent = echo_memory_agent()

    results = await asyncio.gather(
        kiln.run_inline_agent(agent, {"value": "A"}),
        kiln.run_inline_agent(agent, {"value": "B"}),
    )
    assert {r.output for r in results} == {"A", "B"}, "two concurrent runs using the same key must not see each other's write"


@pytest.mark.asyncio
async def test_memory_is_shared_across_runs_that_share_a_session_id():
    kiln = create_kiln()
    agent = echo_memory_agent()

    first = await kiln._agent_runtime.run_inline_agent(agent, {"value": "A"}, session_id="thread-1")
    second = await kiln._agent_runtime.run_inline_agent(agent, {}, session_id="thread-1")
    assert first.output == "A"
    assert second.output == "A", "a second run in the same session should see the first run's write"
