"""Team is the dynamic multi-agent primitive — ctx.team.ask() reaching another roster member directly,
on top of a shared blackboard. Mirrors kiln's TS team.test.ts."""

import pytest
from pydantic import BaseModel

from conftest import ScriptedProvider, runtime_with
from kiln_agents._errors import KilnError
from kiln_agents.agent import define_agent
from kiln_agents.team import Team


class Answer(BaseModel):
    answer: str


@pytest.mark.asyncio
async def test_a_lead_agent_asks_a_teammate_directly_and_gets_its_raw_output_back():
    runtime, _events = runtime_with(ScriptedProvider(json_responses=[{"answer": "42"}]))
    specialist = define_agent(
        name="specialist", provider="scripted", model="test-model",
        execute=lambda ctx: ctx.ai.object(messages=[{"role": "user", "content": "what is the answer?"}], schema=Answer),
    )
    coordinator = define_agent(name="coordinator", provider="scripted", model="test-model", execute=lambda ctx: ctx.team.ask("specialist", {}))

    team = Team(runtime, [specialist, coordinator])
    result = await team.run("coordinator", {})

    assert result.status == "succeeded"
    assert len(result.asks) == 1
    ask = result.asks[0]
    assert ask.to_agent == "specialist"
    assert ask.output == Answer(answer="42")


@pytest.mark.asyncio
async def test_an_ask_record_carries_the_teammates_own_full_trace():
    runtime, _events = runtime_with(ScriptedProvider(json_responses=[{"answer": "42"}]))
    specialist = define_agent(
        name="specialist", provider="scripted", model="test-model",
        execute=lambda ctx: ctx.ai.object(messages=[{"role": "user", "content": "what is the answer?"}], schema=Answer),
    )
    coordinator = define_agent(name="coordinator", provider="scripted", model="test-model", execute=lambda ctx: ctx.team.ask("specialist", {}))

    team = Team(runtime, [specialist, coordinator])
    result = await team.run("coordinator", {})

    ask = result.asks[0]
    assert [e.type for e in ask.trace] == ["AgentStarted", "LLMCallCompleted", "ObjectAttemptSucceeded", "AgentCompleted"]
    assert [e.type for e in result.trace] == ["AgentStarted", "AgentCompleted"], "the lead made no LLM calls of its own here, just dispatched"


@pytest.mark.asyncio
async def test_asking_an_unknown_teammate_raises_a_clear_error_naming_the_real_roster():
    runtime, _events = runtime_with(ScriptedProvider())
    coordinator = define_agent(name="coordinator", provider="scripted", model="test-model", execute=lambda ctx: ctx.team.ask("nobody", {}))
    team = Team(runtime, [coordinator])

    result = await team.run("coordinator", {})
    assert result.status == "failed"
    assert "nobody" in result.error
    assert "coordinator" in result.error


@pytest.mark.asyncio
async def test_a_failed_ask_fails_the_callers_own_run_unless_caught():
    async def flaky(ctx):
        raise ValueError("specialist broke")

    runtime, _events = runtime_with(ScriptedProvider())
    specialist = define_agent(name="specialist", provider="scripted", model="test-model", execute=flaky)
    coordinator = define_agent(name="coordinator", provider="scripted", model="test-model", execute=lambda ctx: ctx.team.ask("specialist", {}))
    team = Team(runtime, [specialist, coordinator])

    result = await team.run("coordinator", {})
    assert result.status == "failed"
    assert "specialist broke" in result.error
    assert len(result.asks) == 1
    assert result.asks[0].error is not None


@pytest.mark.asyncio
async def test_ctx_team_blackboard_is_shared_across_the_whole_team_run():
    async def writer(ctx):
        await ctx.team.set("note", "left by writer")
        return await ctx.team.ask("reader", {})

    async def reader(ctx):
        return {"saw": await ctx.team.get("note")}

    runtime, _events = runtime_with(ScriptedProvider())
    writer_agent = define_agent(name="writer", provider="scripted", model="test-model", execute=writer)
    reader_agent = define_agent(name="reader", provider="scripted", model="test-model", execute=reader)
    team = Team(runtime, [writer_agent, reader_agent])

    result = await team.run("writer", {})
    assert result.output == {"saw": "left by writer"}


@pytest.mark.asyncio
async def test_max_asks_caps_a_runaway_back_and_forth():
    async def ping(ctx):
        await ctx.team.ask("pong", {})

    async def pong(ctx):
        await ctx.team.ask("ping", {})

    runtime, _events = runtime_with(ScriptedProvider())
    ping_agent = define_agent(name="ping", provider="scripted", model="test-model", execute=ping)
    pong_agent = define_agent(name="pong", provider="scripted", model="test-model", execute=pong)
    team = Team(runtime, [ping_agent, pong_agent])

    result = await team.run("ping", {}, max_asks=4)
    assert result.status == "failed"
    assert "exceeded max_asks" in result.error
    assert len(result.asks) <= 5  # bounded, not infinite


@pytest.mark.asyncio
async def test_ctx_team_ask_outside_a_team_run_raises_a_clear_error():
    runtime, _events = runtime_with(ScriptedProvider())
    lone = define_agent(name="lone", provider="scripted", model="test-model", execute=lambda ctx: ctx.team.ask("anyone", {}))

    result = await runtime.run_inline_agent(lone, {})
    assert result.status == "failed"
    assert "needs a real roster" in result.error


def test_two_agents_on_one_roster_sharing_a_name_is_rejected_up_front():
    runtime, _events = runtime_with(ScriptedProvider())
    a = define_agent(name="dup", provider="scripted", model="m", execute=lambda ctx: None)
    b = define_agent(name="dup", provider="scripted", model="m", execute=lambda ctx: None)
    with pytest.raises(KilnError):
        Team(runtime, [a, b])
