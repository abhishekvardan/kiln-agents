"""These are the actual proof for kiln-agents' trace system: every LLM call, tool call, and validation
attempt an agent makes must show up in a structured, ordered trace — not just the agent's final output.
Mirrors kiln's TS trace.test.ts."""

from typing import List

import pytest
from pydantic import BaseModel

from conftest import ScriptedProvider, runtime_with
from kiln_agents.agent import define_agent
from kiln_agents.tool import define_tool
from kiln_agents.providers import ChatResult, ToolCallRequest
from kiln_agents.trace import TraceCapture


class Pick(BaseModel):
    pick: str


def trace_types(entries) -> List[str]:
    return [e.type for e in entries]


@pytest.mark.asyncio
async def test_a_successful_object_call_is_fully_traced():
    runtime, events = runtime_with(ScriptedProvider(json_responses=[{"pick": "b"}]))
    agent = define_agent(
        name="picker",
        provider="scripted",
        model="test-model",
        execute=lambda ctx: ctx.ai.object(messages=[{"role": "user", "content": "pick one"}], schema=Pick),
    )

    run_id = "run-object-success"
    trace = TraceCapture(events, run_id)
    result = await runtime.run_inline_agent(agent, {}, run_id=run_id)
    entries = trace.stop()

    assert result.status == "succeeded"
    assert trace_types(entries) == ["AgentStarted", "LLMCallCompleted", "ObjectAttemptSucceeded", "AgentCompleted"]

    llm_call = entries[1]
    assert llm_call.data["kind"] == "object"
    assert llm_call.data["provider"] == "scripted"
    assert llm_call.data["response"] == {"pick": "b"}
    assert isinstance(llm_call.data["duration_ms"], int)

    success = entries[2]
    assert success.data["output"] == {"pick": "b"}


@pytest.mark.asyncio
async def test_a_validation_failure_that_recovers_is_traced_with_the_raw_invalid_output():
    runtime, events = runtime_with(ScriptedProvider(json_responses=[{"wrong": "shape"}, {"pick": "b"}]))
    agent = define_agent(
        name="picker",
        provider="scripted",
        model="test-model",
        execute=lambda ctx: ctx.ai.object(messages=[{"role": "user", "content": "pick one"}], schema=Pick),
    )

    run_id = "run-object-retry"
    trace = TraceCapture(events, run_id)
    result = await runtime.run_inline_agent(agent, {}, run_id=run_id)
    entries = trace.stop()

    assert result.status == "succeeded"
    assert trace_types(entries) == ["AgentStarted", "LLMCallCompleted", "ObjectAttemptFailed", "LLMCallCompleted", "ObjectAttemptSucceeded", "AgentCompleted"]

    failed = entries[2]
    assert failed.data["attempt"] == 1
    assert failed.data["raw_output"] == {"wrong": "shape"}


@pytest.mark.asyncio
async def test_tool_calling_loop_traces_both_the_request_and_the_result():
    async def count_words(input, ctx):
        return {"count": len(input["text"].split())}

    tool = define_tool(name="count_words", description="Counts words.", execute=count_words, parameters={"type": "object", "properties": {"text": {"type": "string"}}})
    provider = ScriptedProvider(
        tool_call_responses=[
            ChatResult(content="", tool_calls=[ToolCallRequest(id="1", name="count_words", arguments={"text": "hi there"})]),
            ChatResult(content="2 words"),
        ]
    )
    runtime, events = runtime_with(provider)
    agent = define_agent(name="counter", provider="scripted", model="test-model", tools=[tool], execute=lambda ctx: ctx.ai.run(messages=[{"role": "user", "content": "count"}]))

    run_id = "run-tool-loop"
    trace = TraceCapture(events, run_id)
    await runtime.run_inline_agent(agent, {}, run_id=run_id)
    entries = trace.stop()

    requested = next(e for e in entries if e.type == "ToolCallRequested")
    completed = next(e for e in entries if e.type == "ToolCallCompleted")
    assert requested.data["tool"] == "count_words"
    assert completed.data["result"] == {"count": 2}
    assert isinstance(completed.data["duration_ms"], int)

    llm_calls = [e for e in entries if e.type == "LLMCallCompleted"]
    assert len(llm_calls) == 2
    assert llm_calls[0].data["kind"] == "tool_call"


@pytest.mark.asyncio
async def test_an_agent_that_raises_is_traced_with_error_and_a_real_traceback():
    async def broken(ctx):
        raise ValueError("deliberately broken")

    runtime, events = runtime_with(ScriptedProvider())
    agent = define_agent(name="broken", provider="scripted", model="test-model", execute=broken)

    run_id = "run-throws"
    trace = TraceCapture(events, run_id)
    await runtime.run_inline_agent(agent, {}, run_id=run_id)
    entries = trace.stop()

    failed = next(e for e in entries if e.type == "AgentFailed")
    assert failed.data["error"] == "deliberately broken"
    assert "deliberately broken" in failed.data["stack"], "a traceback localizes the failure to a real line, not just a message"
