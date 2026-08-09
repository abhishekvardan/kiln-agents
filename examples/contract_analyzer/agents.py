"""A Python port of the same contract-analyzer agents from the TS demo
(contract-analyzer/backend/src/agents.ts) — same prompts, same schemas, same model, proving kiln_agents
produces equivalent behavior to the TS SDK it's ported from."""

from typing import List

from pydantic import BaseModel, Field

from kiln_agents import AgentContext, define_agent

MODEL = "llama-3.3-70b-versatile"


class Extraction(BaseModel):
    parties: List[str] = Field(description="Every named party to the contract")
    effective_date: str = Field(description="The effective/start date as written in the text")
    termination_clause: str = Field(description="A one-sentence plain summary of how/when either party can terminate")
    key_obligations: List[str] = Field(description="The main obligations each party has, as short bullet points")
    payment_terms: str = Field(description='A one-sentence summary of payment amounts/schedule, or "Not specified"')


async def _extract(ctx: AgentContext) -> Extraction:
    contract_text = ctx.input["contract_text"]
    return await ctx.ai.object(
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract structured terms from contracts. Only use information actually present in the "
                    "text — if something isn't stated, say so explicitly rather than guessing."
                ),
            },
            {"role": "user", "content": contract_text},
        ],
        schema=Extraction,
    )


extractor = define_agent(name="extractor", provider="groq", model=MODEL, execute=_extract)


class FlaggedClause(BaseModel):
    clause: str
    reason: str


class Risk(BaseModel):
    risk_level: str = Field(description='Overall risk to the party reviewing this contract: "low" | "medium" | "high"')
    flagged_clauses: List[FlaggedClause] = Field(description="Specific clauses worth a second look, and why")
    summary: str = Field(description="A short plain-English summary a non-lawyer could understand")


async def _assess_risk(ctx: AgentContext) -> Risk:
    terms: Extraction = ctx.input["terms"]
    contract_text = ctx.input["contract_text"]
    return await ctx.ai.object(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a contract risk assessor. Given already-extracted terms and the original text, "
                    "flag clauses that are unusual, one-sided, or risky, and summarize the contract in plain "
                    "English for someone who isn't a lawyer."
                ),
            },
            {
                "role": "user",
                "content": f"Extracted terms:\n{terms.model_dump_json(indent=2)}\n\nOriginal contract:\n{contract_text}",
            },
        ],
        schema=Risk,
    )


risk_assessor = define_agent(name="risk-assessor", provider="groq", model=MODEL, execute=_assess_risk)
