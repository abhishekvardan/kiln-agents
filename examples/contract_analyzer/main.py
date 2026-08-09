"""Same API contract as contract-analyzer/backend (Node+kiln) and backend2 (Node, no kiln) in the main
agentCLI repo — POST /api/analyze with {"contractText": "..."} returns the same {extraction, risk,
durationMs, steps} shape. This is the Python-native kiln_agents version: same two-agent pipeline, same
kiln.orchestrate() call, different language entirely."""

import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from kiln_agents import PipelineRunContext, PipelineStep, create_kiln

from agents import extractor, risk_assessor

load_dotenv()

if not os.environ.get("GROQ_API_KEY"):
    raise SystemExit("Missing GROQ_API_KEY. Copy .env.example to .env and fill it in, then restart.")

app = FastAPI(title="contract-analyzer (kiln_agents)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

kiln = create_kiln()


class AnalyzeRequest(BaseModel):
    contractText: str


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.post("/api/analyze")
async def analyze(body: AnalyzeRequest):
    contract_text = body.contractText
    if not contract_text or len(contract_text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Paste more contract text (at least a couple of sentences).")

    started = time.monotonic()
    result = await kiln.orchestrate(
        [
            PipelineStep(id="extractor", agent=extractor, input={"contract_text": contract_text}, retries=1),
            PipelineStep(
                id="risk-assessor",
                agent=risk_assessor,
                depends_on=["extractor"],
                retries=1,
                input=lambda ctx: {"terms": ctx.outputs["extractor"], "contract_text": contract_text},
            ),
        ]
    )

    extractor_step = next((s for s in result.steps if s.id == "extractor"), None)
    risk_step = next((s for s in result.steps if s.id == "risk-assessor"), None)

    if not extractor_step or extractor_step.status != "succeeded" or not risk_step or risk_step.status != "succeeded":
        raise HTTPException(status_code=502, detail={"error": "Analysis failed.", "steps": [s.__dict__ for s in result.steps]})

    return {
        "extraction": result.outputs["extractor"].model_dump(by_alias=False),
        "risk": result.outputs["risk-assessor"].model_dump(by_alias=False),
        "durationMs": int((time.monotonic() - started) * 1000),
        "steps": [{"id": s.id, "status": s.status, "attempts": s.attempts} for s in result.steps],
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "4702"))
    print(f"contract-analyzer backend3 (kiln_agents, Python) on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
