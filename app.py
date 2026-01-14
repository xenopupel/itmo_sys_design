import os
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException

from modules.inference import score_call
from modules.llm import build_llm
from modules.schema import load_criteria


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


app = FastAPI(title="QA Scoring Service")

CRITERIA_PATH = Path(_require_env("CRITERIA_PATH"))
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss_20b")
BASE_URL = _require_env("OPENAI_API_BASE")
API_KEY = _require_env("OPENAI_API_KEY")

CRITERIA = load_criteria(CRITERIA_PATH)
LLM = build_llm(model=LLM_MODEL, base_url=BASE_URL, api_key=API_KEY)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/score")
def score(payload: Dict[str, Any]) -> Dict[str, Any]:
    calls: List[Dict[str, Any]] = payload.get("calls") or []
    if not calls:
        raise HTTPException(status_code=400, detail="Payload must include non-empty 'calls'")

    results = []
    for idx, call in enumerate(calls, start=1):
        call_id = str(call.get("call_id", call.get("file_name", f"call_{idx}")))
        try:
            scored = score_call(LLM, call, CRITERIA)
        except Exception as exc:
            scored = [
                {
                    "criterion_id": "",
                    "score": None,
                    "explanation": f"error: {exc}",
                }
            ]
        results.append(
            {
                "call_id": call_id,
                "checklist_type": call.get("checklist_type", ""),
                "results": scored,
            }
        )

    return {"results": results}
