import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from .io_utils import load_calls, write_csv, write_jsonl
from .llm import build_llm
from .prompt import SYSTEM_PROMPT, build_user_prompt
from .schema import Criterion, load_criteria


def extract_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError(f"Failed to parse JSON from model output:\n{text}")


def normalize_score(value: Any) -> Optional[int]:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    return score if score in (0, 1, 2) else None


def score_call(llm, call: Dict[str, Any], criteria: List[Criterion]) -> List[Dict[str, Any]]:
    user_prompt = build_user_prompt(call, criteria)
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)])
    data = extract_json(str(response.content))

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Model output must be a JSON array")

    results: List[Dict[str, Any]] = []
    for item in data:
        cid = str(item.get("criterion_id", "")).strip()
        results.append(
            {
                "criterion_id": cid,
                "score": normalize_score(item.get("score")),
                "explanation": str(item.get("explanation", "")).strip(),
            }
        )
    return results


def run_batch() -> None:
    input_path = Path(os.environ["INPUT_PATH"])
    criteria_path = Path(os.environ["CRITERIA_PATH"])
    output_jsonl = Path(os.environ["OUTPUT_JSONL_PATH"])
    output_csv = Path(os.environ["OUTPUT_CSV_PATH"])

    model = os.getenv("LLM_MODEL", "gpt-oss_20b")
    base_url = os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY")
    limit = int(os.getenv("LIMIT", "0")) or None

    calls = load_calls(input_path)
    if limit:
        calls = calls[:limit]

    criteria = load_criteria(criteria_path)
    criteria_by_id = {c.cid: c.name for c in criteria}

    llm = build_llm(model=model, base_url=base_url, api_key=api_key)

    jsonl_rows = []
    csv_rows = []

    for idx, call in enumerate(calls, start=1):
        call_id = str(call.get("call_id", call.get("file_name", f"call_{idx}")))
        try:
            scored = score_call(llm, call, criteria)
        except Exception as exc:
            scored = [
                {
                    "criterion_id": "",
                    "score": None,
                    "explanation": f"error: {exc}",
                }
            ]

        jsonl_rows.append(
            {
                "call_id": call_id,
                "checklist_type": call.get("checklist_type", ""),
                "results": scored,
            }
        )

        for item in scored:
            csv_rows.append(
                {
                    "call_id": call_id,
                    "criterion_id": item.get("criterion_id", ""),
                    "criterion_name": criteria_by_id.get(item.get("criterion_id", ""), ""),
                    "model_score": item.get("score"),
                    "model_explanation": item.get("explanation", ""),
                }
            )

        print(f"[{idx}/{len(calls)}] scored call_id={call_id}")

    write_jsonl(output_jsonl, jsonl_rows)
    write_csv(output_csv, csv_rows)
