import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_calls(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        calls = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                calls.append(json.loads(line))
        return calls
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError("Unsupported input format. Use .csv or .jsonl")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
