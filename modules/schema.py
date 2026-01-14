from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml  # type: ignore[import-untyped]


@dataclass(frozen=True)
class Criterion:
    cid: str
    name: str
    description: str
    prompt: str


def load_criteria(path: Path) -> List[Criterion]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = data.get("criteria", [])
    criteria: List[Criterion] = []
    for c in raw:
        criteria.append(
            Criterion(
                cid=str(c.get("id", "")).strip(),
                name=str(c.get("name", "")).strip(),
                description=str(c.get("description", "")).strip(),
                prompt=str(c.get("prompt", "")).strip(),
            )
        )
    return criteria
