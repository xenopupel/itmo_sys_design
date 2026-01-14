from typing import Any, Dict, List

from .schema import Criterion

SYSTEM_PROMPT = """
Ты — эксперт по контролю качества работы операторов контакт-центра.
Твоя задача — оценивать, насколько оператор выполняет заданные критерии.

Шкала:
0 — критерий не выполнен
1 — критерий выполнен частично
2 — критерий выполнен полностью

Отвечай строго в JSON-формате, без лишнего текста.
""".strip()


def build_user_prompt(call: Dict[str, Any], criteria: List[Criterion]) -> str:
    transcript = str(call.get("transcript_text", "")).strip()
    checklist_type = str(call.get("checklist_type", "")).strip()

    criteria_block = "\n\n".join(
        [f"{c.cid}. {c.name}\nОписание: {c.description}\nИнструкция: {c.prompt}" for c in criteria]
    )

    checklist_line = f"\nТип чек-листа: {checklist_type}\n" if checklist_type else "\n"

    return f"""
Оцени качество звонка по всем критериям ниже.
{checklist_line}
Критерии:
{criteria_block}

Транскрипт:
\"\"\"text
{transcript}
\"\"\"

Верни JSON-массив такого вида:
[
  {{"criterion_id": "1", "score": 0/1/2, "explanation": "кратко"}},
  ...
]
""".strip()
