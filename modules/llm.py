import os
from typing import Optional

from langchain_openai import ChatOpenAI


def build_llm(model: str, base_url: Optional[str], api_key: Optional[str]) -> ChatOpenAI:
    if not api_key:
        raise ValueError("Missing API key. Set OPENAI_API_KEY or pass via config.")
    if not base_url:
        raise ValueError("Missing base URL. Set OPENAI_API_BASE or pass via config.")

    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = base_url
    os.environ["OPENAI_BASE_URL"] = base_url

    return ChatOpenAI(model=model, temperature=0)
