"""OpenAI-compatible chat client wrapper."""
import json
import re
from typing import Any, Dict, List, Optional

from app.config import settings


def _get_client():
    try:
        from openai import OpenAI  # type: ignore[import-untyped]
        return OpenAI(base_url=settings.chat_url, api_key=settings.openai_api_key)
    except Exception as exc:
        raise RuntimeError(f"openai client unavailable: {exc}")


def chat_complete(messages: List[Dict[str, str]], *, model: Optional[str] = None, temperature: float = 0.0, **kwargs) -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model=model or settings.chat_model,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    return resp.choices[0].message.content or ""


def extract_json(text: str) -> Any:
    """ extracting the first JSON object or array from LLM output."""
    text = text.strip()
    if not text:
        return None
    # code block
    m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if m:
        text = m.group(1).strip()
    # inline JSON object/array
    for pattern in [r"\{[\s\S]*?\}", r"\[[\s\S]*?\]"]:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                continue
    try:
        return json.loads(text)
    except Exception:
        return None
