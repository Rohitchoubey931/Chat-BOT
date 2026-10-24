"""LLM answer generation (Anthropic or OpenAI) grounded in retrieved context."""
import os
from typing import Dict, List, Optional

from app.config import DEFAULT_MODELS, HISTORY_TURNS, LLM_MODEL, LLM_PROVIDER, MAX_TOKENS

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context excerpts.

Rules:
- If the context does not contain the answer, say you could not find it in the documents. Do not guess.
- Be concise and accurate. Keep numbers, names and dates exactly as written in the context.
- Cite the source for each fact in square brackets, for example [report.pdf, p. 3] or [notes.md]."""


class LLMError(Exception):
    """Raised when the LLM provider is misconfigured or the API call fails."""


def format_context(contexts: List[Dict]) -> str:
    blocks = []
    for i, c in enumerate(contexts, start=1):
        meta = c["metadata"]
        page = f", page {meta['page']}" if meta.get("page") else ""
        blocks.append(f"[{i}] (source: {meta['source']}{page})\n{c['text']}")
    return "\n\n".join(blocks)


def _clean_history(history: Optional[List[Dict]]) -> List[Dict]:
    """Keep the last few turns and make sure the conversation starts with a user message."""
    trimmed = [
        {"role": m["role"], "content": m["content"]}
        for m in (history or [])[-HISTORY_TURNS:]
        if m.get("role") in {"user", "assistant"} and m.get("content")
    ]
    while trimmed and trimmed[0]["role"] != "user":
        trimmed.pop(0)
    return trimmed


def generate_answer(question: str, contexts: List[Dict], history: Optional[List[Dict]] = None) -> str:
    user_message = f"Context:\n{format_context(contexts)}\n\nQuestion: {question}"
    messages = _clean_history(history) + [{"role": "user", "content": user_message}]
    model = LLM_MODEL or DEFAULT_MODELS.get(LLM_PROVIDER, "")

    try:
        if LLM_PROVIDER == "anthropic":
            return _ask_anthropic(model, messages)
        if LLM_PROVIDER == "openai":
            return _ask_openai(model, messages)
    except LLMError:
        raise
    except Exception as exc:  # network errors, invalid key, rate limits, etc.
        raise LLMError(f"{LLM_PROVIDER} request failed: {exc}") from exc

    raise LLMError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'anthropic' or 'openai'.")


def _ask_anthropic(model: str, messages: List[Dict]) -> str:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise LLMError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
    from anthropic import Anthropic

    client = Anthropic()
    response = client.messages.create(
        model=model, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT, messages=messages
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def _ask_openai(model: str, messages: List[Dict]) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        raise LLMError("OPENAI_API_KEY is not set. Add it to your .env file.")
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
    )
    return (response.choices[0].message.content or "").strip()
