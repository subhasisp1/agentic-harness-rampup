"""Shared LLM factory for week 2 (same as week 1) — one place to swap the model or backend.

The LLM is an Anthropic model reached through OpenRouter's OpenAI-compatible
API, so LangChain's ChatOpenAI works unchanged with a different base_url.
The key comes from $OPENROUTER_API_KEY and is never hard-coded or committed.
"""

import os

from langchain_openai import ChatOpenAI

MODEL = "anthropic/claude-haiku-4.5"  # slug per https://openrouter.ai/models


def get_llm(temperature=0):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit(
            "OPENROUTER_API_KEY is not set. Get a key at https://openrouter.ai/keys "
            "and run: export OPENROUTER_API_KEY=sk-or-..."
        )
    return ChatOpenAI(
        model=MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        temperature=temperature,
    )
