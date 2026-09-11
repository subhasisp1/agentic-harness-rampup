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
