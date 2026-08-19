"""Day 2, step 1: structured output — a validated object, not vibes.

llm.with_structured_output(PydanticModel) makes the model fill a typed
schema and returns a validated Pydantic instance: wrong shape = error,
not silently trusted text. Motivated by Day 1, when a model swore LCEL
means "Low-Cost Embedded Linux".
"""

from pydantic import BaseModel, Field

from llm import get_llm


class Acronym(BaseModel):
    """One acronym expansion, with the model's own confidence."""

    acronym: str = Field(description="the acronym exactly as asked")
    expansion: str = Field(description="what it stands for")
    domain: str = Field(description="the field it belongs to, 2-4 words")
    confidence: float = Field(ge=0, le=1, description="0-1, how sure the model is")


llm = get_llm().with_structured_output(Acronym)

for acronym in ["LCEL", "IISER"]:
    out = llm.invoke(f"What does {acronym} stand for?")
    print(f"{out!r}  (type: {type(out).__name__})")
