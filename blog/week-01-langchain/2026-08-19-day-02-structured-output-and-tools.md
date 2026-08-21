# Day 2: Typed answers, and the tool calls

*Agentic-harness ramp-up, Week 1 ("Toolbelt"), Wednesday.*

Yesterday one of the models told me, with total confidence, that LCEL stands for
"Low-Cost Embedded Linux". Today is the fix to that: stop trusting free text,
and make the model's output pass through a schema.

## Structured output: a validated object

```python
class Acronym(BaseModel):
    acronym: str
    expansion: str
    domain: str
    confidence: float = Field(ge=0, le=1)

llm = get_llm().with_structured_output(Acronym)
out = llm.invoke("What does LCEL stand for?")
# Acronym(acronym='LCEL', expansion='LangChain Expression Language', ...)
```

What comes back isn't a string I have to look at. It's a Pydantic
instance. Wrong field, wrong type, confidence of 7? `ValidationError`,
loudly, instead of junk flowing silently downstream. Under the hood
LangChain uses the model's tool-calling machinery to force the shape.

## The first tool call, defined manually

I wanted to see the agent loop with my own eyes before any framework runs
it for me. So: one tool, one turn, no magic.

```python
@tool
def calculator(expression: str) -> str:
    """Evaluate a plain arithmetic expression, e.g. '17 * 23 + 4'."""
    ...

llm = get_llm().bind_tools([calculator])
ai = llm.invoke([HumanMessage("What is 1284 * 736, exactly?")])
print(ai.tool_calls)
# [{'name': 'calculator', 'args': {'expression': '1284 * 736'}, ...}]
```

The part that clicked: the model doesn't *run* anything. It answers with a
request, "please call `calculator` with these args", and then just sits
there. I execute the function in plain Python, wrap the result in a
`ToolMessage`, send the whole conversation back, and *now* it answers with
the real number instead of an LLM's idea of arithmetic.

Model proposes, my code disposes. That division of labor is the whole
safety story of a harness: the model never touches the world directly, it
only fills in arguments for functions I chose to expose.

## Lessons learnt

- `bind_tools` doesn't force a tool call: if I ask something chatty I get
  a normal answer with empty `tool_calls`. Check the list, don't assume.
- The `ToolMessage` needs the `tool_call_id` from the request, or the model
  can't match result to question.
- My calculator is `eval` with builtins stripped. Fine for a demo, and a
  preview of a real harness concern: every tool is attack surface.

**Tomorrow:** I wrap this request→execute→respond cycle in a loop and let
it run itself with `create_agent` (LangChain 1.x retired the old
`AgentExecutor`). That's the day this thing starts deserving the word
"agent".
