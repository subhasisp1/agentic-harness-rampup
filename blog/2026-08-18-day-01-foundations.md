# Day 1 — Prompts, models, and my first chain (Week 1: LangChain)

*Agentic-harness ramp-up, Week 1 ("Toolbelt"), Tuesday.*

The end goal of this whole program is an agentic AI harness: prompt the model, let it
call tools in a loop, give it memory, constrain it. Week 1 is LangChain, and today was
the bottom rung: talk to a model, control the prompt, chain the pieces.

Setup: one repo for the whole ramp-up (a folder per week), a venv with `langchain` +
`langchain-ollama`, and a fully local LLM — Ollama serving `qwen2.5vl:32b` on my
RTX 5090. No API keys needed today (OpenRouter stays as a fallback via env var).

## 1. A chat model call is messages in, message out

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5vl:32b", temperature=0)
reply = llm.invoke("In one sentence: what is an agentic AI harness?")
```

The thing that comes back is not a string — it's an `AIMessage`, carrying `content`
plus metadata (my call: 31 input tokens, 43 output). `.invoke("...")` with a bare
string is just shorthand for one human message. That framing matters for everything
later: a *chat* model's native interface is a **list of role-tagged messages**, which
is exactly the shape an agent's action/observation loop will need.

## 2. Message roles, then templates

Three roles did all the work today: `system` (behavior), `human` (the user), and `ai`
(a prior model turn). A trick I liked: append a **fake** `AIMessage` as a fabricated
prior turn, and the model copies its format —

```python
messages = [
    SystemMessage("You are a terse assistant. Answer in a single short line."),
    HumanMessage("What does OCR stand for?"),
    AIMessage("OCR = Optical Character Recognition."),   # fake turn, sets the format
    HumanMessage("And VLM?"),
]
llm.invoke(messages)   # -> "VLM = Vision-Language Model."
```

Hard-coded messages don't reuse, so: `ChatPromptTemplate`, with the changing parts as
`{variables}`. The part I hadn't appreciated: the template is itself invokable, and you
can render and *inspect* the exact messages **before** anything is sent —

```python
prompt = ChatPromptTemplate([
    ("system", "You are a terse assistant. Answer in a single short line, in {language}."),
    ("human", "What does {acronym} stand for?"),
])
rendered = prompt.invoke({"language": "English", "acronym": "LCEL"})
```

Honest footnote: asked what LCEL stands for, my local 32B model confidently answered
*"Low-Cost Embedded Linux."* It's the LangChain Expression Language. Good early
reminder of why the harness will need validation and constraints around the model —
fluent ≠ correct, especially for smaller local models.

## 3. The first chain

LCEL (the real one) composes runnables with `|`:

```python
chain = prompt | llm | StrOutputParser()
chain.invoke({"topic": "prompt templates in LangChain", "n_sentences": 1})
```

Each stage has the same `.invoke()` interface: the prompt turns a dict into messages,
the model turns messages into an `AIMessage`, the parser unwraps it to a string. The
chain is a *value* — build it once, call it with different inputs. Squint and it's the
first primitive of a harness: a typed pipeline around the model instead of ad-hoc
string glue.

## Things that bit me

- **LangChain 1.x, not 0.3.** `pip install langchain` now lands 1.3.x. Fine for today
  (templates and LCEL live in `langchain_core`), but Thursday's plan mentions
  `create_tool_calling_agent` + `AgentExecutor` — those are gone in 1.x, replaced by
  `create_agent`. Plan adjusted.
- **`StrOutputParser` returns a `TextAccessor`**, not a bare `str` — it *is* a `str`
  subclass, so nothing breaks, but the type surprised me in the printout.
- **VRAM math.** A 33.5B Q4 model is ~21 GB of weights; Ollama's default 32k context
  added an 8 GB KV cache and the loader got OOM-killed on a 31 GiB card. Fix:
  `OLLAMA_CONTEXT_LENGTH=8192` — plenty for this week, and everything fits on-GPU.

**Tomorrow:** structured output with Pydantic (`.with_structured_output()`) and the
first tool call, run by hand — the model *asks* for a tool, I execute it, feed the
result back. That's the harness loop, unrolled once.
