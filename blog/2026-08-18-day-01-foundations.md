# Day 1 — Prompts, models, and my first chain (Week 1: LangChain)

*Agentic-harness ramp-up, Week 1 ("Toolbelt"), Tuesday.*

The end goal of this whole program is an agentic AI harness: prompt the model, let it
call tools in a loop, give it memory, constrain it. Week 1 is LangChain, and today was
the bottom rung: talk to a model, control the prompt, chain the pieces.

Setup: one repo for the whole ramp-up (a folder per week), a venv with `langchain` +
`langchain-openai`, and the LLM = **Anthropic's `claude-haiku-4.5` via OpenRouter**.
OpenRouter speaks the OpenAI wire format, so LangChain needs no special integration —
`ChatOpenAI` with a different `base_url`:

```python
# llm.py — the model lives in ONE place; swapping it is a one-line change
from langchain_openai import ChatOpenAI

MODEL = "anthropic/claude-haiku-4.5"

def get_llm(temperature=0):
    return ChatOpenAI(
        model=MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],   # never hard-coded, never committed
        temperature=temperature,
    )
```

(Confession: the first version of today ran on a local Ollama `qwen2.5vl:32b`. It
worked — after a 33.5B model taught me that Ollama's default 32k context adds an 8 GB
KV cache and OOM-kills the loader on a 31 GiB card. We switched to OpenRouter the same
day: one hosted API, stronger model, and the whole "which GPU, which context length"
class of problems disappears for the price of an env var. The local-model option keeps
living in `git log`.)

## 1. A chat model call is messages in, message out

```python
from llm import get_llm

llm = get_llm()
reply = llm.invoke("In one sentence: what is an agentic AI harness?")
```

The thing that comes back is not a string — it's an `AIMessage`, carrying `content`
plus metadata like token usage. `.invoke("...")` with a bare string is just shorthand
for one human message. That framing matters for everything later: a *chat* model's
native interface is a **list of role-tagged messages**, which is exactly the shape an
agent's action/observation loop will need.

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

Honest footnote from the local-model detour: asked what LCEL stands for, the 32B local
model confidently answered *"Low-Cost Embedded Linux."* It's the LangChain Expression
Language. Good early reminder of why the harness will need validation and constraints
around the model — fluent ≠ correct.

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
  (templates and LCEL live in `langchain_core`), but Thursday's plan mentioned
  `create_tool_calling_agent` + `AgentExecutor` — those are gone in 1.x, replaced by
  `create_agent`. Plan adjusted.
- **`StrOutputParser` returns a `TextAccessor`**, not a bare `str` — it *is* a `str`
  subclass, so nothing breaks, but the type surprised me in the printout.
- **Local-model VRAM math** (pre-switch): a 33.5B Q4 model is ~21 GB of weights, and
  the default 32k context added an 8 GB KV cache — `OLLAMA_CONTEXT_LENGTH=8192` was
  the fix. Kept here because the lesson generalizes: context length is memory.

**Tomorrow:** structured output with Pydantic (`.with_structured_output()`) and the
first tool call, run by hand — the model *asks* for a tool, I execute it, feed the
result back. That's the harness loop, unrolled once.
