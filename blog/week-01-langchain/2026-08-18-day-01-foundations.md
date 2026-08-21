# Day 1: Prompts, models, and my first chain

*Agentic-harness ramp-up, Week 1 ("Toolbelt"), Tuesday.*

The whole program points at one thing: an agentic harness. Prompt a model,
let it call tools in a loop, give it memory, keep it constrained. Week 1 is
LangChain, and today was the bottom rung: talk to a model, control the
prompt, chain the pieces.

Setup: one repo for the ramp-up (a folder per week),
a venv, and **`claude-haiku-4.5` via OpenRouter**. OpenRouter speaks the
OpenAI wire format, so `ChatOpenAI` works as-is with a different `base_url`:

```python
# llm.py: the model lives in ONE place
def get_llm(temperature=0):
    return ChatOpenAI(
        model="anthropic/claude-haiku-4.5",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=temperature,
    )
```

(Note: I started with a local Ollama model. It worked, after teaching me
that Ollama's default 32k context adds an 8 GB KV cache and OOM-kills the
loader on a 31 GiB card. Then I swapped to OpenRouter: one env var instead
of a local model. The local setup lives on in `git log`.)

## Messages in, message out

```python
reply = get_llm().invoke("In one sentence: what is an agentic AI harness?")
```

What comes back isn't a string. It's an `AIMessage` with content plus token
usage. A bare string is just shorthand for one human message; the model's
real interface is a **list of role-tagged messages** (`system`, `human`,
`ai`). That shape matters later: it's exactly what an agent's
action/observation loop will need.

Favorite trick of the day: append a *fake* `ai` turn and the model copies
its format:

```python
HumanMessage("What does OCR stand for?"),
AIMessage("OCR = Optical Character Recognition."),  # fake, sets the format
HumanMessage("And VLM?"),
# -> "VLM = Vision-Language Model."
```

## Templates, then a chain

Hard-coded messages don't reuse, so `ChatPromptTemplate` with `{variables}`.
The underrated part: I can render and *inspect* the exact messages before
I send anything. Then LCEL composes it all with `|`:

```python
chain = prompt | llm | StrOutputParser()
chain.invoke({"topic": "prompt templates", "n_sentences": 1})
```

Every stage speaks `.invoke()`: dict → messages → `AIMessage` → string. The
chain is a value: I build it once and call it with anything. It's the first
harness primitive: a typed pipeline.

## Things that took time

- `pip install langchain` now lands **1.x**, and Thursday's planned
  `AgentExecutor` is gone; it's `create_agent` now. I adjusted the plan.
- I asked what LCEL stands for, and the local 32B model said *"Low-Cost
  Embedded Linux."* (It's the LangChain Expression Language.) Fluent ≠
  correct; that's tomorrow's argument for structured output.
- Context length is memory: the KV-cache OOM above generalizes well beyond
  Ollama.

**Tomorrow:** Pydantic-validated output, and the first tool call, which I
run by hand: the model asks, I execute, I feed the result back. One turn of
the harness loop.
