# Agentic Harness Ramp-Up

A weekly learn-in-public program working toward building an **agentic AI harness**:
prompt the model → let it call tools in a loop → give it memory → constrain it.

## Layout

```
week-01-langchain/   Week 1 — "Toolbelt": LangChain fundamentals ending in a simple agent
week-02-langgraph/   Week 2: the same Toolbelt with the agent loop written by hand in LangGraph
blog/                daily posts (Markdown, one folder per week, one file per day)
```

## Week 1 — Toolbelt (LangChain)

A small CLI assistant that chats, returns validated JSON when asked, and calls a few
local tools (calculator, date/time, notes lookup). The LLM is an Anthropic model
(`anthropic/claude-haiku-4.5`) reached through [OpenRouter](https://openrouter.ai)'s
OpenAI-compatible API — set `OPENROUTER_API_KEY` before running. The model lives in
one place (`week-01-langchain/llm.py`) so swapping it is a one-line change.

- **Tue** — foundations: env, first chat-model call, prompt templates + message roles, first LCEL chain
- **Wed** — structured output (Pydantic) + first tool call by hand
- **Thu** — the agent loop (`create_agent`, the LangChain 1.x API)
- **Fri** — conversation memory (checkpointer), the Toolbelt CLI, wrap-up post

### Running week 1

```bash
cd week-01-langchain
python3 -m venv .venv && .venv/bin/pip install -e .
export OPENROUTER_API_KEY=sk-or-...        # https://openrouter.ai/keys
.venv/bin/python 01_first_chat.py         # Tue: smallest chat-model call
.venv/bin/python 02_prompt_templates.py   # Tue: templates, rendered before sending
.venv/bin/python 03_lcel_chain.py         # Tue: prompt | model | parser
.venv/bin/python 04_structured_output.py  # Wed: validated Pydantic output
.venv/bin/python 05_manual_tool_call.py   # Wed: one tool call executed by hand
.venv/bin/python 06_agent_loop.py         # Thu: create_agent, traced loop
.venv/bin/python 07_toolbelt.py           # Fri: the deliverable — interactive CLI with memory
```

## Week 2: Toolbelt on LangGraph

The same assistant with the agent loop written by hand as a LangGraph `StateGraph`: the state,
the nodes, the edges. On top of it: a router (plain chat, tools, or a validated JSON answer), a
tool-call budget kept in the state, approval before any tool runs (`interrupt`), and memory in
SQLite so a conversation continues after a restart. Same model and `llm.py` as week 1.

- **Tue**: the graph by hand: message-list state, a model node, a tool node, one conditional edge
- **Wed**: state fields and branching: router node, JSON-answer node, the budget in the state
- **Thu**: human in the loop: `interrupt()` before each tool call, approve or reject, resume
- **Fri**: `SqliteSaver`, `--thread` for several conversations side by side, the Toolbelt CLI

### Running week 2

```bash
cd week-02-langgraph
python3 -m venv .venv && .venv/bin/pip install -e .
export OPENROUTER_API_KEY=sk-or-...
.venv/bin/python 01_graph_by_hand.py         # Tue: the loop create_agent was hiding, traced node by node
.venv/bin/python 02_state_and_branching.py   # Wed: router, JSON node, budget in the state; four runs
.venv/bin/python 03_human_in_the_loop.py     # Thu: approve or reject each proposed tool call
.venv/bin/python 04_toolbelt.py              # Fri: the deliverable; --thread NAME, --budget N
```

`graph.py` holds the graph used by 03 and 04, `tools.py` the three tools. Conversations are
stored in `week-02-langgraph/toolbelt.sqlite`, which is not committed.
