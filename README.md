# Agentic Harness Ramp-Up

A weekly learn-in-public program working toward building an **agentic AI harness**:
prompt the model → let it call tools in a loop → give it memory → constrain it.

Each week: one small app with an agreed library, ~2–4 hrs/day, a short daily blog post,
and something runnable by Friday.

## Layout

```
week-01-langchain/   Week 1 — "Toolbelt": LangChain fundamentals ending in a simple agent
blog/                daily posts (Markdown, one file per day)
prd.json             this week's PRD items with verification steps (Ralph-loop style)
progress.txt         append-only log of completed tasks, decisions, files changed
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
