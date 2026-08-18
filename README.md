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
local tools (calculator, date/time, notes lookup). No web APIs — the LLM runs locally
via [Ollama](https://ollama.com) (`qwen2.5vl:32b`), or OpenRouter if `OPENROUTER_API_KEY`
is set.

- **Tue** — foundations: env, first chat-model call, prompt templates + message roles, first LCEL chain
- **Wed** — structured output (Pydantic) + first tool call by hand
- **Thu** — the agent loop (`create_tool_calling_agent` + `AgentExecutor`)
- **Fri** — conversation memory, CLI polish, README, demo, wrap-up post

### Running week 1

```bash
cd week-01-langchain
python3 -m venv .venv && .venv/bin/pip install -e .
# make sure ollama is serving: ollama serve   (model: qwen2.5vl:32b)
.venv/bin/python 01_first_chat.py
.venv/bin/python 02_prompt_templates.py
.venv/bin/python 03_lcel_chain.py
```
