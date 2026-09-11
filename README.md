# Agentic Harness Ramp-Up

A weekly learn-in-public program working toward building an **agentic AI harness**:
prompt the model → let it call tools in a loop → give it memory → constrain it.

## Layout

```
week-01-langchain/   Week 1 — "Toolbelt": LangChain fundamentals ending in a simple agent
week-02-langgraph/   Week 2: the same Toolbelt with the agent loop written by hand in LangGraph
week-03-mcp/         Week 3: the same Toolbelt with every tool served over MCP
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

## Week 3: Toolbelt over MCP

The same assistant, with the tools moved out of the process. Three MCP servers speak JSON-RPC
over stdio: `toolbelt_server.py` (the week-1 tools), `files_server.py` (list, read, write inside
a `sandbox/` folder), and `mcp-server-fetch`, a reference server installed from PyPI (web access,
not a line of it ours). `servers.json` says which servers to start and which tools may run
without asking; write and fetch ask first. The graph is week 2's, gone async.

Two venvs on purpose. `.venv` runs the servers on the mcp 2.x SDK; `.venv-client` (Python 3.12)
runs the graph with `langchain-mcp-adapters`, which pins `mcp<2`. Client and server are separate
processes, so they share nothing but the protocol; `interrupt` under an async graph is also why
the client needs Python 3.11+ (see `requirements-client.txt`).

- **Mon**: the protocol by hand: an MCP server, then raw JSON-RPC piped into it (no client library)
- **Tue**: the client: tools discovered at startup, the graph and CLI move to asyncio
- **Wed**: a files server, and a per-tool policy table instead of a blanket y/n
- **Thu**: a third-party server joins by configuration only
- **Fri**: `servers.json` + the Toolbelt CLI: threads, budget, policy, SQLite memory

### Running week 3

```bash
cd week-03-mcp
python3 -m venv .venv && .venv/bin/pip install -e .            # server side (mcp 2.x)
uv venv .venv-client --python 3.12                             # client side (needs 3.11+)
uv pip install -p .venv-client/bin/python -r requirements-client.txt
export OPENROUTER_API_KEY=sk-or-...
.venv/bin/python 01_speak_jsonrpc.py                # Mon: the wire by hand; no key needed
.venv-client/bin/python 02_langgraph_client.py      # Tue: discovery + the async graph
.venv-client/bin/python 03_files_and_policy.py      # Wed: silent reads, the write asks
.venv-client/bin/python 04_fetch.py                 # Thu: someone else's server
.venv-client/bin/python 05_toolbelt.py              # Fri: the deliverable; --thread, --budget, --servers
```

Conversations are stored in `week-03-mcp/toolbelt.sqlite`; the file tools only touch
`week-03-mcp/sandbox/`. Neither is committed.
