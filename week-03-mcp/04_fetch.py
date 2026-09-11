import asyncio
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from graph import DEFAULT_ACTION, POLICY, build_graph

HERE = Path(__file__).resolve().parent
PY = str(HERE / ".venv" / "bin" / "python")
SERVERS = {
    "toolbelt": {"transport": "stdio", "command": PY, "args": [str(HERE / "toolbelt_server.py")]},
    "files": {"transport": "stdio", "command": PY, "args": [str(HERE / "files_server.py")]},
    "fetch": {
        "transport": "stdio",
        "command": str(HERE / ".venv-client" / "bin" / "python"),
        "args": ["-m", "mcp_server_fetch"],
    },
}
config = {"configurable": {"thread_id": "demo"}}


def approve(proposal):
    args = ", ".join(f"{k}={v!r}" for k, v in proposal["args"].items())
    try:
        answer = input(f"Run {proposal['tool']}({args})? [y/n] ")
    except EOFError:
        return False
    return answer.strip().lower().startswith("y")


async def main():
    tools = await MultiServerMCPClient(SERVERS).get_tools()
    print("discovered over MCP, with policy:")
    for name in sorted(t.name for t in tools):
        print(f"  {name}: {POLICY.get(name, DEFAULT_ACTION)}")
    toolbelt = build_graph(InMemorySaver(), tools)
    question = "Fetch https://example.com and tell me what its main heading says."
    payload = {"messages": [("human", question)], "max_tool_calls": 3}
    result = await toolbelt.ainvoke(payload, config)
    while "__interrupt__" in result:
        answer = approve(result["__interrupt__"][0].value)
        result = await toolbelt.ainvoke(Command(resume=answer), config)
    for message in result["messages"][1:]:
        message.pretty_print()


asyncio.run(main())
