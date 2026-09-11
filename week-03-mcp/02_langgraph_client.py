import asyncio
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from graph import MAX_TOOL_CALLS, build_graph

HERE = Path(__file__).resolve().parent
SERVERS = {
    "toolbelt": {
        "transport": "stdio",
        "command": str(HERE / ".venv" / "bin" / "python"),
        "args": [str(HERE / "toolbelt_server.py")],
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
    print("discovered over MCP:", ", ".join(sorted(t.name for t in tools)))
    toolbelt = build_graph(InMemorySaver(), tools)
    question = ("According to my notes, how many GB of VRAM headroom were left on the GPU "
                "with the weights and the 32k-context KV cache loaded?")
    payload = {"messages": [("human", question)], "max_tool_calls": MAX_TOOL_CALLS}
    result = await toolbelt.ainvoke(payload, config)
    while "__interrupt__" in result:
        print("paused, next node:", toolbelt.get_state(config).next)
        answer = approve(result["__interrupt__"][0].value)
        result = await toolbelt.ainvoke(Command(resume=answer), config)
    for message in result["messages"][1:]:
        message.pretty_print()


asyncio.run(main())
