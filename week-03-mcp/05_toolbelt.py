import argparse
import asyncio
import json
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from graph import MAX_TOOL_CALLS, build_graph

HERE = Path(__file__).resolve().parent
DB = HERE / "toolbelt.sqlite"

parser = argparse.ArgumentParser(description="Toolbelt: the week-2 graph, tools over MCP.")
parser.add_argument("--thread", default="cli", help="conversation name (default: cli)")
parser.add_argument("--budget", type=int, default=MAX_TOOL_CALLS, help="tool calls per turn")
parser.add_argument("--servers", default=str(HERE / "servers.json"), help="server config file")
args = parser.parse_args()
config = {"configurable": {"thread_id": args.thread}}


def resolve(part):
    candidate = HERE / part
    return str(candidate) if candidate.exists() else part


def load_servers(path):
    cfg = json.loads(path.read_text())
    servers = {}
    for name, conn in cfg["servers"].items():
        conn = dict(conn)
        conn["command"] = resolve(conn["command"])
        conn["args"] = [resolve(a) for a in conn.get("args", [])]
        servers[name] = conn
    policy = {name: "auto" for name in cfg.get("policy", {}).get("auto", [])}
    return servers, policy


def approve(proposal):
    arguments = ", ".join(f"{k}={v!r}" for k, v in proposal["args"].items())
    try:
        answer = input(f"Run {proposal['tool']}({arguments})? [y/n] ")
    except EOFError:
        return False
    return answer.strip().lower().startswith("y")


async def run(toolbelt, payload):
    result = await toolbelt.ainvoke(payload, config)
    while "__interrupt__" in result:
        answer = approve(result["__interrupt__"][0].value)
        result = await toolbelt.ainvoke(Command(resume=answer), config)
    return result


async def main():
    servers, policy = load_servers(Path(args.servers))
    tools = await MultiServerMCPClient(servers).get_tools()
    print("tools over MCP:", ", ".join(sorted(t.name for t in tools)))
    async with AsyncSqliteSaver.from_conn_string(str(DB)) as saver:
        toolbelt = build_graph(saver, tools, policy=policy)
        saved = await toolbelt.aget_state(config)
        on_record = len(saved.values.get("messages", []))
        print(f"Toolbelt, thread '{args.thread}': {on_record} messages on record.")
        print("'exit' or Ctrl-D to quit.")
        if saved.next:
            print("Finishing the run that was paused last time.")
            interrupts = saved.tasks[0].interrupts
            payload = Command(resume=approve(interrupts[0].value)) if interrupts else None
            print((await run(toolbelt, payload))["messages"][-1].content)
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if line in ("exit", "quit"):
                break
            if not line:
                continue
            payload = {"messages": [("human", line)], "max_tool_calls": args.budget}
            print((await run(toolbelt, payload))["messages"][-1].content)


asyncio.run(main())
