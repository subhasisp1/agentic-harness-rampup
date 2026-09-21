"""The week-3 harness, loaded from servers.json.

"""

import json
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import Command

from graph import build_graph

HERE = Path(__file__).resolve().parent
SERVERS_JSON = HERE / "servers.json"


def _resolve(part):
    candidate = HERE / part
    return str(candidate) if candidate.exists() else part


def load_config(path=SERVERS_JSON):
    config = json.loads(Path(path).read_text())
    servers = {}
    for name, connection in config["servers"].items():
        connection = dict(connection)
        connection["command"] = _resolve(connection["command"])
        connection["args"] = [_resolve(arg) for arg in connection.get("args", [])]
        servers[name] = connection
    policy = {name: "auto" for name in config.get("policy", {}).get("auto", [])}
    return servers, policy


async def build_toolbelt(checkpointer, path=SERVERS_JSON):
    servers, policy = load_config(path)
    tools = await MultiServerMCPClient(servers).get_tools()
    return build_graph(checkpointer, tools, policy=policy), tools


def approve(proposal):
    arguments = ", ".join(f"{k}={v!r}" for k, v in proposal["args"].items())
    try:
        answer = input(f"Run {proposal['tool']}({arguments})? [y/n] ")
    except EOFError:
        return False
    return answer.strip().lower().startswith("y")


async def run_turn(toolbelt, payload, config, decide=approve):
    """One turn, answering any approval pauses with decide()."""
    result = await toolbelt.ainvoke(payload, config)
    while "__interrupt__" in result:
        answer = decide(result["__interrupt__"][0].value)
        result = await toolbelt.ainvoke(Command(resume=answer), config)
    return result
