"""Day 4: Toolbelt on LangGraph, with memory that survives a restart.

The Day 3 loop as a CLI, with two changes. SqliteSaver replaces
InMemorySaver, so a conversation lives in toolbelt.sqlite next to this file
and continues after a restart. --thread names the conversation, so several
can exist side by side; --budget sets the tool-call budget per turn. If the
last session ended in the middle of an approval, the paused run is finished
first, so the history never carries a tool call without a result. Exit with
'exit', 'quit' or Ctrl-D.
"""

import argparse
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from graph import MAX_TOOL_CALLS, build_graph

parser = argparse.ArgumentParser(description="Toolbelt: a CLI assistant on a hand-written graph.")
parser.add_argument("--thread", default="cli", help="conversation name (default: cli)")
parser.add_argument(
    "--budget", type=int, default=MAX_TOOL_CALLS, help="tool calls allowed per turn"
)
args = parser.parse_args()
config = {"configurable": {"thread_id": args.thread}}
DB = Path(__file__).with_name("toolbelt.sqlite")


def approve(proposal):
    args = ", ".join(f"{k}={v!r}" for k, v in proposal["args"].items())
    try:
        answer = input(f"Run {proposal['tool']}({args})? [y/n] ")
    except EOFError:
        return False
    return answer.strip().lower().startswith("y")


def run(toolbelt, payload):
    result = toolbelt.invoke(payload, config)
    while "__interrupt__" in result:
        result = toolbelt.invoke(Command(resume=approve(result["__interrupt__"][0].value)), config)
    return result


with SqliteSaver.from_conn_string(str(DB)) as saver:
    toolbelt = build_graph(saver)
    saved = toolbelt.get_state(config)
    on_record = len(saved.values.get("messages", []))
    print(f"Toolbelt, thread '{args.thread}': {on_record} messages on record.")
    print("'exit' or Ctrl-D to quit.")
    if saved.next:
        print("Finishing the run that was paused last time.")
        interrupts = saved.tasks[0].interrupts
        payload = Command(resume=approve(interrupts[0].value)) if interrupts else None
        print(run(toolbelt, payload)["messages"][-1].content)
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if line in ("exit", "quit"):
            break
        if not line:
            continue
        result = run(toolbelt, {"messages": [("human", line)], "max_tool_calls": args.budget})
        print(result["messages"][-1].content)
