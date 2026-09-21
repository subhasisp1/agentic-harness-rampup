"""The Toolbelt CLI, with this week's instrumentation behind a flag.

Week 3 shipped this CLI and week 4 measured the harness from one-shot scripts.
This is the two joined. --trace attaches a tracer to every turn, prints what
the turn used and what it cost, and appends the run to traces.sqlite, so the
numbers come from a real conversation instead of a demo. --tree prints the
span tree as well. Without either flag the chat is exactly week 3's.
"""

import argparse
import asyncio
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from graph import MAX_TOOL_CALLS
from harness import SERVERS_JSON, approve, build_toolbelt, run_turn
from trace_store import save
from tracing import Tracer, print_tree

HERE = Path(__file__).resolve().parent
DB = HERE / "toolbelt.sqlite"
TRACES = HERE / "traces.sqlite"

parser = argparse.ArgumentParser(description="Toolbelt: the CLI, with tracing behind a flag.")
parser.add_argument("--thread", default="cli", help="conversation name (default: cli)")
parser.add_argument("--budget", type=int, default=MAX_TOOL_CALLS, help="tool calls per turn")
parser.add_argument("--servers", default=str(SERVERS_JSON), help="server config file")
parser.add_argument("--trace", action="store_true", help="measure every turn and save it")
parser.add_argument("--tree", action="store_true", help="print the span tree too, implies --trace")
args = parser.parse_args()
args.trace = args.trace or args.tree
config = {"configurable": {"thread_id": args.thread}}
session = []


async def turn(toolbelt, payload):
    """One turn, traced when the flag is on."""
    tracer = Tracer() if args.trace else None
    per_turn = dict(config, callbacks=[tracer]) if tracer else config
    return await run_turn(toolbelt, payload, per_turn, approve), tracer


def measured(tracer):
    if tracer is None:
        return
    if args.tree:
        print_tree(tracer)
    tokens_in, tokens_out = tracer.tokens()
    used = ", ".join(tracer.tools_ran()) or "no tools"
    print(f"[{used} | {tokens_in} in + {tokens_out} out | "
          f"${tracer.cost():.6f} | {tracer.seconds():.1f} s]")
    save(tracer, TRACES, f"cli:{args.thread}")
    session.append((tokens_in + tokens_out, tracer.cost(), tracer.seconds()))


async def main():
    async with AsyncSqliteSaver.from_conn_string(str(DB)) as saver:
        toolbelt, tools = await build_toolbelt(saver, Path(args.servers))
        print("tools over MCP:", ", ".join(sorted(t.name for t in tools)))
        saved = await toolbelt.aget_state(config)
        on_record = len(saved.values.get("messages", []))
        tracing = "  tracing on, saving to traces.sqlite." if args.trace else ""
        print(f"Toolbelt, thread '{args.thread}': {on_record} messages on record.{tracing}")
        print("'exit' or Ctrl-D to quit.")
        if saved.next:
            print("Finishing the run that was paused last time.")
            interrupts = saved.tasks[0].interrupts
            pending = Command(resume=approve(interrupts[0].value)) if interrupts else None
            result, tracer = await turn(toolbelt, pending)
            print(result["messages"][-1].content)
            measured(tracer)
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
            result, tracer = await turn(toolbelt, payload)
            print(result["messages"][-1].content)
            measured(tracer)
    if session:
        print(f"\n{len(session)} turns, {sum(s[0] for s in session)} tokens, "
              f"${sum(s[1] for s in session):.6f}, {sum(s[2] for s in session):.1f} s. "
              f"Traces in {TRACES.name}.")


asyncio.run(main())
