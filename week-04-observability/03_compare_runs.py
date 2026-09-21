import asyncio
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from graph import MAX_TOOL_CALLS
from harness import build_toolbelt, run_turn
from trace_store import recent, save, slowest, totals
from tracing import Tracer

DB = Path(__file__).with_name("traces.sqlite")
QUESTION = ("According to my notes, how many GB of VRAM headroom were left on the GPU "
            "with the weights and the 32k-context KV cache loaded?")


async def main():
    toolbelt, _ = await build_toolbelt(InMemorySaver())
    for attempt in (1, 2):
        tracer = Tracer()
        config = {"configurable": {"thread_id": f"store-{attempt}"}, "callbacks": [tracer]}
        payload = {"messages": [("human", QUESTION)], "max_tool_calls": MAX_TOOL_CALLS}
        await run_turn(toolbelt, payload, config)
        save(tracer, DB, f"notes question, run {attempt}")
        print(f"run {attempt}: ${tracer.cost():.6f}, {tracer.seconds():.2f} s, "
              f"{sum(tracer.tokens())} tokens")

    print(f"\nrecent runs in {DB.name}:")
    for label, started, tokens_in, tokens_out, cost, seconds in recent(DB):
        print(f"  {started}  {label:<24} in {tokens_in:>6} out {tokens_out:>5} "
              f"${cost:.6f} {seconds:>6.2f} s")

    runs, tokens_in, tokens_out, cost = totals(DB)
    print(f"\n{runs} runs on record, {tokens_in + tokens_out} tokens, ${cost:.6f} all told.")

    print("\nslowest spans ever recorded:")
    for name, kind, ms in slowest(DB):
        print(f"  {name:<24} [{kind}] {ms:>8.1f} ms")


asyncio.run(main())
