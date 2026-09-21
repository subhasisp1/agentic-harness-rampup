import asyncio

from langgraph.checkpoint.memory import InMemorySaver

from graph import MAX_TOOL_CALLS
from harness import build_toolbelt, run_turn
from otel import export
from tracing import Tracer, print_tree

QUESTION = ("According to my notes, how many GB of VRAM headroom were left on the GPU "
            "with the weights and the 32k-context KV cache loaded?")


async def main():
    toolbelt, _ = await build_toolbelt(InMemorySaver())
    tracer = Tracer()
    config = {"configurable": {"thread_id": "otel"}, "callbacks": [tracer]}
    payload = {"messages": [("human", QUESTION)], "max_tool_calls": MAX_TOOL_CALLS}
    await run_turn(toolbelt, payload, config)

    print_tree(tracer, title="my own tree, as on Tuesday")
    print(f"\nthe same {len(tracer.order)} spans as OpenTelemetry:\n")
    export(tracer)


asyncio.run(main())
