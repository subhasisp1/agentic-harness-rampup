import asyncio

from langgraph.checkpoint.memory import InMemorySaver

from graph import MAX_TOOL_CALLS
from harness import build_toolbelt, run_turn
from tracing import Tracer, print_tree

TURNS = [
    ("chat", "Hello. In one sentence, what are you?"),
    ("tools", "According to my notes, how many GB of VRAM headroom were left on the GPU "
              "with the weights and the 32k-context KV cache loaded?"),
    ("json", "What does LCEL stand for? Answer as JSON."),
]


async def main():
    toolbelt, _ = await build_toolbelt(InMemorySaver())
    rows = []
    for label, question in TURNS:
        tracer = Tracer()
        config = {"configurable": {"thread_id": f"cost-{label}"}, "callbacks": [tracer]}
        payload = {"messages": [("human", question)], "max_tool_calls": MAX_TOOL_CALLS}
        result = await run_turn(toolbelt, payload, config)

        print(f"=== {label}: {question[:60]}")
        print("answer:", str(result["messages"][-1].content)[:100].replace("\n", " "))
        print_tree(tracer)
        print("model calls per node:")
        for node, (calls, t_in, t_out, cost) in sorted(tracer.by_node().items()):
            print(f"  {node:<14} {calls} call  in {t_in:>6}  out {t_out:>5}  ${cost:.6f}")
        tokens_in, tokens_out = tracer.tokens()
        rows.append((label, tokens_in, tokens_out, tracer.cost(), tracer.seconds()))
        print()

    print(f"{'turn':<8} {'in':>8} {'out':>7} {'cost':>11} {'seconds':>9}")
    for label, tokens_in, tokens_out, cost, seconds in rows:
        print(f"{label:<8} {tokens_in:>8} {tokens_out:>7} {cost:>10.6f} {seconds:>9.2f}")
    print(f"\nthree turns cost ${sum(r[3] for r in rows):.6f}.")


asyncio.run(main())
