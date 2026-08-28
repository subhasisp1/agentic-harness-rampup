"""Day 3: model proposes, human approves, code runs.

The tool node now calls interrupt() before running anything. The graph
stops, the checkpointer saves the state, and invoke() returns with an
__interrupt__ entry that describes the proposed call. This script prints
it, asks y/n, and resumes with Command(resume=answer). A rejected call gets
a ToolMessage saying so, and the model answers with what it has. One
question, two tool calls, two approvals. InMemorySaver means the pause
lives only inside this process; Day 4 makes it survive a restart.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from graph import MAX_TOOL_CALLS, build_graph

toolbelt = build_graph(InMemorySaver())
config = {"configurable": {"thread_id": "demo"}}


def approve(proposal):
    args = ", ".join(f"{k}={v!r}" for k, v in proposal["args"].items())
    try:
        answer = input(f"Run {proposal['tool']}({args})? [y/n] ")
    except EOFError:
        return False
    return answer.strip().lower().startswith("y")


question = ("According to my notes, how many GB of VRAM headroom were left on the GPU "
            "with the weights and the 32k-context KV cache loaded?")
payload = {"messages": [("human", question)], "max_tool_calls": MAX_TOOL_CALLS}

result = toolbelt.invoke(payload, config)
while "__interrupt__" in result:
    print("paused, next node:", toolbelt.get_state(config).next)
    result = toolbelt.invoke(Command(resume=approve(result["__interrupt__"][0].value)), config)

for message in result["messages"][1:]:
    message.pretty_print()
