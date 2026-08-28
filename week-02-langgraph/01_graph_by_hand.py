"""Day 1: the graph by hand.

Last week create_agent returned a compiled LangGraph. This is that graph written
out: the state is the message list, one node calls the model, one node runs the
tools the model asked for, and one conditional edge decides between running
tools and ending. Same tools and question as Week 1 Day 3, so the answer
(2 GB of headroom) should match. Prints the graph as a Mermaid diagram, then
the trace node by node.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from llm import get_llm


@tool
def calculator(expression: str) -> str:
    """Evaluate a plain arithmetic expression, e.g. '31 - (21 + 8)'."""
    return str(eval(expression, {"__builtins__": {}}, {}))


@tool
def current_datetime() -> str:
    """Current local date and time."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@tool
def read_notes() -> str:
    """Read my local hardware notes file."""
    return Path(__file__).with_name("notes.txt").read_text()


TOOLS = {t.name: t for t in (calculator, current_datetime, read_notes)}
SYSTEM = SystemMessage(
    "Take facts from the tools, and do all arithmetic with the calculator. Do not guess numbers."
)
llm = get_llm().bind_tools(list(TOOLS.values()))


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def call_model(state: State) -> State:
    return {"messages": [llm.invoke([SYSTEM, *state["messages"]])]}


def run_tools(state: State) -> State:
    results = []
    for call in state["messages"][-1].tool_calls:
        output = TOOLS[call["name"]].invoke(call["args"])
        results.append(ToolMessage(content=output, tool_call_id=call["id"]))
    return {"messages": results}


def route(state: State) -> str:
    return "tools" if state["messages"][-1].tool_calls else END


builder = StateGraph(State)
builder.add_node("model", call_model)
builder.add_node("tools", run_tools)
builder.add_edge(START, "model")
builder.add_conditional_edges("model", route, ["tools", END])
builder.add_edge("tools", "model")
graph = builder.compile()

print(graph.get_graph().draw_mermaid())

question = ("According to my notes, how many GB of VRAM headroom were left "
            "on the GPU with the weights and the 32k-context KV cache loaded?")

config = {"recursion_limit": 10}
for update in graph.stream({"messages": [("human", question)]}, config, stream_mode="updates"):
    for node, out in update.items():
        print(f"--- node: {node} ---")
        for message in out["messages"]:
            message.pretty_print()
