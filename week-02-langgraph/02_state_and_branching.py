"""Day 2: state is a schema, edges are decisions.

Yesterday's state was only the message list. Today it gains fields of its
own: the router's decision, a count of tool calls made, and a budget for
them. A router node runs first and picks one of three paths: plain chat,
yesterday's tool loop, or a validated JSON answer (Week 1 Day 2's structured
output as its own node). The runaway guardrail moves out of recursion_limit
and into the state: once the budget is spent, a stop node refuses further
tool calls and the run ends. Four runs show the four paths.
"""

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from llm import get_llm
from tools import TOOLS

MAX_TOOL_CALLS = 3
SYSTEM = SystemMessage(
    "You are Toolbelt, a terse assistant. When tools are available, take facts from them "
    "and do all arithmetic with the calculator. Do not guess numbers."
)
ROUTER_SYSTEM = SystemMessage(
    "Classify the user's message. chat: conversation, or a question you can answer from "
    "general knowledge. tools: it needs my local notes file, arithmetic, or the current date "
    "or time. json: the user asks for the answer as JSON. Prefer tools over json when the "
    "facts must come from the tools."
)


class Route(BaseModel):
    """The router's decision."""

    mode: Literal["chat", "tools", "json"]


class Answer(BaseModel):
    """A short answer with the model's own confidence."""

    answer: str
    confidence: float = Field(ge=0, le=1, description="0-1, how sure the model is")


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    mode: str
    tool_calls_made: Annotated[int, operator.add]
    max_tool_calls: int


llm = get_llm()
llm_with_tools = llm.bind_tools(list(TOOLS.values()))
router_llm = llm.with_structured_output(Route)
json_llm = llm.with_structured_output(Answer)


def router(state: State) -> State:
    decision = router_llm.invoke([ROUTER_SYSTEM, state["messages"][-1]])
    return {"mode": decision.mode}


def chat(state: State) -> State:
    return {"messages": [llm.invoke([SYSTEM, *state["messages"]])]}


def call_model(state: State) -> State:
    return {"messages": [llm_with_tools.invoke([SYSTEM, *state["messages"]])]}


def run_tools(state: State) -> State:
    results = []
    for call in state["messages"][-1].tool_calls:
        output = TOOLS[call["name"]].invoke(call["args"])
        results.append(ToolMessage(content=output, tool_call_id=call["id"]))
    return {"messages": results, "tool_calls_made": len(results)}


def json_answer(state: State) -> State:
    answer = json_llm.invoke([SYSTEM, *state["messages"]])
    return {"messages": [AIMessage(content=answer.model_dump_json(indent=2))]}


def stop(state: State) -> State:
    budget = state.get("max_tool_calls", MAX_TOOL_CALLS)
    refusal = f"Not run: this run's tool-call budget ({budget}) is spent."
    calls = state["messages"][-1].tool_calls
    return {"messages": [ToolMessage(content=refusal, tool_call_id=c["id"]) for c in calls]}


def by_mode(state: State) -> str:
    return state["mode"]


def after_model(state: State) -> str:
    if not state["messages"][-1].tool_calls:
        return END
    budget = state.get("max_tool_calls", MAX_TOOL_CALLS)
    return "tools" if state["tool_calls_made"] < budget else "stop"


builder = StateGraph(State)
builder.add_node("router", router)
builder.add_node("chat", chat)
builder.add_node("model", call_model)
builder.add_node("tools", run_tools)
builder.add_node("json_answer", json_answer)
builder.add_node("stop", stop)
builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router", by_mode, {"chat": "chat", "tools": "model", "json": "json_answer"}
)
builder.add_conditional_edges("model", after_model, ["tools", "stop", END])
builder.add_edge("tools", "model")
builder.add_edge("chat", END)
builder.add_edge("json_answer", END)
builder.add_edge("stop", END)
graph = builder.compile()

print(graph.get_graph().draw_mermaid())

NOTES_QUESTION = ("According to my notes, how many GB of VRAM headroom were left on the GPU "
                  "with the weights and the 32k-context KV cache loaded?")
RUNS = [
    ("Hello! In one sentence, what can you do?", MAX_TOOL_CALLS),
    (NOTES_QUESTION, MAX_TOOL_CALLS),
    ("What does LCEL stand for? Answer as JSON.", MAX_TOOL_CALLS),
    (NOTES_QUESTION, 1),
]

for question, budget in RUNS:
    print(f"\n=== {question}  (max_tool_calls={budget})")
    path = []
    for kind, data in graph.stream(
        {"messages": [("human", question)], "max_tool_calls": budget},
        stream_mode=["updates", "values"],
    ):
        if kind == "updates":
            path.extend(data)
        else:
            state = data
    print("path:", " -> ".join(path))
    print(f"mode={state['mode']}  tool_calls_made={state['tool_calls_made']}")
    print(state["messages"][-1].content)
