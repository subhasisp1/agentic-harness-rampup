"""The Toolbelt graph as a module.
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
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
    tool_calls_made: int
    max_tool_calls: int


llm = get_llm()
llm_with_tools = llm.bind_tools(list(TOOLS.values()))
router_llm = llm.with_structured_output(Route)
json_llm = llm.with_structured_output(Answer)


def router(state: State) -> State:
    decision = router_llm.invoke([ROUTER_SYSTEM, state["messages"][-1]])
    return {"mode": decision.mode, "tool_calls_made": 0}


def chat(state: State) -> State:
    return {"messages": [llm.invoke([SYSTEM, *state["messages"]])]}


def call_model(state: State) -> State:
    return {"messages": [llm_with_tools.invoke([SYSTEM, *state["messages"]])]}


def run_tools(state: State) -> State:
    calls = state["messages"][-1].tool_calls
    approved = [interrupt({"tool": c["name"], "args": c["args"]}) for c in calls]
    results = []
    for call, ok in zip(calls, approved, strict=True):
        if ok:
            output = TOOLS[call["name"]].invoke(call["args"])
        else:
            output = "Rejected by the user; not run."
        results.append(ToolMessage(content=output, tool_call_id=call["id"]))
    ran = sum(1 for ok in approved if ok)
    return {"messages": results, "tool_calls_made": state["tool_calls_made"] + ran}


def json_answer(state: State) -> State:
    answer = json_llm.invoke([SYSTEM, *state["messages"]])
    return {"messages": [AIMessage(content=answer.model_dump_json(indent=2))]}


def stop(state: State) -> State:
    budget = state.get("max_tool_calls", MAX_TOOL_CALLS)
    refusal = f"Not run: this turn's tool-call budget ({budget}) is spent."
    calls = state["messages"][-1].tool_calls
    return {"messages": [ToolMessage(content=refusal, tool_call_id=c["id"]) for c in calls]}


def by_mode(state: State) -> str:
    return state["mode"]


def after_model(state: State) -> str:
    if not state["messages"][-1].tool_calls:
        return END
    budget = state.get("max_tool_calls", MAX_TOOL_CALLS)
    return "tools" if state["tool_calls_made"] < budget else "stop"


def build_graph(checkpointer):
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
    return builder.compile(checkpointer=checkpointer)
