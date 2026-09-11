from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from llm import get_llm

MAX_TOOL_CALLS = 3
POLICY = {
    "calculator": "auto",
    "current_datetime": "auto",
    "read_notes": "auto",
    "list_files": "auto",
    "read_file": "auto",
    "write_file": "ask",
}
DEFAULT_ACTION = "ask"  # tools the policy does not name must ask
SYSTEM = SystemMessage(
    "You are Toolbelt, a terse assistant. When tools are available, take facts from them "
    "and do all arithmetic with the calculator. Do not guess numbers."
)
ROUTER_SYSTEM = SystemMessage(
    "Classify the user's message. chat: conversation, or a question you can answer from "
    "general knowledge. tools: it needs my local notes file, arithmetic, or the current date "
    "or time, or reading, writing or listing my sandbox files, or fetching a web page. "
    "json: the user asks for the answer as JSON. Prefer tools over json when the "
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
    approvals: list


llm = get_llm()
router_llm = llm.with_structured_output(Route)
json_llm = llm.with_structured_output(Answer)
llm_with_tools = None  # bound in build_graph, once the tools are discovered
TOOLS: dict = {}


async def router(state: State) -> State:
    decision = await router_llm.ainvoke([ROUTER_SYSTEM, state["messages"][-1]])
    return {"mode": decision.mode, "tool_calls_made": 0}


async def chat(state: State) -> State:
    return {"messages": [await llm.ainvoke([SYSTEM, *state["messages"]])]}


async def call_model(state: State) -> State:
    return {"messages": [await llm_with_tools.ainvoke([SYSTEM, *state["messages"]])]}


def approve_tools(state: State) -> State:
    calls = state["messages"][-1].tool_calls
    approvals = []
    for call in calls:
        if POLICY.get(call["name"], DEFAULT_ACTION) == "auto":
            approvals.append(True)
        else:
            approvals.append(interrupt({"tool": call["name"], "args": call["args"]}))
    return {"approvals": approvals}


async def run_tools(state: State) -> State:
    calls = state["messages"][-1].tool_calls
    results = []
    for call, ok in zip(calls, state["approvals"], strict=True):
        if ok:
            output = await TOOLS[call["name"]].ainvoke(call["args"])
            if isinstance(output, list):  # MCP results are content blocks; keep the text
                output = "".join(b["text"] for b in output if b.get("type") == "text")
        else:
            output = "Rejected by the user; not run."
        results.append(ToolMessage(content=output, tool_call_id=call["id"]))
    ran = sum(1 for ok in state["approvals"] if ok)
    return {"messages": results, "tool_calls_made": state["tool_calls_made"] + ran}


async def json_answer(state: State) -> State:
    answer = await json_llm.ainvoke([SYSTEM, *state["messages"]])
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
    return "approve" if state["tool_calls_made"] < budget else "stop"


def build_graph(checkpointer, tools, policy=None):
    global TOOLS, llm_with_tools, POLICY
    if policy is not None:
        POLICY = policy
    TOOLS = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)
    builder = StateGraph(State)
    builder.add_node("router", router)
    builder.add_node("chat", chat)
    builder.add_node("model", call_model)
    builder.add_node("approve", approve_tools)
    builder.add_node("tools", run_tools)
    builder.add_node("json_answer", json_answer)
    builder.add_node("stop", stop)
    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router", by_mode, {"chat": "chat", "tools": "model", "json": "json_answer"}
    )
    builder.add_conditional_edges("model", after_model, ["approve", "stop", END])
    builder.add_edge("approve", "tools")
    builder.add_edge("tools", "model")
    builder.add_edge("chat", END)
    builder.add_edge("json_answer", END)
    builder.add_edge("stop", END)
    return builder.compile(checkpointer=checkpointer)
