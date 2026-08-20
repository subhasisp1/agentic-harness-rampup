"""Day 3: the loop runs itself — create_agent.

Yesterday we executed one tool call by hand. create_agent wraps that same
request -> execute -> respond cycle in a loop (a LangGraph state machine):
the model keeps calling tools until it answers in plain text. The question
needs read_notes THEN calculator, so the loop visibly iterates;
recursion_limit is the guardrail that stops a runaway agent.
"""

from pathlib import Path

from langchain.agents import create_agent
from langchain_core.tools import tool

from llm import get_llm


@tool
def calculator(expression: str) -> str:
    """Evaluate a plain arithmetic expression, e.g. '31 - (21 + 8)'."""
    return str(eval(expression, {"__builtins__": {}}, {}))


@tool
def read_notes() -> str:
    """Read my local hardware notes file."""
    return Path(__file__).with_name("notes.txt").read_text()


agent = create_agent(
    get_llm(),
    [calculator, read_notes],
    system_prompt="Take facts from the tools, and do all arithmetic with the calculator. Do not guess numbers.",
)

question = ("According to my notes, how many GB of VRAM headroom were left "
            "on the GPU with the weights and the 32k-context KV cache loaded?")

for step in agent.stream({"messages": [("human", question)]}, {"recursion_limit": 10}, stream_mode="values"):
    step["messages"][-1].pretty_print()
