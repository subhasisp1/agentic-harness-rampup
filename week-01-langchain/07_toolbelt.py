"""Day 4: Toolbelt — the week's pieces as one small CLI assistant.

create_agent + three local tools + memory. The agent state is the message
list; checkpointer=InMemorySaver() persists it between invokes under a
thread_id, so a follow-up like "multiply that by 2" refers to the previous
turn and works. Exit with 'exit', 'quit' or Ctrl-D.
"""

from datetime import datetime, timezone
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

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


agent = create_agent(
    get_llm(),
    [calculator, current_datetime, read_notes],
    system_prompt="You are Toolbelt, a terse CLI assistant. Take facts from "
    "the tools and do all arithmetic with the calculator. Do not guess numbers.",
    checkpointer=InMemorySaver(),
)
config = {"configurable": {"thread_id": "cli"}}

print("Toolbelt — week-1 assistant. 'exit' or Ctrl-D to quit.")
while True:
    try:
        line = input("> ").strip()
    except EOFError:
        break
    if line in ("exit", "quit"):
        break
    if not line:
        continue
    result = agent.invoke({"messages": [("human", line)]}, config)
    print(result["messages"][-1].content)
