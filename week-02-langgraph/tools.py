"""The Toolbelt's three local tools, shared by the week-2 scripts.

Same tools as week 1: a toy calculator (eval with builtins stripped), the
current local date and time, and the hardware notes file. TOOLS maps each
tool's name to the tool, which is how the tool node looks up what the
model asked for.
"""

from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool


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
