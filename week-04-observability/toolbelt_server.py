from datetime import datetime, timezone
from pathlib import Path

from mcp.server import MCPServer

mcp = MCPServer("toolbelt")


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate a plain arithmetic expression, e.g. '31 - (21 + 8)'."""
    return str(eval(expression, {"__builtins__": {}}, {}))


@mcp.tool()
def current_datetime() -> str:
    """Current local date and time."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@mcp.tool()
def read_notes() -> str:
    """Read my local hardware notes file."""
    return Path(__file__).with_name("notes.txt").read_text()


if __name__ == "__main__":
    mcp.run()
