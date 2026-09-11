from pathlib import Path

from mcp.server import MCPServer

SANDBOX = Path(__file__).with_name("sandbox").resolve()
mcp = MCPServer("files")


def _inside(relative: str):
    target = (SANDBOX / relative).resolve()
    return target if target.is_relative_to(SANDBOX) else None


@mcp.tool()
def list_files() -> str:
    """List the files in the sandbox folder."""
    names = sorted(p.name for p in SANDBOX.iterdir() if p.is_file())
    return "\n".join(names) or "(empty)"


@mcp.tool()
def read_file(path: str) -> str:
    """Read a text file from the sandbox folder."""
    target = _inside(path)
    if target is None:
        return "Refused: the path leaves the sandbox."
    if not target.is_file():
        return f"No such file: {path}"
    return target.read_text()


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write a text file inside the sandbox folder."""
    target = _inside(path)
    if target is None:
        return "Refused: the path leaves the sandbox."
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Wrote {len(content)} characters to {path}."


if __name__ == "__main__":
    SANDBOX.mkdir(exist_ok=True)
    mcp.run()
