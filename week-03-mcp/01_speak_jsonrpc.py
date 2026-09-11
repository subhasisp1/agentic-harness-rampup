import json
import subprocess
import sys

server = subprocess.Popen(
    [sys.executable, "toolbelt_server.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
)


def send(message):
    print(f">>> {json.dumps(message)}")
    server.stdin.write(json.dumps(message) + "\n")
    server.stdin.flush()


def receive():
    reply = json.loads(server.stdout.readline())
    print(f"<<< {json.dumps(reply)}\n")
    return reply


send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
    "protocolVersion": "2026-07-28", "capabilities": {},
    "clientInfo": {"name": "speak-jsonrpc", "version": "0.1"}}})
receive()
send({"jsonrpc": "2.0", "method": "notifications/initialized"})

send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
tools = receive()["result"]["tools"]
print("tools on offer:", ", ".join(t["name"] for t in tools), "\n")

send({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
    "name": "calculator", "arguments": {"expression": "31 - (21 + 8)"}}})
print("calculator says:", receive()["result"]["content"][0]["text"], "\n")

send({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
    "name": "no_such_tool", "arguments": {}}})
unknown = receive()["result"]
print("unknown tool -> isError:", unknown["isError"], "-", unknown["content"][0]["text"], "\n")

send({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {
    "name": "calculator", "arguments": {"expression": "31 -"}}})
bad = receive()["result"]
print("bad expression -> isError:", bad["isError"], "-", bad["content"][0]["text"][:80])

server.stdin.close()
server.wait(timeout=5)
