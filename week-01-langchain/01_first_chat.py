"""Day 1, step 1: the smallest possible chat-model call.

A chat model takes messages in and returns an AIMessage — not a bare string.
`.invoke()` with a plain string is shorthand for a single human message.
"""

from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5vl:32b", temperature=0)

reply = llm.invoke("In one sentence: what is an agentic AI harness?")

print("reply text:\n ", reply.content)
print("\nwhat came back is a", type(reply).__name__)
print("token usage:", reply.usage_metadata)
