"""Day 2, step 2: the agent loop unrolled ONCE, by hand.

bind_tools tells the model what it MAY call; it replies with a tool_call
REQUEST (name + args), not a result. We execute the tool in plain Python,
feed a ToolMessage back, and only then does the model answer.
"""

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from llm import get_llm


@tool
def calculator(expression: str) -> str:
    """Evaluate a plain arithmetic expression, e.g. '17 * 23 + 4'."""
    return str(eval(expression, {"__builtins__": {}}, {}))  


llm = get_llm().bind_tools([calculator])

messages = [HumanMessage("What is 1284 * 736, exactly?")]
ai = llm.invoke(messages)
print("model asks :", ai.tool_calls)

call = ai.tool_calls[0]
result = calculator.invoke(call["args"])
print("we execute :", f"{call['name']}({call['args']}) -> {result}")

messages += [ai, ToolMessage(content=result, tool_call_id=call["id"])]
print("final      :", llm.invoke(messages).content)
