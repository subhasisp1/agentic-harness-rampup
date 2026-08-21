# Day 3: The loop runs itself

*Agentic-harness ramp-up, Week 1 ("Toolbelt"), Thursday.*

Yesterday I ran one tool call by hand: the model requested, my code executed,
I fed the result back, the model answered. Today that cycle goes into a loop
that runs without me. That is the whole difference between "a model with
tools" and an agent.

## create_agent

LangChain 1.x replaced the old `AgentExecutor` with `create_agent`:

```python
agent = create_agent(
    get_llm(),
    [calculator, read_notes],
    system_prompt="Take facts from the tools, and do all arithmetic "
                  "with the calculator. Do not guess numbers.",
)
```

What comes back is a compiled LangGraph state machine with two nodes,
`model` and `tools`. The model node decides: tool call or final answer.
Tool calls route to the tools node and loop back with the result; a plain
answer ends the run. Same cycle as yesterday's manual version, just wired
into a graph.

## Making the loop actually loop

A one-tool question would finish in a single pass, so the test question
forces a sequence: *"According to my notes, how many GB of VRAM headroom
were left on the GPU with the weights and the 32k-context KV cache
loaded?"* The agent has to read the notes file first, then hand the
numbers to the calculator; it can't skip a step.

```python
for step in agent.stream({"messages": [("human", question)]},
                         {"recursion_limit": 10}, stream_mode="values"):
    step["messages"][-1].pretty_print()
```

The streamed trace shows each turn of the loop: the question, a
`read_notes` call, the file content coming back, a `calculator` call with
`31 - (21 + 8)`, the result `2`, and then the final sentence using it.
The numbers are the Day-1 OOM story; now the agent answers it instead
of me.

`recursion_limit` is the guardrail: if the model kept requesting tools
forever, the graph stops after a fixed number of steps instead of burning
tokens. First constraint of the harness, one config key.

## Lessons learnt

- The system prompt does real work. Without "do all arithmetic with the
  calculator", the model happily does the subtraction itself. That is
  correct here, but the habit I want is: facts and math come from tools.
- `stream_mode="values"` emits the full message list each step; printing
  only the last message per step gives a clean trace of the loop.
- The agent input is `{"messages": [...]}`, not a bare string. The state
  the graph carries is a conversation, which is exactly what Friday's
  memory work will build on.

**Tomorrow:** multi-turn memory and a tidy CLI, then the week-1 wrap-up.
