# Day 4: Memory, and the week-1 wrap-up

*Agentic-harness ramp-up, Week 1 ("Toolbelt"), Friday.*

Thursday's agent answered one question and forgot it existed. Today it
remembers, and the week's pieces become one small CLI.

## Memory is just persisted state

The agent's state is the message list; I already saw that in
`{"messages": [...]}`. So multi-turn memory is not a new component, it is
the same state persisted between invokes. In LangChain 1.x that is a
checkpointer:

```python
agent = create_agent(
    get_llm(),
    [calculator, current_datetime, read_notes],
    system_prompt="You are Toolbelt, a terse CLI assistant. ...",
    checkpointer=InMemorySaver(),
)
config = {"configurable": {"thread_id": "cli"}}
result = agent.invoke({"messages": [("human", line)]}, config)
```

Every invoke with the same `thread_id` loads the saved conversation,
appends the new turn, and saves again. I ask "what is 31 - (21 + 8)?" and
then "multiply that by 2", and the second answer works because the first
exchange is still in the state. Different `thread_id`, different
conversation. `InMemorySaver` lives in process memory; a database-backed
checkpointer would be the same one-line swap.

## The Toolbelt CLI

`07_toolbelt.py` is the week's deliverable: an `input()` loop around the
agent, three local tools (calculator, current date/time, notes lookup),
one fixed `thread_id` so the session is one continuous conversation.
About 55 lines, and most of them are the tools.

## Week 1 in four lines

The harness framing I started with was: prompt the model, let it call
tools in a loop, give it memory, constrain it. That turned out to be the
literal table of contents:

- **Tue** (prompt the model): messages, templates, a first chain.
- **Wed** (tools): validated output, one tool call I executed by hand.
- **Thu** (the loop): `create_agent` runs the cycle itself, `recursion_limit` constrains it.
- **Fri** (memory): a checkpointer persists the state.

Every layer is small, and each one sits on the previous one's shape: the
message list is the interface all the way down.

## Lessons learnt

- Verify the installed API before writing code. I checked `create_agent`'s
  signature and the `InMemorySaver` import in the venv first; the 1.x
  rename wave makes remembered APIs unreliable.
- A CLI is the cheapest way to feel memory working: type a follow-up that
  only makes sense with context, and watch it resolve.

**Next week:** LangGraph. I rebuild this same loop with explicit state and
branching, one level lower.
