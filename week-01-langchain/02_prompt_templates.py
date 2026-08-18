"""Day 1, step 2: message roles, then prompt templates.

Part A sends the three roles by hand: system (behavior), human (the user),
ai (a prior model turn — here faked to steer the format of the next answer).

Part B builds the same shape with ChatPromptTemplate so the changing parts
become {variables}; .invoke(dict) renders real messages you can inspect
BEFORE anything is sent to the model.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from llm import get_llm

llm = get_llm()

# --- Part A: explicit messages, one per role ---
messages = [
    SystemMessage("You are a terse assistant. Answer in a single short line."),
    HumanMessage("What does OCR stand for?"),
    AIMessage("OCR = Optical Character Recognition."),  # fake prior turn: sets the X = ... format
    HumanMessage("And VLM?"),
]
print("A) explicit roles ->", llm.invoke(messages).content)

# --- Part B: the same idea as a reusable template ---
prompt = ChatPromptTemplate(
    [
        ("system", "You are a terse assistant. Answer in a single short line, in {language}."),
        ("human", "What does {acronym} stand for?"),
    ]
)

rendered = prompt.invoke({"language": "English", "acronym": "LCEL"})
print("\nB) rendered messages (nothing sent yet):")
for m in rendered.to_messages():
    print(f"   [{m.type}] {m.content}")

print("B) model answer ->", llm.invoke(rendered).content)
