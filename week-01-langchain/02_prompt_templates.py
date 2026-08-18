"""Day 1, step 2: prompt templates.

ChatPromptTemplate builds a role-tagged conversation with the changing parts
as {variables}; .invoke(dict) renders real messages you can inspect BEFORE
anything is sent to the model.
"""

from langchain_core.prompts import ChatPromptTemplate

from llm import get_llm

llm = get_llm()

prompt = ChatPromptTemplate(
    [
        ("system", "You are a terse assistant. Answer in a single short line, in {language}."),
        ("human", "What does {acronym} stand for?"),
    ]
)

rendered = prompt.invoke({"language": "English", "acronym": "LCEL"})
print("rendered messages (nothing sent yet):")
for m in rendered.to_messages():
    print(f"   [{m.type}] {m.content}")

print(" model answer ->", llm.invoke(rendered).content)
