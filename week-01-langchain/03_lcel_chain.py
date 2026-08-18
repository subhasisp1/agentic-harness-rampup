"""Day 1, step 3: a first LCEL chain — prompt | model | parser.

The | operator composes Runnables into a pipeline: the prompt renders
messages, the model answers, StrOutputParser unwraps AIMessage.content.
The chain is a value — call .invoke() on it with different inputs.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm import get_llm

prompt = ChatPromptTemplate(
    [
        ("system", "You explain things to busy engineers in exactly {n_sentences} sentence(s)."),
        ("human", "Explain {topic}."),
    ]
)
llm = get_llm()

chain = prompt | llm | StrOutputParser()

for topic in ["prompt templates in LangChain", "why chat models take a list of messages"]:
    out = chain.invoke({"topic": topic, "n_sentences": 1})
    print(f"{topic}\n  -> {out}  (type: {type(out).__name__})\n")
