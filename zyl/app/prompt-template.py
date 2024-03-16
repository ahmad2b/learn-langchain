from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Instantiate Model
llm = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0.7,
)

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Generate a list of 10 synonyms for the following word. Return the result in a comma-separated list.",
        ),
        ("human", "{input}"),
    ]
)

# Create LLM Chain
chain = prompt | llm

res = chain.invoke({"input": "happy"})

print(res)
