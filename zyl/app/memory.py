from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

model = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.7)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a friendly AI Assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

# chain = prompt | model
chain = LLMChain(llm=model, prompt=prompt, memory=memory, verbose=True)

msg1 = {"input": "Hello"}

resp1 = chain.invoke(msg1)

print(resp1)

msg2 = {"input": "What is Langchain?"}

resp2 = chain.invoke(msg2)

print(resp2)

msg3 = {"input": "My name is MAS"}

resp3 = chain.invoke(msg3)
print(resp3)

msg4 = {"input": "What is my name"}

resp4 = chain.invoke(msg4)
print(resp4)
