from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.2,  # The lower the temperature, the more coherent the response
)

response = llm.stream("What is the meaning of life?")

# print(response)

for chunk in response:
    print(chunk.content, end="", flush=True)
