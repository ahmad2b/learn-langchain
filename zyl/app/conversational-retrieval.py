from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder


def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    splitdocs = splitter.split_documents(docs)

    return splitdocs


def create_db(docs):
    embedder = OpenAIEmbeddings()

    vectorStore = FAISS.from_documents(docs, embedding=embedder)
    return vectorStore


def create_chain(vectorStore):
    model = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.4)

    # prompt = ChatPromptTemplate.from_template(
    #     """
    #         Answer the user's question:
    #         Context: {context}
    #         Question: {input}
    #     """
    # )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the user's questions based on the context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt,
    )

    retriever = vectorStore.as_retriever(search_kwargs={"k": 3})
    retrieval_chain = create_retrieval_chain(retriever, chain)

    return retrieval_chain


def process_chat(chain, question, chat_history):
    response = chain.invoke({"input": question, "chat_history": chat_history})

    return response["answer"]


if __name__ == "__main__":
    docs = get_documents_from_web(
        "https://python.langchain.com/docs/expression_language/"
    )
    vectorStore = create_db(docs)
    chain = create_chain(vectorStore)

    chat_history = [
        # HumanMessage(content="Hello"),
        # AIMessage(content="Hi there! how can I assist you today?"),
        # HumanMessage(content="My name is MAS!"),
    ]

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        response = process_chat(chain, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print(f"\nAssistant: {response}")
