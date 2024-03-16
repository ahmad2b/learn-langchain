from dotenv import load_dotenv

load_dotenv()


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain

# docA = Document(
#     page_content="LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together. LCEL was designed from day 1 to support putting prototypes in production, with no code changes, from the simplest “prompt + LLM” chain to the most complex chains (we’ve seen folks successfully run LCEL chains with 100s of steps in production). To highlight a few of the reasons you might want to use LCEL:"
# )


def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    splitdocs = splitter.split_documents(docs)

    return splitdocs


def create_db(docs):
    embedder = OpenAIEmbeddings()

    vectorStore = FAISS.from_documents(docs, embedding=embedder)
    return vectorStore


def create_chain(vectorStore):
    model = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.4)

    prompt = ChatPromptTemplate.from_template(
        """
            Answer the user's question:
            Context: {context}
            Question: {input}
        """
    )

    # chain = prompt | model
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt,
    )

    retriever = vectorStore.as_retriever(search_kwargs={"k": 1})
    retrieval_chain = create_retrieval_chain(retriever, chain)

    return retrieval_chain


docs = get_documents_from_web("https://python.langchain.com/docs/expression_language/")
vectorStore = create_db(docs)
chain = create_chain(vectorStore)

response = chain.invoke({"input": "What is LCEL?"})

print(response)
