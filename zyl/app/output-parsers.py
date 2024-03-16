from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    CommaSeparatedListOutputParser,
    JsonOutputParser,
)
from langchain_core.pydantic_v1 import BaseModel, Field

model = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.7)


def call_string_output_parsers():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Tell me a joke about the following subject",
            ),
            ("human", "{input}"),
        ]
    )

    parser = StrOutputParser()

    chain = prompt | model | parser

    return chain.invoke({"input": "chicken"})


def call_list_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Generate a list of 10 synonyms for the following word. Return the result in a comma-separated list.",
            ),
            ("human", "{input}"),
        ]
    )

    parser = CommaSeparatedListOutputParser()

    chain = prompt | model | parser

    return chain.invoke({"input": "happy"})


def call_json_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract information from the following phrase. \nFormatting Instructions: {format_instructions}",
            ),
            ("human", "{phrase}"),
        ]
    )

    class Person(BaseModel):
        name: str = Field(description="The name of the person")
        age: int = Field(description="The age of the person")

    parser = JsonOutputParser(pydantic_object=Person)

    chain = prompt | model | parser

    return chain.invoke(
        {
            "phrase": "Max is 25 years old",
            "format_instructions": parser.get_format_instructions(),
        },
    )


# print(call_string_output_parsers())

# print(call_list_output_parser())

print(call_json_output_parser())
