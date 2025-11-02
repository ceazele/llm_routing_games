from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI
import json
import os



os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCAHIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = ""

os.environ["OPENAI_API_KEY"] = ""
model = ChatOpenAI(model="gpt-4o-2024-05-13")


from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)


store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# Specify the Pydantic format of output
class Route(BaseModel):
    route: str = Field(description="choice of route")

# Set up a parser
parser = PydanticOutputParser(pydantic_object=Route)


with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)




config = {"configurable": {"session_id": session_id}}
