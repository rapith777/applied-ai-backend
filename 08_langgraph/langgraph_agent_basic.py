from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode


# Define tools the LLM can use.
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# Tools available to the agent.
tools = [
    multiply,
    add
]


# Create the LLM.
llm = ChatOllama(
    model="llama3.2:3b"
)


# Give the LLM access to the tools.
llm_with_tools = llm.bind_tools(
    tools
)


def agent_node(state: MessagesState):

    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }

