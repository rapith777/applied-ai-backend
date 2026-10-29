from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b 


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


tools = [
    add,
    multiply
]


llm = ChatOllama(
    model="llama3.2:3b"
)


llm_with_tools = llm.bind_tools(tools)


def agent_node(state: MessagesState):

    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


tool_node =ToolNode(tools)

graph_builder = StateGraph(MessagesState)

graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_edge("tools", "agent")


def route_after_agent(state :MessagesState):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"
    return END


graph_builder.add_conditional_edges("agent", route_after_agent)

graph = graph_builder.compile()

question = "First add 2 and 4, then multiply the result by 3."

initial_state = {
    "messages":[
        HumanMessage(content=question)
    ]
}

result = graph.invoke(initial_state)

last_message = result["messages"][-1]
print(last_message.content)
