from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode


# Define tools the LLM can request.
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# Tools available to the LLM and ToolNode.
tools = [
    multiply,
    add
]


# Create the LLM and make the tools available to it.
llm = ChatOllama(
    model="llama3.2:3b"
)

llm_with_tools = llm.bind_tools(tools)


# Agent node:
# Reads the current message history and asks the LLM what to do next.
def agent_node(state: MessagesState):

    response = llm_with_tools.invoke(
        state["messages"]
    )

    # Add the new AI response to the graph's message history.
    return {
        "messages": [response]
    }


# ToolNode executes tool calls requested by the LLM.
tool_node = ToolNode(tools)


# Create a graph whose shared state contains message history.
graph_builder = StateGraph(MessagesState)


# Register the nodes in the graph.
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)


# Start by sending the user message to the agent.
graph_builder.add_edge(START, "agent")


# After tools run, return to the agent so it can see the tool result.
graph_builder.add_edge("tools", "agent")


# Decide where to go after the agent runs.
def route_after_agent(state: MessagesState):

    # Get the latest message from the message history.
    last_message = state["messages"][-1]

    # If the LLM requested tools, go to the tools node.
    if last_message.tool_calls:
        return "tools"

    # Otherwise the LLM has finished.
    return END


# Use the routing function after every agent response.
graph_builder.add_conditional_edges(
    "agent",
    route_after_agent
)


# Convert the graph design into a runnable graph.
graph = graph_builder.compile()


# Create the user's starting message.
question = "What is the output after adding 2 and 4?"

initial_state = {
    "messages": [
        HumanMessage(content=question)
    ]
}


# Run the complete graph starting with the initial state.
result = graph.invoke(initial_state)

final_message = result["messages"][-1]

print(final_message.content)
