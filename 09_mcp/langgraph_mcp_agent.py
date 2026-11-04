
import asyncio
from pathlib import Path

from langchain.mcp import MCPAdapter
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode


server_path = Path("09_mcp/mcp_server_basic.py")


async def main():
    # Connect to the MCP server and discover its tools.
    async with MCPAdapter(server_path) as adapter:
        tools = await adapter.list_tools()

        # Give the MCP tools to the LLM.
        llm = ChatOllama(model="llama3.2:3b")
        llm_with_tools = llm.bind_tools(tools)

        # Agent reads messages and generates the next response.
        async def agent_node(state: MessagesState):
            response = await llm_with_tools.ainvoke(
                state["messages"]
            )

            return {
                "messages": [response]
            }

        # Execute tools provided by the MCP server.
        tool_node = ToolNode(tools)

        # Build the LangGraph workflow.
        graph_builder = StateGraph(MessagesState)

        graph_builder.add_node("agent", agent_node)
        graph_builder.add_node("tools", tool_node)

        graph_builder.add_edge(START, "agent")
        graph_builder.add_edge("tools", "agent")

        # Decide whether to run a tool or finish.
        def route_after_agent(state: MessagesState):
            last_message = state["messages"][-1]

            if last_message.tool_calls:
                return "tools"

            return END

        graph_builder.add_conditional_edges(
            "agent",
            route_after_agent
        )

        # Create the runnable graph.
        graph = graph_builder.compile()

        # Starting question.
        question = "What is the sum of 15 and 27? Use the add tool to calculate it."

        initial_state = {
            "messages": [
                HumanMessage(content=question)
            ]
        }

        # Run the graph asynchronously.
        result = await graph.ainvoke(initial_state)

        # Print the final AI answer.
        final_message = result["messages"][-1]
        print(final_message.content)

        for message in result["messages"]:
            print(type(message).__name__)
            print("Content:", message.content)
            print("Tool calls:", getattr(message, "tool_calls", None))
            print("--------------------")


if __name__ == "__main__":
    asyncio.run(main())
