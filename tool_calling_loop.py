from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


# Define tools the LLM can use.
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# List of tools given to the LLM.
tools = [
    multiply,
    add
]


# Dictionary used by Python to find the real tool
# from the tool name returned by the LLM.
tools_by_name = {
    "multiply": multiply,
    "add": add
}


# Create the model.
llm = ChatOllama(
    model="llama3.2:3b"
)


# Give the model access to the tools.
llm_with_tools = llm.bind_tools(tools)


# User question.
question = "Add 25 and 17, and also multiply 6 by 8."


# Start conversation history.
messages = [
    HumanMessage(content=question)
]


# First LLM call.
# The LLM decides which tool to use.
response = llm_with_tools.invoke(messages)


# Save the LLM tool request in the conversation history.
messages.append(response)


# Execute every tool requested by the LLM.
for tool_call in response.tool_calls:

    # Convert the tool name, such as "add",
    # into the actual LangChain tool object.
    selected_tool = tools_by_name[
        tool_call["name"]
    ]

    # Execute the selected tool with the arguments
    # chosen by the LLM.
    tool_result = selected_tool.invoke(
        tool_call
    )

    # Save the tool result in the conversation history.
    messages.append(tool_result)


# Send the updated conversation back to the LLM.
final_response = llm_with_tools.invoke(
    messages
)


# Print the final answer.
print(final_response.content)
