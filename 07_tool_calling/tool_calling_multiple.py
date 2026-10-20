from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


# 1. Create a multiplication tool.
# @tool converts this normal Python function into a LangChain tool.
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# 2. Create an addition tool.
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# 3. Create the LLM.
llm = ChatOllama(
    model="llama3.2:3b"
)


# 4. Give the LLM access to both tools.
# The LLM can now choose between multiply and add.
llm_with_tools = llm.bind_tools([
    multiply,
    add
])


# 5. Create the user's question.
question = "What is 25 plus 17?"


# 6. Start the conversation history.
# HumanMessage represents the user's message.
# We use a list because more messages will be added later.
messages = [
    HumanMessage(content=question)
]


# 7. Send the conversation to the LLM.
# The LLM reads the question and decides whether a tool is needed.
response = llm_with_tools.invoke(messages)


# 8. Add the LLM's response to the conversation history.
# This response contains the tool request chosen by the LLM.
messages.append(response)


# 9. Get the first tool request chosen by the LLM.
# Example:
# {
#     "name": "add",
#     "args": {"a": 25, "b": 17},
#     ...
# }
tool_call = response.tool_calls[0]


# 10. Create a lookup dictionary.
# The key is the tool name returned by the LLM.
# The value is the actual LangChain tool object.
tools_by_name = {
    "multiply": multiply,
    "add": add
}


# 11. Find the actual tool that matches the LLM's choice.
# If tool_call["name"] is "add",
# this becomes tools_by_name["add"].
selected_tool = tools_by_name[
    tool_call["name"]
]


# 12. Execute the selected tool.
# tool_call contains the arguments the LLM selected,
# for example: a=25 and b=17.
tool_result = selected_tool.invoke(
    tool_call
)


# 13. Add the tool result to the conversation history.
# Now the LLM can see the result when we call it again.
messages.append(tool_result)


# 14. Send the updated conversation back to the LLM.
# The LLM can now see the tool result and produce a final answer.
final_response = llm_with_tools.invoke(
    messages
)


# 15. Print the final natural-language answer.
print(final_response.content)