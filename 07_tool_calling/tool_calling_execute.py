from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


# 1. Create tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# 2. Create LLM
llm = ChatOllama(
    model="llama3.2:3b"
)


# 3. Give tool to LLM
llm_with_tools = llm.bind_tools(
    [multiply]
)


# 4. User question
question = "What is 1234 multiplied by 5678?"


# 5. creating a container of conversation messages for history.
messages = [
    HumanMessage(content=question)
]


# 6. so llm already know there is multiply and invoking with user messages
response = llm_with_tools.invoke(
    messages
)


# 7. append the LLM response to the conversation history
messages.append(
    response
)


# 8. Take first tool call out of response.
# No multiplication has been executed yet.
tool_call = response.tool_calls[0]


# 9. Execute the requested LangChain tool.
# @tool allows multiply to use .invoke().
tool_result = multiply.invoke(
    tool_call
)


# 10. Append the tool result to messages so we can send it back to the LLM.
messages.append(
    tool_result
)


# 11. Send the updated conversation back to the tool-enabled LLM
# so it can read the tool result and produce the final answer.
final_response = llm_with_tools.invoke(
    messages
)


# 12. Print the final natural-language answer.
print(final_response.content)
