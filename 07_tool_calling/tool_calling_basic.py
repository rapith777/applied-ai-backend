from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# 1. Creating a normal Python function
#    and turn it into a LangChain tool
@tool
def multiply(a:int, b:int) -> int:
    """multiplying two numbers"""
    return a * b


# 2. Create LLM
llm = ChatOllama(
    model="llama3.2:3b"
)


# 3. Give the tool to the LLM
llm_with_tools = llm.bind_tools(
    [multiply]
)


# 4. Ask something that should require the tool
question = "What is 1234 multiplied by 5678?"


# 5. Let the LLM decide what to do
response = llm_with_tools.invoke(
    question
    )


# 6. Inspect the model response
print("Model content:")
print(response.content)

print("\nTool calls:")
print(response.tool_calls)