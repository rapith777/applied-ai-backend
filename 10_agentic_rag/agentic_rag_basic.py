# Imports
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage


# Embedding model and vector database
embeddings = OllamaEmbeddings(
    model="embeddinggemma"
)

vector_store = Chroma(
    collection_name="pdf_notes",
    embedding_function=embeddings,
    persist_directory="pdf_chroma_db"
)


# Retrieval tool
@tool
def search_documents(query: str) -> str:
    """Search the PDF knowledge base for relevant information."""

    documents = vector_store.similarity_search(query, k=3)

    retrieved_chunks = []

    for document in documents:
        retrieved_chunks.append(document.page_content)

    return "\n\n".join(retrieved_chunks)


# LLM with tool access
llm = ChatOllama(model="llama3.2:3b")
llm_with_tools = llm.bind_tools([search_documents])

tool_node = ToolNode([search_documents])


# Initialize graph with conversation state
graph_builder = StateGraph(MessagesState)


# Agent node: processes messages and decides whether to call a tool
def agent_node(state: MessagesState):

    response = llm_with_tools.invoke(
        state["messages"]
    )

    # add_messages appends this response to the existing message history
    return {"messages": [response]}


# Register nodes
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)


# Define graph flow
graph_builder.add_edge(START, "agent")
graph_builder.add_edge("tools", "agent")

# Route to tools if requested, otherwise end execution
graph_builder.add_conditional_edges(
    "agent",
    tools_condition
)

graph = graph_builder.compile()

question = "What are the main concepts discussed in the PDF? Explain the key points based on the document."

initial_state = {
    "messages":[HumanMessage(content=question)]
}

result = graph.invoke(initial_state)

print(result["messages"][-1].content)

# Inspect complete conversation history
for message in result["messages"]:

    print(f"\nMessage Type: {message.type}")
    print(f"Content: {message.content}")

    # Check tool calls only if the message supports them
    if getattr(message, "tool_calls", None):
        print(f"Tool Calls: {message.tool_calls}")

