
from pathlib import Path

from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition


# Embedding model and vector database
embeddings = OllamaEmbeddings(
    model="embeddinggemma"
)

# Locate the existing ChromaDB folder in the project root
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "pdf_chroma_db"

vector_store = Chroma(
    collection_name="pdf_notes",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR)
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


# Initialize graph
graph_builder = StateGraph(MessagesState)


# Agent node
def agent_node(state: MessagesState):

    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {"messages": [response]}


# Register nodes
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)


# Define graph flow
graph_builder.add_edge(START, "agent")
graph_builder.add_edge("tools", "agent")

graph_builder.add_conditional_edges(
    "agent",
    tools_condition
)


# Compile graph for execution
graph = graph_builder.compile()
