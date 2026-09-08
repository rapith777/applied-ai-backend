from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore


def load_document(file_path):
    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        return file.read()


# 1. READ
document_text = load_document(
    "documents/fastapi_notes.txt"
)


# 2. CUT
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20
)

chunks = text_splitter.split_text(
    document_text
)


# 3. CREATE EMBEDDING MODEL
embeddings = OllamaEmbeddings(
    model="embeddinggemma"
)


# 4. KEEP - create vector store
vector_store = InMemoryVectorStore.from_texts(
    texts=chunks,
    embedding=embeddings
)


# 5. CREATE RETRIEVER
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 2
    }
)


# 6. USER QUESTION
question = "How does FastAPI validate request data?"


# 7. FIND - retrieve relevant chunks
results = retriever.invoke(
    question
)


# 8. BUILD CONTEXT
context = ""

for result_doc in results:
    context = (
        context
        + result_doc.page_content
        + "\n"
    )


# 9. BUILD PROMPT
prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{question}
"""


# 10. CREATE CHAT MODEL
llm = ChatOllama(
    model="llama3.2:3b"
)


# 11. ASK LLM
response = llm.invoke(
    prompt
)


# 12. SHOW RESULT
print("Retrieved Context:")
print(context)

print("Answer:")
print(response.content)
