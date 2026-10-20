from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


def load_document(file_path):
    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        return file.read()


# 1. Load document text
document_text = load_document(
    "documents/fastapi_notes.txt"
)


# 2. Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20
)


# 3. Split document into chunks
chunks = text_splitter.split_text(
    document_text
)


# 4. Create embedding model
embeddings = OllamaEmbeddings(
    model="embeddinggemma"
)


# 5. Create vector store
vector_store = InMemoryVectorStore.from_texts(
    texts=chunks,
    embedding=embeddings
)


# 6. Create retriever from vector store
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 2
    }
)


# 7. User question
question = "How does FastAPI validate request data?"


# 8. Ask retriever for relevant chunks
results = retriever.invoke(
    question
)


# 9. Print retrieved chunks
for index, result_doc in enumerate(
    results,
    start=1
):

    print(f"Result {index}:")
    print(result_doc.page_content)
    print("-----")
    