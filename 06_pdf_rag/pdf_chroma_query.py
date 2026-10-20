from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# 1. Create embedding model
# Needed to embed the NEW question
embeddings = OllamaEmbeddings(
    model="embeddinggemma"
)


# 2. Open the existing PDF Chroma database
vector_store = Chroma(
    collection_name="pdf_notes",
    embedding_function=embeddings,
    persist_directory="pdf_chroma_db"
)


# 3. Create retriever
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 2
    }
)


# 4. Ask a question
question = "How does FastAPI validate incoming data?"


# 5. Retrieve relevant PDF chunks
results = retriever.invoke(
    question
)


# 6. Print retrieved chunks and metadata
for index, result_doc in enumerate(
    results,
    start=1
):

    print(f"Result {index}:")

    print("Text:")
    print(result_doc.page_content)

    print("Source:")
    print(result_doc.metadata.get("source"))

    print("Page:")
    print(result_doc.metadata.get("page"))

    print("Page label:")
    print(result_doc.metadata.get("page_label"))

    print("-----")
