from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# 1. Create the same embedding model
embeddings = OllamaEmbeddings(
    model="embeddinggemma"
)


# 2. Open the already existing Chroma database
vector_store = Chroma(
    collection_name="fastapi_notes",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)


# 3. Create retriever
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 2
    }
)


# 4. Ask a question
question = "How does FastAPI validate request data?"


# 5. Retrieve relevant chunks
results = retriever.invoke(
    question
)


# 6. Print results
for index, result_doc in enumerate(results,start=1):

    print(f"Result {index}:")
    print(result_doc.page_content)
    print("-----")