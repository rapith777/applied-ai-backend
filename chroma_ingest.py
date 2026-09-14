from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def load_document(file_path):
    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        return file.read()


# 1. Read document
document_text = load_document(
    "documents/fastapi_notes.txt"
)


# 2. Create splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20
)


# 3. Split document
chunks = text_splitter.split_text(
    document_text
)


# 4. Create embedding model
embeddings = OllamaEmbeddings(
    model="embeddinggemma"
)


# 5. Create Chroma database
#    and save chunks + embeddings to disk
vector_store = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    collection_name="fastapi_notes",
    persist_directory="chroma_db"
)


print("Document stored successfully.")
print("Number of chunks:", len(chunks))
