from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings


def load_document(file_path):
    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        text = file.read()

    return text


# 1. Load document
document = load_document(
    "documents/fastapi_notes.txt"
)


# 2. Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20
)


# 3. Split document
chunks = text_splitter.split_text(document)


# 4. Create embedding object
embeddings = OllamaEmbeddings(
    model="embeddinggemma"
)


# 5. Embed all chunks
chunk_vectors = embeddings.embed_documents(chunks)


# 6. Inspect result
for index, chunk in enumerate(chunks, start=1):

    vector = chunk_vectors[index - 1]

    print(f"Chunk {index}:")
    print(chunk)

    print("Vector length:")
    print(len(vector))

    print("-----")
