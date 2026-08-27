import requests


document = """
FastAPI is a modern Python framework for building APIs.
It uses Python type hints for validation and documentation.
FastAPI supports asynchronous programming using async and await.
It automatically generates interactive API documentation.
Pydantic is commonly used with FastAPI for request and response validation.
Docker can be used to package a FastAPI application into containers.
A database such as PostgreSQL can be connected to store application data.
Authentication can be added using tokens or JWT.
"""


def split_into_chunks(text, chunk_size):
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        start = i
        end = i + chunk_size

        chunk_words = words[start:end]

        chunk = " ".join(chunk_words)

        chunks.append(chunk)

    return chunks


def create_embedding(text):
    url = "http://127.0.0.1:11434/api/embed"

    payload = {
        "model": "embeddinggemma",
        "input": text
    }

    response = requests.post(
        url,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    response_data = response.json()

    return response_data["embeddings"][0]


document_chunks = split_into_chunks(
    document,
    20
)


chunk_data = []


for chunk in document_chunks:

    embedding = create_embedding(chunk)

    chunk_data.append({
        "text": chunk,
        "embedding": embedding
    })


for item in chunk_data:

    print(item["text"])

    print(
        "Embedding length:",
        len(item["embedding"])
    )

    print("-----")