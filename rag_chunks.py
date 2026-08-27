import math
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


def split_into_chunks(text, chunk_size, overlap):
    words = text.split()

    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):

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


def dot_product(vector_a, vector_b):
    total = 0

    for a, b in zip(vector_a, vector_b):
        total = total + (a * b)

    return total


def magnitude(vector):
    total = 0

    for value in vector:
        total = total + (value * value)

    return math.sqrt(total)


def cosine_similarity(vector_a, vector_b):
    dot = dot_product(vector_a, vector_b)

    magnitude_a = magnitude(vector_a)
    magnitude_b = magnitude(vector_b)

    similarity = dot / (magnitude_a * magnitude_b)

    return similarity


def generate_answer(prompt):
    url = "http://127.0.0.1:11434/api/chat"

    payload = {
        "model": "llama3.2:3b",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False
    }

    response = requests.post(
        url,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    response_data = response.json()

    return response_data["message"]["content"]


# 1. Split document into overlapping chunks

document_chunks = split_into_chunks(
    document,
    20,
    5
)


# 2. Create embedding for every chunk

chunk_data = []

for chunk in document_chunks:

    embedding = create_embedding(chunk)

    chunk_data.append({
        "text": chunk,
        "embedding": embedding
    })


# 3. User question

question = "How does FastAPI validate request data?"


# 4. Create embedding for the question

question_embedding = create_embedding(question)


# 5. Compare question with every chunk

results = []

for item in chunk_data:

    score = cosine_similarity(
        question_embedding,
        item["embedding"]
    )

    results.append({
        "score": score,
        "text": item["text"]
    })


# 6. Rank chunks from most relevant to least relevant

results.sort(
    key=lambda item: item["score"],
    reverse=True
)


# 7. Take the top-k chunks

k = 2

top_results = results[:k]


# 8. Build context

context = ""

for result in top_results:
    context = context + result["text"] + "\n"


# 9. Build the RAG prompt

prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{question}
"""


# 10. Send prompt to Llama

answer = generate_answer(prompt)


# 11. Show result

print("Retrieved context:")
print(context)

print("RAG answer:")
print(answer)
