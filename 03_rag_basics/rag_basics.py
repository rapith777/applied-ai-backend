import math
import requests


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

    answer = response_data["message"]["content"]

    return answer


documents = [
    "FastAPI is a Python framework for building web APIs.",
    "Docker is a tool for packaging applications into containers.",
    "Bananas are yellow fruits that grow in tropical regions."
]


question = "How can I create an API using Python?"

question_vector = create_embedding(question)

results = []

for document in documents:

    document_vector = create_embedding(document)

    score = cosine_similarity(
        question_vector,
        document_vector
    )

    results.append({
        "score": score,
        "document": document
    })


results.sort(
    key=lambda item: item["score"],
    reverse=True
)


k = 2
top_results = results[:k]

context = ""

for result in top_results:
    context = context + result["document"] + "\n"


prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer only using the provided context.
"""


print("Retrieved context:")
print(context)

print("Final prompt:")
print(prompt)

answer = generate_answer(prompt)

print("RAG answer:")
print(answer)