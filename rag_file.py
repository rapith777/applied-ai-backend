import math
import requests


# 1. Load document from file
def load_document(file_path):
    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        text = file.read()

    return text


# 2. Split document into overlapping chunks
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


# 3. Create embedding
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


# 4. Dot product
def dot_product(vector_a, vector_b):
    total = 0

    for a, b in zip(vector_a, vector_b):
        total = total + (a * b)

    return total


# 5. Vector magnitude
def magnitude(vector):
    total = 0

    for value in vector:
        total = total + (value * value)

    return math.sqrt(total)


# 6. Cosine similarity
def cosine_similarity(vector_a, vector_b):
    dot = dot_product(vector_a, vector_b)

    magnitude_a = magnitude(vector_a)
    magnitude_b = magnitude(vector_b)

    similarity = dot / (
        magnitude_a * magnitude_b
    )

    return similarity


# 7. Send final prompt to LLM
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


# -----------------------------------
# RAG PIPELINE
# -----------------------------------


# 8. Load real document
document = load_document(
    "documents/fastapi_notes.txt"
)


# 9. Split document into chunks
document_chunks = split_into_chunks(
    document,
    20,
    5
)


# 10. Create embedding for every chunk
chunk_data = []

for chunk in document_chunks:

    embedding = create_embedding(chunk)

    chunk_data.append({
        "text": chunk,
        "embedding": embedding
    })


# 11. User question
question = "How does FastAPI validate request data?"


# 12. Create embedding for question
question_embedding = create_embedding(
    question
)


# 13. Compare question with every chunk
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


# 14. Rank chunks by similarity
results.sort(
    key=lambda item: item["score"],
    reverse=True
)


# 15. Select top chunks
k = 2

top_results = results[:k]


# 16. Build context
context = ""

for result in top_results:
    context = (
        context
        + result["text"]
        + "\n"
    )


# 17. Build prompt for LLM
prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{question}
"""


# 18. Hand prompt to LLM
answer = generate_answer(prompt)


# 19. Show result
print("Retrieved context:")
print(context)

print("RAG answer:")
print(answer)