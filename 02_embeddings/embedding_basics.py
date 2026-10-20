import requests

def create_embedding(text):

    url = "http://127.0.0.1:11434/api/embed"

    payload = {
        "model" : "embeddinggemma",
        "input" : text
    }

    response = requests.post(
        url,
        json = payload,
        timeout = 180
    )

    response.raise_for_status()

    response_data = response.json()

    embedding = response_data["embeddings"][0]

    return embedding


vector = create_embedding(
    "what is FastApi"
)

print(type(vector))
print("length of vector:", len(vector))
print("first 10 numbers:",vector[:10])
