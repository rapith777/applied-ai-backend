import requests


def get_ollama_models():
    url = "http://127.0.0.1:11434/api/tags"

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    return response.json()


def extract_model_names(data):
    names = []

    for model in data["models"]:
        names.append(model["name"])

    return names


result = get_ollama_models()
model_names = extract_model_names(result)

print(model_names)

