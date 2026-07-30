import requests

def get_ollama_models():
    url = "http://127.0.0.1:11434/api/tags"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    return data

result = get_ollama_models()

for model in result["models"]:
    print (model["name"])

