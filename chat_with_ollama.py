import requests


def chat_with_ollama(user_prompt):
    url = "http://127.0.0.1:11434/api/chat"

    payload = {
        "model": "llama3.2:3b",
        "messages": [
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "stream": False
    }

    response = requests.post(
        url,
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    response_data = response.json()

    assistant_message = response_data["message"]
    assistant_text = assistant_message["content"]

    return assistant_text


answer = chat_with_ollama(
    "Explain Python functions in one sentence."
)

print(answer)