import requests


def chat_with_ollama(system_instruction, user_prompt, model_name ):
    url = "http://127.0.0.1:11434/api/chat"

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": system_instruction
            },

            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "stream": False
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=60
        )

        response.raise_for_status()

    except requests.exceptions.Timeout:
        return "Ollama took too long to respond."
    

    response_data = response.json()

    assistant_message = response_data["message"]
    assistant_text = assistant_message["content"]

    return assistant_text


answer = chat_with_ollama(
    "you are a python instructor.",
    "what is fastapi",
    "llama3.2:3b"
)

print(answer)