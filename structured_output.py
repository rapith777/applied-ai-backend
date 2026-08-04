import json
import requests

from pydantic import BaseModel

class TopicExplanation(BaseModel):
    topic: str
    definition: str
    example: str

def explain_topic_as_json(topic):
    url = "http://127.0.0.1:11434/api/chat"

    user_prompt = (
        f"Explain {topic}."
        "Return JSON with these fields: "
        "topic, definition, example."
    
    )

    payload = {
        "model": "llama3.2:3b",
        "messages": [
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "stream": False,
        "format": "json"
    }

    response = requests.post(
        url,
        json = payload,
        timeout = 60
    )


    response_data = response.json()

    assistant_text = response_data["message"]["content"]

    structured_data = json.loads(assistant_text)

    validated_data = TopicExplanation.model_validate(structured_data)

    return validated_data


result = explain_topic_as_json("FastAPI")

print(result)
print(result.definition)
