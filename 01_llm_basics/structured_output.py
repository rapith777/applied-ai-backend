import requests

from pydantic import BaseModel

class TopicExplanation(BaseModel):
    topic: str
    definition: str
    example: str

def explain_topic_as_json(topic):
    url = "http://127.0.0.1:11434/api/chat"

    user_prompt = (
        f"Explain {topic} for a beginner. "
        "Fill every field with real information about the topic. "
        "Do not return placeholder values such as 'string'. "
        "Return topic, definition, and example."
    
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
        "format": TopicExplanation.model_json_schema(),
        "options": {
        "temperature": 0
        }
    }

    response = requests.post(
        url,
        json = payload,
        timeout = 60
    )


    response_data = response.json()

    assistant_text = response_data["message"]["content"]

    validated_data = TopicExplanation.model_validate_json(assistant_text)

    return validated_data


result = explain_topic_as_json("FastAPI")

print(result)
print(result.topic)
print(result.definition)
print(result.example)
