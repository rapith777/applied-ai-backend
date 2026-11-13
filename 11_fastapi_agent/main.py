
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from agent import graph


# Initialize FastAPI
app = FastAPI()


# Request model
class ChatRequest(BaseModel):
    question: str


# Health check endpoint
@app.get("/")
def home():

    return {
        "message": "Agentic RAG API is running"
    }


# Chat endpoint
@app.post("/chat")
def chat(request: ChatRequest):

    # Convert user question into LangGraph input
    initial_state = {
        "messages": [
            HumanMessage(content=request.question)
        ]
    }

    # Execute Agentic RAG graph
    result = graph.invoke(initial_state)

    # Extract final AI response
    answer = result["messages"][-1].content

    # Return response to frontend
    return {"answer": answer}
