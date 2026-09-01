from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model = "embeddinggemma" 
)

text = "FastAPI is a Python framework for building APIs."

vector = embeddings.embed_query(text)

print(vector)

print("length of vector:",len(vector))