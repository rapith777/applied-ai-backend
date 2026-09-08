from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate


def load_document(file_path):
    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        return file.read()


# 1. READ
document_text = load_document(
    "documents/fastapi_notes.txt"
)


# 2. CUT
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20
)

chunks = text_splitter.split_text(
    document_text
)


# 3. EMBEDDING MODEL
embeddings = OllamaEmbeddings(
    model="embeddinggemma"
)


# 4. VECTOR STORE
vector_store = InMemoryVectorStore.from_texts(
    texts=chunks,
    embedding=embeddings
)


# 5. RETRIEVER
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 2
    }
)


# 6. QUESTION
question = "How does FastAPI validate request data?"


# 7. RETRIEVE RELEVANT DOCUMENTS
results = retriever.invoke(
    question
)


# 8. BUILD CONTEXT
context = ""

for result_doc in results:
    context = (
        context
        + result_doc.page_content
        + "\n"
    )


# 9. CREATE LANGCHAIN PROMPT TEMPLATE
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using only the provided context."
        ),
        (
            "user",
            """
Context:
{context}

Question:
{question}
"""
        )
    ]
)


# 10. FILL THE TEMPLATE
prompt = prompt_template.invoke(
    {
        "context": context,
        "question": question
    }
)


# 11. CREATE LLM
llm = ChatOllama(
    model="llama3.2:3b"
)


# 12. SEND PROMPT TO LLM
response = llm.invoke(
    prompt
)


# 13. OUTPUT
print("Retrieved Context:")
print(context)

print("Answer:")
print(response.content)
