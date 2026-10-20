from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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


# 3. EMBEDDINGS
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


# 7. RETRIEVE DOCUMENTS
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


# 9. PROMPT TEMPLATE
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


# 10. LLM
llm = ChatOllama(
    model="llama3.2:3b"
)


# 11. OUTPUT PARSER
output_parser = StrOutputParser()


# 12. CREATE CHAIN
chain = (
    prompt_template
    | llm
    | output_parser
)


# 13. RUN CHAIN
answer = chain.invoke(
    {
        "context": context,
        "question": question
    }
)


# 14. OUTPUT
print("Retrieved Context:")
print(context)

print("Answer:")
print(answer)
