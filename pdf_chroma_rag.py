from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 1. Create embedding model
# Needed for new questions
embeddings = OllamaEmbeddings(
    model="embeddinggemma"
)


# 2. Open existing PDF Chroma database
vector_store = Chroma(
    collection_name="pdf_notes",
    embedding_function=embeddings,
    persist_directory="pdf_chroma_db"
)


# 3. Create retriever
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 2
    }
)


# 4. User question
question = "How does FastAPI validate incoming data?"


# 5. Retrieve relevant PDF chunks
results = retriever.invoke(
    question
)


# 6. Build context from retrieved Documents
context_parts = []

for result_doc in results:
    context_parts.append(
        result_doc.page_content
    )

context = "\n\n".join(
    context_parts
)


# 7. Create prompt template
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


# 8. Create LLM
llm = ChatOllama(
    model="llama3.2:3b"
)


# 9. Output parser
output_parser = StrOutputParser()


# 10. Create prompt -> LLM -> text chain
answer_chain = (
    prompt_template
    | llm
    | output_parser
)


# 11. Generate answer
answer = answer_chain.invoke(
    {
        "context": context,
        "question": question
    }
)


# 12. Print answer
print("Question:")
print(question)

print("\nAnswer:")
print(answer)


# 13. Print sources
print("\nSources:")

for index, result_doc in enumerate(
    results,
    start=1
):

    print(f"Source {index}:")

    print(
        "File:",
        result_doc.metadata.get("source")
    )

    print(
        "Page:",
        result_doc.metadata.get("page_label")
    )

    print("-----")
    