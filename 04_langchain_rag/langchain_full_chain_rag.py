from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def load_document(file_path):
    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        return file.read()



def format_docs(docs):

    contents = []

    for doc in docs:
        contents.append(doc.page_content)

    context = "\n\n".join(contents)

    return context

#def format_docs(docs):
#   return "\n\n".join(
#       doc.page_content
#       for doc in docs
#   )


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


# 6. PROMPT
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


# 7. LLM
llm = ChatOllama(
    model="llama3.2:3b"
)


# 8. OUTPUT PARSER
output_parser = StrOutputParser()


# 9. COMPLETE RAG CHAIN
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt_template
    | llm
    | output_parser
)


# 10. QUESTION
question = "How does FastAPI validate request data?"


# 11. RUN EVERYTHING
answer = rag_chain.invoke(
    question
)


# 12. OUTPUT
print("Answer:")
print(answer)
