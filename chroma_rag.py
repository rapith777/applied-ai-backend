from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


embeddings = OllamaEmbeddings(
    model = "embeddinggemma"
)


vector_store = Chroma(
    collection_name="fastapi_notes",
    embedding_function = embeddings,
    persist_directory = "chroma_db",
)


retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 2
        }
)


def format_docs(docs):

    contents = []

    for doc in docs:
        contents.append(
            doc.page_content
        )

    context = "\n\n".join(
        contents
    )

    return context


prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer the question using only the provided context."
    ),
    (
        "user",
        """
        context : {context}
        Question : {question}

        """
    )
]

)


llm = ChatOllama(
    model="llama3.2:3b"
)


OutputParser = StrOutputParser()


rag_chain = (
    {
        "context" : retriever | format_docs,
        "question" : RunnablePassthrough()
    }
    | prompt_template
    | llm
    | OutputParser
)


question = "How does FastAPI validate request data?"


answer = rag_chain.invoke(
    question
)


print("Question:")
print(question)

print("Answer:")
print(answer)
