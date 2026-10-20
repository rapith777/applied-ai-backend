from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# 1. Load PDF
loader = PyPDFLoader(
    "documents/sample.pdf"
)

documents = loader.load()


# 2. Split PDF Documents into smaller Documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20
)

chunks = text_splitter.split_documents(
    documents
)


# 3. Create embedding model
embeddings = OllamaEmbeddings(
    model="embeddinggemma"
)


# 4. Store Document chunks in Chroma
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="pdf_notes",
    persist_directory="pdf_chroma_db"
)


# 5. Print what was stored
print("PDF stored successfully.")
print("Number of chunks:", len(chunks))


for index, chunk in enumerate(
    chunks,
    start=1
):

    print(f"Chunk {index}:")
    print(chunk.page_content)

    print("Metadata:")
    print(chunk.metadata)

    print("-----")
    