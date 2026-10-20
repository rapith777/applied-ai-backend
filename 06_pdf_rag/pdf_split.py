from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Load PDF
loader = PyPDFLoader(
    "documents/sample.pdf"
)

documents = loader.load()


# 2. Create splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20
)


# 3. Split Document objects
chunks = text_splitter.split_documents(
    documents
)


# 4. Inspect chunks
for index, chunk in enumerate(
    chunks,
    start=1
):

    print(f"Chunk {index}:")

    print("Text:")
    print(chunk.page_content)

    print("Metadata:")
    print(chunk.metadata)

    print("-----")
    