from langchain_community.document_loaders import PyPDFLoader


# 1. Create PDF loader
loader = PyPDFLoader(
    "documents/sample.pdf"
)


# 2. Load PDF
documents = loader.load()


# 3. Inspect pages
for index, doc in enumerate(
    documents,
    start=1
):

    print(f"Document {index}:")

    print("Text:")
    print(doc.page_content)

    print("Metadata:")
    print(doc.metadata)

    print("-----")