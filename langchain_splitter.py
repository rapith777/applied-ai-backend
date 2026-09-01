from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_document(file_path):
    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        text = file.read()

    return text


# 1. Load our txt file
document = load_document(
    "documents/fastapi_notes.txt"
)


# 2. Create the LangChain text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20
)


# 3. Split the document
chunks = text_splitter.split_text(document)


# 4. Print chunks
for index, chunk in enumerate(chunks, start=1):

    print(f"Chunk {index}:")

    print(chunk)

    print("-----")