from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import OllamaEmbeddings  # Or langchain_ollama if upgraded
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import constants

# Load all JSON documents in a directory
def load_json_documents(directory, jq_schema):
    all_documents = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)

            loader = JSONLoader(
                file_path=filepath,
                jq_schema=jq_schema,
                text_content=False
            )
            documents = loader.load()
            all_documents.extend(documents)

    return all_documents

# Load JSON
json_docs = load_json_documents(constants.json_dataset_path, jq_schema=".inspectiondetails[]")

print(f"Loaded {len(json_docs)} inspection records.")

# Embedding setup
embeddings = OllamaEmbeddings(model=constants.ollama_local_embeddings_model, show_progress=True)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    add_start_index=True,
)

# Split documents into chunks
texts = text_splitter.split_documents(json_docs)

print(f"Number of chunks created: {len(texts)}")
if texts:
    print("Sample chunk content:\n", texts[0].page_content)

# Vector store creation
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=constants.persist_directory_path
)

print("Vectorstore for JSON inspection data created.")
