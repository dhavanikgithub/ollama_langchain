from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.docstore.document import Document
import constants

# Load documents
loader = DirectoryLoader(constants.text_dataset_path, glob="**/*.txt")
documents = loader.load()

print(f"{len(documents)} documents loaded.")

# Initialize embeddings
embeddings = OllamaEmbeddings(model=constants.ollama_local_embeddings_model, show_progress=True)

# Function to split content line-by-line and group dynamically
def dynamic_chunk_lines(documents, max_chars=500):
    chunked_docs = []

    for doc in documents:
        lines = doc.page_content.splitlines()
        current_chunk = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if len(current_chunk) + len(line) + 1 <= max_chars:
                current_chunk += line + "\n"
            else:
                chunked_docs.append(Document(page_content=current_chunk.strip(), metadata=doc.metadata))
                current_chunk = line + "\n"

        # Add last chunk
        if current_chunk.strip():
            chunked_docs.append(Document(page_content=current_chunk.strip(), metadata=doc.metadata))

    return chunked_docs

# Split into dynamic chunks
texts = dynamic_chunk_lines(documents, max_chars=500)

print(f"Generated {len(texts)} chunks for embedding.")

# Create and persist the vectorstore
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=constants.persist_directory_path
)

print("✅ Vector store created and persisted.")
