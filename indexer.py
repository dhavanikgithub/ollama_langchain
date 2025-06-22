
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import constants

# Load documents from a directory
loader = DirectoryLoader(constants.text_dataset_path, glob="**/*.txt")

print("dir loaded loader")

documents = loader.load()

print(len(documents))

# # Create embeddingsclear
embeddings = OllamaEmbeddings(model=constants.ollama_local_embeddings_model, show_progress=True)

# # Create Semantic Text Splitter
# text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="interquartile")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=300,
    add_start_index=True,
)

# # Split documents into chunks
texts = text_splitter.split_documents(documents)

# # Create vector store
vectorstore = Chroma.from_documents(
    documents=texts, 
    embedding= embeddings,
    persist_directory=constants.persist_directory_path)

print("vectorstore created")