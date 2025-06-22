# ollama_local_llm_model = "gemma2:9b"
ollama_local_llm_model = "deepseek-r1:8b"
# ollama_local_embeddings_model = "nomic-embed-text:v1.5"
ollama_local_embeddings_model = "mxbai-embed-large"


persist_directory_name = "db-chroma-index"
persist_directory_path = f"./{persist_directory_name}"


json_dataset_path = "./dataset/json"
text_dataset_path = "./dataset/txt"

qdrant_host = "localhost"
qdrant_port = 6333
qdrant_collection = "inspection_details_data"

top_k = 500
