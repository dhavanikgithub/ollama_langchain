from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import constants
import numpy as np

# === Embedding Model ===
embeddings = OllamaEmbeddings(model=constants.ollama_local_embeddings_model, show_progress=False)

# === Vector DB ===
db = Chroma(
    persist_directory=constants.persist_directory_path,
    embedding_function=embeddings
)

# === Prompt Template ===
template = """<bos><start_of_turn>user
Answer the question based only on the following context and extract out a meaningful answer.
Please write in full sentences with correct spelling and punctuation. If it makes sense, use lists.
If the context doesn't contain the answer, just respond that you are unable to find an answer.

CONTEXT: {context}

QUESTION: {question}
<end_of_turn>
<start_of_turn>model
ANSWER:"""
prompt = ChatPromptTemplate.from_template(template)

# === Language Model ===
llm = ChatOllama(
    model=constants.ollama_local_llm_model,
    keep_alive="3h",
    max_tokens=512,
    temperature=0
)

# === Hybrid Retrieval Logic ===
def hybrid_retrieve(question, db, embedding_model, mode="hybrid", k=5, fetch_k=20, lambda_mult=0.5):
    """Perform retrieval using similarity, MMR, or hybrid."""
    # Embed the query
    query_embedding = embedding_model.embed_query(question)

    # Step 1: Similarity search
    similar_docs = db.similarity_search_by_vector(query_embedding, k=fetch_k)

    # Step 2: Attach embedding to each document for MMR (manually)
    for doc in similar_docs:
        doc.metadata["embedding"] = embedding_model.embed_query(doc.page_content)

    # Step 3: Use retrieval mode
    if mode == "similarity":
        return similar_docs[:k]
    elif mode == "mmr":
        return mmr_filter(query_embedding, similar_docs, k=k, lambda_mult=lambda_mult)
    elif mode == "hybrid":
        return mmr_filter(query_embedding, similar_docs, k=k, lambda_mult=lambda_mult)
    else:
        raise ValueError("Invalid mode. Use 'similarity', 'mmr', or 'hybrid'.")

# === Manual MMR Filter ===
def mmr_filter(query_embedding, documents, k=5, lambda_mult=0.5):
    """Apply basic MMR reranking to documents using cosine similarity."""
    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    selected = []
    candidate_embeddings = [doc.metadata["embedding"] for doc in documents]
    docs_remaining = documents.copy()

    while len(selected) < k and docs_remaining:
        max_score = -1
        best_doc = None

        for i, doc in enumerate(docs_remaining):
            rel_score = cosine_sim(query_embedding, doc.metadata["embedding"])
            div_score = max([cosine_sim(doc.metadata["embedding"], s.metadata["embedding"]) for s in selected], default=0)
            score = lambda_mult * rel_score - (1 - lambda_mult) * div_score

            if score > max_score:
                max_score = score
                best_doc = doc

        if best_doc:
            selected.append(best_doc)
            docs_remaining.remove(best_doc)

    return selected

# === Embed and store document embeddings in metadata ===
# (Needed for manual MMR filtering)
def preload_embeddings(docs):
    for doc in docs:
        doc.metadata["embedding"] = embeddings.embed_query(doc.page_content)
    return docs

# === RAG Chain Definition ===
def build_rag_chain(mode="hybrid"):
    def get_docs(question):
        return hybrid_retrieve(
            question,
            db=db,
            embedding_model=embeddings,
            mode=mode
        )


    return (
        {"context": get_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

# === Chat Loop ===
def ask_question(rag_chain, question):
    print("Answer:\n\n", end=" ", flush=True)
    for chunk in rag_chain.stream(question):
        print(chunk.content, end="", flush=True)
    print("\n")

# === Main Interactive CLI ===
if __name__ == "__main__":
    print("🔍 Retrieval Modes: similarity | mmr | hybrid")
    mode = input("Choose retrieval mode [default=hybrid]: ").strip().lower() or "hybrid"
    rag_chain = build_rag_chain(mode)

    while True:
        user_question = input("\nAsk a question (or type 'quit' to exit): ")
        if user_question.lower() == 'quit':
            break
        ask_question(rag_chain, user_question)
