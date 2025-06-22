

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import constants

# # Create embeddingsclear
embeddings = OllamaEmbeddings(model=constants.ollama_local_embeddings_model, show_progress=False)

db = Chroma(persist_directory=constants.persist_directory_path,
            embedding_function=embeddings)

# # Create retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs= {"k": constants.top_k}
)


# # Create Ollama language model
local_llm = constants.ollama_local_llm_model

llm = ChatOllama(model=local_llm,
                 keep_alive="3h", 
                 max_tokens=512,  
                 temperature=0)

# Create prompt template
template = """<bos><start_of_turn>user\nAnswer the question based only on the following context and extract out a meaningful answer. \
Please write in full sentences with correct spelling and punctuation. if it makes sense use lists. \
If the context doen't contain the answer, just respond that you are unable to find an answer. \

CONTEXT: {context}

QUESTION: {question}

<end_of_turn>
<start_of_turn>model\n
ANSWER:"""
prompt = ChatPromptTemplate.from_template(template)

# Create the RAG chain using LCEL with prompt printing and streaming output
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Function to ask questions
# def ask_question(question):
#     print("Answer:\n\n", end=" ", flush=True)
#     for chunk in rag_chain.stream(question):
#         print(chunk.content, end="", flush=True)
#     print("\n")

def ask_question(question, filename="qa_log.txt"):
    print("Answer:\n\n", end=" ", flush=True)

    # Start writing question immediately
    with open(filename, "a", encoding="utf-8") as f:
        f.write("<UserQuestion>\n")
        f.write(question.strip() + "\n")
        f.write("</UserQuestion>\n")
        f.write("<ModelAnswer>\n")
        f.flush()
        try:
            for chunk in rag_chain.stream(question):
                content = chunk.content
                print(content, end="", flush=True)
                f.write(content)
                f.flush()  # Ensure it's written to disk immediately
        except KeyboardInterrupt:
            f.write("\n\n[!] Process interrupted. Saving partial answer.\n")
            f.flush()
            print("\n[!] Process interrupted. Saving partial answer.\n")
        finally:
            f.write("\n</ModelAnswer>\n\n")  # Ensure closing tag is written
            f.flush()
            print("\n\nAnswer saved to file.\n\n")

# Example usage
if __name__ == "__main__":
    while True:
        user_question = input("Ask a question (or type 'quit' to exit): ")
        if user_question.lower() == 'quit':
            break
        answer = ask_question(user_question)
        # print("\nFull answer received.\n")

