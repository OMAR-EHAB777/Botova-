import logging
from fastapi import HTTPException
from pathlib import Path
from langchain.schema import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain import hub

logging.basicConfig(level=logging.INFO)

# Define paths and configurations
PDF_FOLDER = "./pdfs"
VECTOR_STORE_PATH = "./faiss_index"

def load_pdfs(folder: str):
    pdf_files = Path(folder).glob("*.pdf")
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for pdf in pdf_files:
        loader = PyPDFLoader(str(pdf))
        try:
            raw_documents = loader.load()
            documents.extend(text_splitter.split_documents(raw_documents))
        except Exception as e:
            logging.error(f"Failed to load PDF {pdf}: {e}")
    return documents

def update_faiss_store(faiss_store, documents):
    # Create embeddings for new documents
    embeddings = OpenAIEmbeddings()
    new_vector_store = FAISS.from_documents(documents, embeddings)
    faiss_store.merge_from(new_vector_store)
    faiss_store.save_local(VECTOR_STORE_PATH)
    return faiss_store

def initialize_faiss_store():
    documents = load_pdfs(PDF_FOLDER)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    return vector_store

def handle_rag_chat(request, faiss_store, chat_model):
    try:
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        combine_docs_chain = create_stuff_documents_chain(
            chat_model, retrieval_qa_chat_prompt
        )
        
        retrieve_chain = create_retrieval_chain(
            faiss_store.as_retriever(), combine_docs_chain
        )

        # Prepare chat history
        history = [
            HumanMessage(content=msg["content"]) if msg["role"] == "human" else AIMessage(content=msg["content"])
            for msg in request.history
        ]
        user_message = HumanMessage(content=request.message)
        history.append(user_message)

        # Execute the retrieval chain process
        response = retrieve_chain.invoke({"input": request.message, "chat_history": history})
        ai_response = {"role": "ai", "content": response["answer"]}
        return {"reply": ai_response, "updated_history": request.history + [ai_response]}

    except Exception as e:
        logging.error("Error processing RAG chat", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Initialization logic which might be executed once at startup in main.py
if __name__ == "__main__":
    if Path(VECTOR_STORE_PATH).exists():
        faiss_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True,
        )
        # Optionally update FAISS store with any new PDFs
        new_documents = load_pdfs(PDF_FOLDER)
        if new_documents:
            faiss_store = update_faiss_store(faiss_store, new_documents)
    else:
        faiss_store = initialize_faiss_store()