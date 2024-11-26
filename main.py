import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from langchain_openai import ChatOpenAI
# Load environment variables from .env file
load_dotenv()  # Defaults to loading a file named ".env"

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found in .env file")

os.environ["OPENAI_API_KEY"] = api_key

## Initialize the ChatOpenAI model with shared configuration
chat_model = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo") 
#Import functions from the separate chatbot modules
from conversational_bot import handle_conversational_chat
from rag_bot import handle_rag_chat
from tool_bot import handle_tool_chat
from rag_bot import initialize_faiss_store
# Initialize FAISS store at startup
faiss_store = initialize_faiss_store() 
# Create a FastAPI app instance
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],         # Allow all HTTP methods
    allow_headers=["*"]          # Allow all headers
)

# Define a data model for the chat requests using Pydantic
class ChatRequest(BaseModel):
    history: List[Dict[str, str]]
    message: str

@app.post("/chat/conversational")
async def chat_conversational(request: ChatRequest):
    print("Received request for conversational bot")
    return handle_conversational_chat(request, chat_model)

@app.post("/chat/rag")
async def chat_rag(request: ChatRequest):
    return handle_rag_chat(request, faiss_store,  chat_model)

@app.post("/chat/tool")
async def chat_tool(request: ChatRequest):
    return handle_tool_chat(request, chat_model)

# Run the server if this script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)