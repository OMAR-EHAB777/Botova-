import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.tools import  tool
from langchain.agents import initialize_agent, AgentType
# Load environment variables from .env file
load_dotenv()  # Defaults to loading a file named ".env"

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found in .env file")

os.environ["OPENAI_API_KEY"] = api_key

## Initialize the ChatOpenAI model with shared configuration
chat_model = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo") 
# Define tools
from langchain.tools import tool
import logging

@tool("order_details")
def get_order_details(orderID: str) -> str:
    """
    Fetches order details for a given order ID.
    """
    orderID = orderID.replace('orderID = ', '').strip('" ')
    # Log the sanitized order ID
    logging.info(f"Sanitized order ID: {orderID}")
    mocked_orders = {
        "12345": "Order 12345: Status - Shipped, Items - 3, Total - $50.00",
        "67890": "Order 67890: Status - Delivered, Items - 1, Total - $15.00",
        "54321": "Order 54321: Status - Processing, Items - 5, Total - $120.00",
    }
    # Ensure the order ID is correctly formatted and fetched
    order_details = mocked_orders.get(orderID)
    if not order_details:
        logging.warning(f"Order ID {orderID} not found.")
    return order_details or f"Order ID {orderID} not found. Please check and try again."


@tool("weather")
def get_weather(location: str) -> str:
    """
    Fetches the weather for a given location using an external API.
    :param location: Location to fetch the weather for.
    :return: Weather information as a string.
    """
    return f"The weather in {location} is sunny with a temperature of 25°C."
# Initialize the agent with the tools
tools = [get_order_details, get_weather]
agent = initialize_agent(
    tools=tools,
    llm=chat_model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True  # Enables retry on parsing errors
)

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
    # Pass the initialized agent to the handler
    return handle_tool_chat(request, agent)

# Run the server if this script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)