import logging
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.tools import Tool, tool
from typing import List, Dict
from datetime import datetime

# Setup logging configuration
logging.basicConfig(level=logging.INFO)

@tool("order_details")
def get_order_details(orderID: str) -> str:
    """
    Fetches order details for a given order ID.
    :param orderID: The ID of the order to fetch details for.
    :return: Order details as a string.
    """
    logging.info(f"Retrieving details for order ID: {orderID}")
    mocked_orders = {
        "12345": "Order 12345: Status - Shipped, Items - 3, Total - $50.00",
        "67890": "Order 67890: Status - Delivered, Items - 1, Total - $15.00",
        "54321": "Order 54321: Status - Processing, Items - 5, Total - $120.00",
    }
    return mocked_orders.get(orderID, f"Order ID {orderID} not found. Please check and try again.")

@tool("weather")
def get_weather(location: str) -> str:
    """
    Fetches the weather for a given location using an external API.
    :param location: Location to fetch the weather for.
    :return: Weather information as string.
    """
    logging.info(f"Fetching weather for location: {location}")
    # Here, you might replace this with an API call:
    return f"The weather in {location} is sunny with a temperature of 25°C."

def handle_tool_chat(request, chat_model):
    try:
        # Convert history to LangChain message format
        history = [
            HumanMessage(content=msg["content"]) if msg["role"] == "human" else AIMessage(content=msg["content"])
            for msg in request.history
        ]

        # Append the new user message
        history.append(HumanMessage(content=request.message))

        # Get the AI response
        response = chat_model.invoke(history)  # Use 'invoke' instead of 'run'

        # Append the AI's response
        ai_response = {"role": "ai", "content": response.content}
        return {"reply": ai_response, "updated_history": request.history + [ai_response]}

    except Exception as e:
        logging.error("Error during tool-based chat processing", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")