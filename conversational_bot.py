import logging
from fastapi import HTTPException
from langchain.schema import HumanMessage, AIMessage

logging.basicConfig(level=logging.INFO)

def format_chat_history(history):
    return [
        HumanMessage(content=msg["content"]) if msg["role"] == "human" else AIMessage(content=msg["content"])
        for msg in history
    ]

def handle_conversational_chat(request, chat_model):
    try:
        logging.info("Formatting chat history for model input")
        history = format_chat_history(request.history)
        history.append(HumanMessage(content=request.message))

        logging.info("Sending request to AI model")
        response = chat_model.invoke(history)
        ai_response = {"role": "ai", "content": response.content}
        
        logging.info("Returning AI response to client")
        return {"reply": ai_response, "updated_history": request.history + [ai_response]}

    except Exception as e:
        logging.error("Error during chat processing", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))