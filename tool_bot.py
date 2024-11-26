import logging
from fastapi import HTTPException
from langchain.schema import HumanMessage, AIMessage

logging.basicConfig(level=logging.INFO)

def handle_tool_chat(request, agent):
    try:
        logging.debug(f"Incoming chat request: {request}")

        history = [
            HumanMessage(content=msg["content"]) if msg["role"] == "human" else AIMessage(content=msg["content"])
            for msg in request.history
        ]

        logging.debug(f"Constructed history for agent: {history}")
        history.append(HumanMessage(content=request.message))

        response = agent.invoke(history)

        logging.debug(f"Agent response: {response}")

        ai_content = response.get('output', 'Default response if content key is missing')

        ai_response = {"role": "ai", "content": ai_content}
        return {"reply": ai_response, "updated_history": request.history + [ai_response]}

    except Exception as e:
        logging.error("Error during tool-based chat processing", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")