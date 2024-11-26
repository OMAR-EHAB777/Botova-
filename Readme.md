# Botova: AI-Powered Chatbot Platform

Botova is an innovative platform that leverages artificial intelligence to create intelligent and adaptive chatbots. The name "Botova" combines “bot” with the Latin suffix “-ova,” symbolizing change, transformation, and evolution—capturing the essence of dynamic AI-driven bots that evolve and adapt to meet the needs of businesses and users.

## Project Overview

Botova is designed to transform the way bots interact with users and handle complex tasks. It enables context-aware conversations, automates tasks such as HR operations, and converts raw data into actionable insights through dynamic text-to-SQL queries and reports. The platform is versatile, capable of handling a wide range of applications.

The "ova" in Botova emphasizes its adaptability and continuous improvement. Unlike static systems, the bots in Botova grow smarter with every interaction, learning and evolving to provide increasingly accurate and relevant responses. This flexibility allows businesses to deploy bots that not only offer intelligence but also evolve with the changing demands of their industries.

## Future Vision

Botova aims to become a fully-fledged SaaS platform, empowering businesses to harness the power of AI in diverse applications.

## Project Structure

- **main.py**: Initializes the FastAPI server and configures chatbot endpoints.
- **chat_ui.html**: Provides a user interface to interact with the chatbots.
- **conversational_bot.py**: Handles straightforward conversational responses using OpenAI's language model.
- **rag_bot.py**: Implements Retrieval-Augmented Generation, utilizing FAISS for efficient document retrieval and response generation.
- **tool_bot.py**: Integrates external tools, such as order details and weather APIs, to enhance responses.
- **static/**: Contains additional static files like CSS, JS, and images, if any.

## Setup and Installation

### Requirements

- Python 3.x
- FastAPI
- Uvicorn
- LangChain
- dotenv

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/OMAR-EHAB777/Botova-Ai.git
    cd Botova-Ai
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Set up your environment variables:**

    Create a `.env` file and add your OpenAI API key:

    ```
    OPENAI_API_KEY=your_openai_api_key
    ```

4. **Run the FastAPI server:**

    ```bash
    uvicorn main:app --reload
    ```

5. **Serve the static HTML file:**

    Navigate to the `Front` directory and run a local server:

    ```bash
    python -m http.server 8001
    ```

    Access the chat UI at `http://localhost:8001/chat_ui.html`.

## Usage

- **Switch between chatbot types**: Use the navbar buttons to select Conversational, RAG, or Tool modes.
- **Dark Mode**: Toggle dark mode with the moon/sun icon.
- **Chat**: Type messages and interact with the AI using the designated input field.
- **Save Chat History**: Download chat history as a PDF.
- **Clear Chat**: Reset the chat window.

## Troubleshooting

If you encounter issues, ensure:

- Both the FastAPI server and the static server are running correctly.
- CORS is properly configured in `main.py`.
- URLs in the fetch requests point to the correct FastAPI server.

## Future Enhancements

- Add more advanced NLP tasks.
- Expand tool functionalities to integrate with real APIs.
- Improve UI for a better user experience.

## License

This project is open-source and available under the [MIT License](LICENSE).
