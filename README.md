# LLM Inference Server via ChatOpenAI (Starlette + WebSocket)

This server provides an interface to OpenAI-compatible LLM APIs (including DeepSeek, OpenAI itself, and others) using Langchain's `ChatOpenAI` client. It uses a WebSocket connection for interactive generation with optional tool support and is built with the Starlette web framework.

## Features

*   **Model Client:** Uses `langchain_openai.ChatOpenAI` for broad compatibility with OpenAI-spec APIs.
*   **Configurable Endpoint:** Target different LLMs (DeepSeek, OpenAI, etc.) by changing environment variables for the API Base URL, API Key, and Model Name.
*   **Langchain Integration:** Leverages `langchain` for LLM interaction.
*   **Interface:** Primarily uses WebSocket with JSON-RPC 2.0 messages for interactive streaming generation.
*   **Tool Use:** Supports defining tools (OpenAI JSON Schema compatible) and handling function calls during generation.
*   **Framework:** Built with Starlette, using Pydantic models (`models.py`) for validation.
*   **Logging:** Outputs logs to `inference_server.log` (rotating) and the console.
*   **Dependency Management:** Uses `uv` and `pyproject.toml`.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd inference-server
    ```

2.  **Create and Activate Virtual Environment (using uv):**
    ```bash
    uv venv
    source .venv/bin/activate 
    ```

3.  **Install Dependencies:**
    ```bash
    uv pip install -e .
    ```
    This installs the project and its dependencies (including `langchain-openai`, `starlette`, `uvicorn`, etc.) in editable mode.

4.  **Configure Environment Variables:**
    Create a `.env` file in the project root directory. Configure it for your target LLM API.

    **Example for DeepSeek:**
    ```dotenv
    # .env (Example for DeepSeek)
    API_KEY_ENV_VAR="DEEPSEEK_API_KEY"  # Tells server which env var holds the key
    DEEPSEEK_API_KEY="your_deepseek_api_key" # Your actual key
    
    LLM_BASE_URL="https://api.deepseek.com/v1" # DeepSeek API endpoint
    LLM_MODEL_NAME="deepseek-chat"          # Specific DeepSeek model
    ```

    **Example for OpenAI:**
    ```dotenv
    # .env (Example for OpenAI)
    API_KEY_ENV_VAR="OPENAI_API_KEY"
    OPENAI_API_KEY="your_openai_api_key"
    
    # LLM_BASE_URL is optional for OpenAI (defaults work), but can be set if needed
    # LLM_BASE_URL="https://api.openai.com/v1"
    LLM_MODEL_NAME="gpt-4o" 
    ```
    
    *   `API_KEY_ENV_VAR` (Optional): Specifies the *name* of the environment variable that contains the actual API key. Defaults to `DEEPSEEK_API_KEY` if not set. Set this to `OPENAI_API_KEY` if using OpenAI.
    *   `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / etc.: Provide the actual API key using the variable name you specified in `API_KEY_ENV_VAR`.
    *   `LLM_BASE_URL` (Optional): The base URL for the API endpoint. Defaults to `https://api.deepseek.com/v1` if not set. Use `https://api.openai.com/v1` for standard OpenAI.
    *   `LLM_MODEL_NAME` (Optional): The specific model identifier for the target API. Defaults to `deepseek-chat` if not set.

## Running the Server

```bash
uv run python -m inference_server_lib.server_starlette
```

The server will start, initialize the `ChatOpenAI` client targeting the configured endpoint, and typically listen on `http://0.0.0.0:8000`.

## Usage (WebSocket Client)

Connect to the WebSocket endpoint: `ws://localhost:8000/ws/{session_id}`

1.  **Create a Session (HTTP POST):**
    Send a POST request to `/create_session` with an optional JSON body defining tools:
    ```json
    // Example request body (optional tools)
    {
      "tools": [
        {
          "name": "get_weather",
          "description": "Get the current weather for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": { "type": "string", "description": "City name" }
            },
            "required": ["location"]
          }
        }
      ]
    }
    ```
    The response will contain a `session_id`.

2.  **Connect WebSocket:**
    Use the obtained `session_id` to connect to `ws://localhost:8000/ws/{session_id}`.

3.  **Send Messages (JSON-RPC over WebSocket):**

    *   **Generate Text:**
        ```json
        {
          "jsonrpc": "2.0",
          "id": 1, 
          "method": "generate",
          "params": {
            "history": [
              { "role": "user", "content": "Hello, tell me a joke." }
            ]
            // Optional: "message": "Hello..." (if history is empty)
            // Optional: "tools": [...] (override session tools for this request)
            // Optional: generation parameters like "max_new_tokens", "temperature"
          }
        }
        ```

    *   **Send Tool Result (if requested by server):**
        ```json
        {
          "jsonrpc": "2.0",
          "id": 5, 
          "method": "tool_result",
          "params": {
            "task_id": "task_xyz123", // ID from the function_call_request notification
            "results": [
              {
                "tool_call_id": "call_abc456", // ID from the function_call_request notification
                "tool_name": "get_weather",
                "result": {"temperature": "15 C", "condition": "Cloudy"} 
                // OR if error: "is_error": true, "error_message": "API limit reached"
              }
              // Can include multiple results if multiple tool calls were requested concurrently
            ]
          }
        }
        ```

    *   **Cancel Generation (Optional):**
        ```json
        {"jsonrpc": "2.0", "id": 99, "method": "cancel"}
        ```

4.  **Receive Notifications (JSON-RPC over WebSocket):**
    The server sends notifications (messages without an `id`) for events:
    *   `text_chunk`: `{ "session_id": "...", "task_id": "...", "chunk": "..." }`
    *   `function_call_request`: `{ "session_id": "...", "task_id": "...", "request_id": "call_abc456", "tool_name": "...", "parameters": {...} }`
    *   `end`: `{ "session_id": "...", "task_id": "...", "status_message": "...", "final_text": "..." }` (final_text included on completion)
    *   `error`: `{ "session_id": "...", "task_id": "...", "error": { "code": ..., "message": "...", "data": ... } }`

## Project Structure

*   `src/inference_server_lib/`: Contains the core server code.
    *   `server_starlette.py`: Main Starlette application, WebSocket handling.
    *   `shared.py`: Shared state, LLM client initialization (`initialize_llm`), logging setup.
    *   `utils.py`: Helper functions for Langchain message conversion and streaming generation.
    *   `models.py`: Pydantic models for API validation.
*   `pyproject.toml`: Project metadata and dependencies (managed with `uv`).
*   `.env` (gitignored): For storing API keys, base URLs, model names.
*   `README.md`: This file.
*   `inference_server.log`: Log file output.
*   `.gitignore`: Specifies files ignored by Git.

## Notes

*   Ensure your API key for the chosen `LLM_PROVIDER` (OpenAI, Anthropic, DeepSeek, etc.) is correctly set in your `.env` file.
*   Default models are specified in `shared.py` if `LLM_MODEL_NAME` is not set for a provider.
*   To add support for another LLM provider:
    1.  Install its Langchain integration package (e.g., `uv pip install langchain-google-genai`).
    2.  Add the package to `pyproject.toml` dependencies.
    3.  Update `shared.py` in the `initialize_llm` function with a new `elif provider == "yourprovider":` block.
    4.  Set the `LLM_PROVIDER` and its API key in your `.env` file.
*   Error handling exists, but review logs (`inference_server.log`) for detailed debugging information. 