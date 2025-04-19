# LLM Inference Server with Tool Usage

This repository contains a FastAPI-based inference server designed to interact with Large Language Models (LLMs) like Qwen-14B. It provides a backend for conversational AI applications, enabling multi-turn dialogue and the integration of external tools.

## Key Features

*   **JSON-RPC 2.0 Interface:** Uses a standard JSON-RPC endpoint (`/rpc`) for structured communication between a client and the LLM server.
*   **Multi-Turn Conversation:** Manages conversation history (`history`) allowing for contextually relevant interactions over multiple turns.
*   **Tool Calling Support:** Implements a mechanism for the LLM to request the execution of predefined tools. The server requests the tool call, the client executes it, and returns the result.
*   **Streaming Endpoint:** Provides a Server-Sent Events (SSE) endpoint (`/stream`) for receiving generated text tokens in real-time.
*   **Configurable LLM:** Easily configure the LLM model to use (via Hugging Face ID or local path) within `server.py`. Uses 4-bit quantization (`bitsandbytes`) for potentially reduced resource usage.
*   **Loop Prevention:** Includes logic (prompt engineering and server-side checks) to mitigate potential issues like repetitive tool calling loops.
*   **Built with FastAPI:** Leverages the high-performance FastAPI framework and Uvicorn ASGI server.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd inference-server
    ```

2.  **Install Dependencies:**
    It's recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\\Scripts\\activate`
    ```
    *(Suggestion: Create a `requirements.txt` file with the necessary packages)*. You will likely need:
    ```bash
    pip install fastapi uvicorn "pydantic>=2.0" "transformers>=4.38.0" "torch>=2.1.0" accelerate bitsandbytes sentencepiece Jinja2
    # Add any other specific dependencies for your chosen model or tools
    ```
    *Note: Ensure you have the correct version of PyTorch compatible with your CUDA version if using a GPU.*

3.  **Configure Model:**
    *   Open `server.py`.
    *   Modify the `MODEL_ID` variable:
        *   **Option 1 (Hugging Face ID):** Set it to the model ID (e.g., `"Qwen/Qwen1.5-14B-Chat"`). The model will be downloaded on the first run if not cached.
        *   **Option 2 (Local Path):** Set it to the relative or absolute path of your locally downloaded model files (e.g., `"./Qwen-14B"`). Ensure the model files exist at this location.
    *   Make sure the correct option (local path or ID) is uncommented.

4.  **Hardware Requirements:**
    *   Running large models, even quantized ones, requires significant resources.
    *   A CUDA-enabled GPU with sufficient VRAM (e.g., >10GB for a 14B 4-bit model) is highly recommended for reasonable performance.
    *   Ensure you have the necessary CUDA toolkit and drivers installed if using a GPU.

## Running the Server

Execute the main server script:

```bash
python server.py
```

The server will start, load the model (which might take time), and begin listening on `http://0.0.0.0:8000` by default. You'll see logs indicating the model loading progress and when the server is ready.

## API Endpoints

### 1. `/rpc` (POST)

Handles JSON-RPC 2.0 requests for multi-turn chat and tool interactions.

**Methods:**

*   **`create_message`:**
    *   **Purpose:** The primary method for conversing with the LLM, handling history, tool results, and generating the next response (either final text or a tool call request).
    *   **Params:** See `CreateMessageParams` in `server.py` (includes `message`, `history`, `tools`, `tool_results`, `max_new_tokens`, etc.).
    *   **Result:** See `CreateMessageResult` in `server.py` (includes `type` ["final_text" or "tool_calls"], `content`, `history`, `iteration_count`).
    *   **Flow:**
        1.  Client sends user message (or history + tool results).
        2.  Server generates prompt including history, tool definitions, and instructions.
        3.  LLM generates response.
        4.  Server parses response. If it's a valid tool call request (and allowed), returns `type: "tool_calls"`. Otherwise, returns `type: "final_text"`.
        5.  If `tool_calls` received, client executes tools and sends results back via another `create_message` call with `tool_results`.
*   **`tool_list`:** (Currently returns an empty list, tools should be provided in `create_message`)
    *   **Purpose:** Intended to list available tools (though currently implemented to expect tools within the `create_message` call).
    *   **Params:** None.
    *   **Result:** `ToolListResult` containing a `tools` list.

### 2. `/stream` (POST)

Handles requests for streaming LLM responses using Server-Sent Events (SSE).

*   **Purpose:** Get real-time token generation from the LLM. Suitable for displaying responses progressively.
*   **Params:** Uses `CreateMessageParams` similar to `create_message`, but tool results are *not* processed server-side in this endpoint. The client must manage the history including tool results before making the `/stream` request.
*   **Response:** `text/event-stream` containing SSE messages. Each message typically has the format `data: {"delta": "new text chunk"}\n\n`. Errors might be sent as `data: {"error": "error message"}\n\n`.

## Project Structure

*   `server.py`: The main FastAPI application file containing the server logic, model loading, API endpoints, and helper functions.
*   `(Suggested) requirements.txt`: File listing Python dependencies.
*   `README.md`: This file.
*   `Qwen-14B/` (Example local model folder - if using Option 2 in Setup):
    ```text
    Qwen-14B/
    ├── .cache/             # Optional cache files
    │   └── huggingface/
    ├── figures/            # Optional figure files from model repo
    ├── config.json
    ├── generation_config.json
    ├── model-00001-of-00004.safetensors
    ├── model-00002-of-00004.safetensors
    ├── model-00003-of-00004.safetensors
    ├── model-00004-of-00004.safetensors
    ├── model.safetensors.index.json
    ├── tokenizer_config.json
    └── tokenizer.json
    ``` 