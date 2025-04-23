# Qwen-14B Inference Server (Starlette + WebSocket)

This server provides an interface to a locally hosted Qwen-14B transformer model, primarily using a WebSocket connection for interactive generation with optional tool support. It uses the Starlette web framework and leverages libraries like `transformers`, `torch`, `accelerate`, and `bitsandbytes` for model loading and inference.

## Features

*   **Model:** Serves the Qwen-14B model (configured to load from the local `./Qwen-14B` directory).
*   **Quantization:** Uses 4-bit quantization via `bitsandbytes` for potentially lower VRAM usage.
*   **Interface:** Primarily uses WebSocket with JSON-RPC 2.0 messages for interactive streaming generation.
*   **Tool Use:** Supports defining tools during session creation and handling function calls during generation.
*   **Framework:** Built with Starlette, using Pydantic models (`models.py`) for validation.
*   **Logging:** Outputs logs to `inference_server.log` (rotating) and the console.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Model Download:**
    *   This server expects the Qwen-14B model files to be present in a directory named `Qwen-14B` within the project root.
    *   You can download the model using `huggingface-cli` or `git lfs`:
        ```bash
        # Example using huggingface-cli (ensure you are logged in: huggingface-cli login)
        huggingface-cli download --repo-type model Qwen/Qwen1.5-14B --local-dir Qwen-14B --local-dir-use-symlinks False

        # Or using git lfs (requires git-lfs installed)
        # git lfs install
        # git clone https://huggingface.co/Qwen/Qwen1.5-14B Qwen-14B # Adjust model ID if needed
        ```
    *   Verify the `MODEL_ID` variable in `shared.py` points correctly to `./Qwen-14B`.

3.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure you have compatible PyTorch/CUDA versions installed if using GPU acceleration).*

## Running the Server

Execute the main server script:

```bash
python server_starlette.py
```

The server will start, load the model (this might take time and significant RAM/VRAM), and listen on `0.0.0.0:8000`. You should see log output indicating the server status and model loading progress/completion.

## Usage (API)

Interaction primarily happens over WebSocket after creating a session via HTTP.

### 1. Create a Session (HTTP)

Send a POST request to `/create_session`.

*   **Endpoint:** `POST /create_session`
*   **Request Body (optional, `application/json`):**
    ```json
    {
      "tools": [
        {
          "name": "your_tool_name",
          "description": "Description of what the tool does.",
          "parameters": {
            "type": "object",
            "properties": {
              "param1": { "type": "string", "description": "Param 1 description" },
              "param2": { "type": "integer", "description": "Param 2 description" }
            },
            "required": ["param1"]
          }
        }
        // Add more tools if needed
      ]
    }
    ```
*   **Response (`application/json`):**
    ```json
    {
      "session_id": "some-unique-session-id-uuid"
    }
    ```
    Store the `session_id` for the WebSocket connection.

### 2. Connect via WebSocket

Establish a WebSocket connection to:

`ws://<server_host>:8000/ws/{session_id}`

Replace `<server_host>` with the server's IP address or hostname (e.g., `localhost`, `127.0.0.1`) and `{session_id}` with the ID obtained from `/create_session`.

### 3. Communicate via JSON-RPC (WebSocket)

Send and receive JSON-RPC 2.0 messages over the established WebSocket connection.

**Client -> Server Methods:**

*   **`generate`:** Starts a generation task.
    *   Params (`GenerateRequestParams` in `models.py`):
        *   `message`: (Optional) The initial user message (string).
        *   `history`: (Optional) List of previous messages (`List[HistoryMessage]`). Use either `message` or `history`.
        *   `tools`: (Optional) Override the tools registered with the session for this specific request.
        *   `max_new_tokens`, `temperature`, `do_sample`, etc. (see `models.py` for details).
    *   Example Request:
        ```json
        {
          "jsonrpc": "2.0",
          "method": "generate",
          "params": {
            "message": "What is the weather like in London?",
            "max_new_tokens": 512
          },
          "id": 1
        }
        ```
    *   Response (Initial): `{"jsonrpc": "2.0", "result": {"status": "processing"}, "id": 1}`

*   **`tool_result`:** Sends the result of a tool call requested by the model.
    *   Params (`ToolResultParams`):
        *   `task_id`: The `task_id` received in the `function_call_request` notification.
        *   `results`: A list containing one result object.
            *   `tool_name`: Name of the tool that was called.
            *   `result`: The data returned by the tool execution.
            *   `isError`: Boolean indicating if the tool execution resulted in an error.
            *   `error_message`: (Optional) String error message if `isError` is true.
    *   Example Request:
        ```json
        {
          "jsonrpc": "2.0",
          "method": "tool_result",
          "params": {
            "task_id": "task-id-from-llm",
            "results": [
              {
                "tool_name": "get_weather",
                "result": {"temperature": "15°C", "condition": "Cloudy"},
                "isError": false
              }
            ]
          },
          "id": 2
        }
        ```
    *   Response: `{"jsonrpc": "2.0", "result": {"status": "received"}, "id": 2}` (Server will then continue generation).

*   **`cancel`:** Attempts to cancel the currently active `generate` task.
    *   Params: None
    *   Example Request: `{"jsonrpc": "2.0", "method": "cancel", "id": 3}`
    *   Response: `{"jsonrpc": "2.0", "result": {"status": "cancelled"}, "id": 3}` or error if no task active.

**Server -> Client Notifications (Methods):**

*   **`text_chunk`:** A piece of generated text.
    *   Params (`TextChunkParams`): `session_id`, `task_id`, `chunk`.
*   **`function_call_request`:** Model requests a tool/function call.
    *   Params (`FunctionCallRequestParams`): `session_id`, `task_id`, `tool_calls` (list of `ToolCallRequest` objects with `tool` name and `parameters`).
*   **`final_text`:** The complete final text response for a turn (after potential tool calls).
    *   Params (`StreamEndParams`): `session_id`, `task_id`, `final_text`.
*   **`end`:** Signals the end of the generation stream for the current request.
    *   Params (`StreamEndParams`): `session_id`, `task_id`.
*   **`error`:** Indicates an error occurred during processing.
    *   Params (`ErrorNotificationParams`): `session_id`, `task_id` (optional), `error` (JsonRpcError object with `code`, `message`).

## Project Structure

*   `server_starlette.py`: Main Starlette application, WebSocket logic, HTTP endpoints.
*   `shared.py`: Shared state, model/tokenizer loading (`load_model`), logging setup (`setup_logging`), `MODEL_ID` configuration.
*   `utils.py`: Helper functions, including prompt generation (`generate_conversation_prompt`) and the core streaming generation logic (`generate_interactive_stream_ws`).
*   `models.py`: Pydantic models for API request/response validation and internal data structures.
*   `tasks.py`: (Currently seems minimal/unused).
*   `Qwen-14B/`: Directory containing the local Qwen-14B model files (needs to be downloaded).
*   `requirements.txt`: Python dependencies.
*   `inference_server.log`: Log file output.
*   `.gitignore`: Specifies files ignored by Git.

## Notes

*   Ensure sufficient RAM and VRAM are available for loading and running the Qwen-14B model, even with quantization.
*   The model path is hardcoded in `shared.py` to `./Qwen-14B`. Adjust if needed.
*   Error handling exists, but review logs (`inference_server.log`) for detailed debugging information. 