# LLM Inference Server with Tool Usage (using Celery)

This repository contains a FastAPI-based inference server designed to interact with Large Language Models (LLMs) like Qwen-14B. It provides a backend for conversational AI applications, enabling multi-turn dialogue, integration of external tools, and real-time streaming output via Celery background tasks.

## Architecture Overview

This server employs a dual-protocol approach:

1.  **JSON-RPC 2.0 (`/rpc` endpoint):** Used for **initiating** potentially long-running tasks like LLM inference. It immediately returns a Celery Task ID.
2.  **Server-Sent Events (SSE) (`/stream/{taskId}` endpoint):** Used for **receiving the results** of a previously initiated task. The client connects using the Celery Task ID to get status updates (`PENDING`, `STARTED`, `SUCCESS`, `FAILURE`) and the final result.

The core LLM processing is offloaded to background **Celery workers**, using **Redis** as both the message broker and the result backend. The FastAPI server handles HTTP requests, sends tasks to the Celery broker via Redis, and streams results via SSE by polling the task status/result from the Redis backend.

## Key Features

*   **Dual Protocol:** Asynchronous task initiation via JSON-RPC (`/rpc`) and result streaming via SSE (`/stream/{taskId}`).
*   **Background Task Processing:** Uses **Celery and Redis** to offload LLM inference to separate worker processes.
*   **Task State Polling:** The `/stream` endpoint polls the Celery task status (`PENDING`, `STARTED`, `SUCCESS`, `FAILURE`) and streams updates and the final result.
*   **Multi-Turn Conversation & Tool Calling:** Logic remains in prompt generation; execution is asynchronous.
*   **Configurable LLM & Loop Prevention:** Safeguards remain in prompt generation.
*   **Built with FastAPI & Celery:** Combines FastAPI, Celery, and Redis.

## Workflow

1.  **Client Request (`/rpc`):** Sends JSON-RPC request (`create_message`) with history/message.
2.  **Server Sends Task (`/rpc`):**
    *   Generates LLM prompt.
    *   Sends the task to the Celery broker (Redis) using `task.delay()`.
    *   **Immediately returns** JSON-RPC response with the unique `taskId` (Celery Task ID).
3.  **Worker Processing (Background):**
    *   A Celery worker process picks up the task from the Redis broker.
    *   Initializes the LLM pipeline if not already done in that worker process.
    *   Executes the LLM inference task (`run_llm_inference_task`).
    *   Saves the result dictionary (containing `cleaned_response_text`, etc.) back to the Redis result backend.
4.  **Client Connects to Stream (`/stream/{taskId}`):** Connects via SSE using the `taskId`.
5.  **Server Polls & Streams Results (`/stream/{taskId}`):**
    *   The server endpoint gets a Celery `AsyncResult` object using the `taskId`.
    *   Periodically polls `async_result.state`.
    *   Sends status updates (`PENDING`, `STARTED`, etc.) via SSE.
    *   When state is `SUCCESS`:
        *   Retrieves the result dictionary (`async_result.result`).
        *   Parses the result (checks for tool calls, etc.).
        *   Sends the final result via SSE (`{"type": "final_text", ...}` or `{"type": "tool_calls", ...}`).
    *   When state is `FAILURE`:
        *   Retrieves error info (`async_result.traceback`).
        *   Sends an SSE error event.
    *   Sends an SSE end event when finished or failed.
6.  **Tool Execution (Client-Side):** If result is `tool_calls`, client executes them.
7.  **Loop:** Client sends tool results back via a *new* `/rpc` call, starting a new task.

## Setup

1.  **Clone Repository:**
    ```bash
    git clone <repository-url>
    cd inference-server
    ```

2.  **Install Dependencies:**
    Use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate # Windows: venv\\Scripts\\activate
    ```
    Install required packages (create `requirements.txt` recommended):
    ```bash
    # Core FastAPI, Transformers, Torch
    pip install fastapi uvicorn "pydantic>=2.0" "transformers>=4.38.0" "torch>=2.1.0" accelerate bitsandbytes sentencepiece Jinja2 
    # SSE Streaming
    pip install sse-starlette
    # Celery with Redis support
    pip install "celery[redis]>=5.0" 
    ```
    *Note: Ensure correct PyTorch/CUDA version.*

3.  **Run Redis Server:**
    Celery uses Redis as the message broker and result backend. Ensure Redis server is running and accessible (default: `localhost:6379`).
    *   Installation: [https://redis.io/docs/getting-started/installation/](https://redis.io/docs/getting-started/installation/)
    *   Running: `redis-server`

4.  **Configure Model:**
    *   Open `shared.py`.
    *   Modify the `MODEL_ID` variable if needed.

5.  **Hardware Requirements:**
    *   Significant RAM/VRAM, CUDA toolkit/drivers for GPU.

## Running the Server and Worker

Run two separate processes:

1.  **FastAPI Server:**
    ```bash
    python server.py
    ```
    Handles HTTP requests, sends tasks to Celery.

2.  **Celery Worker:**
    Open *another terminal* (same directory & environment). Run:
    ```bash
    # Basic command (adjust log level as needed):
    celery -A celery_app worker --loglevel=info 
    
    # On Windows, you might need the 'eventlet' or 'gevent' pool 
    # if you encounter concurrency issues (install them first: pip install eventlet/gevent):
    # celery -A celery_app worker --loglevel=info -P eventlet 
    # celery -A celery_app worker --loglevel=info -P gevent 
    ```
    This connects to Redis, discovers tasks (like `run_llm_inference_task`), and executes them.

Server listens on `http://0.0.0.0:8000` by default.

## API Endpoints

### 1. `/rpc` (POST)

Initiates LLM tasks asynchronously via JSON-RPC 2.0.

**Methods:**

*   **`create_message`:** Takes history/message/tools, sends task to Celery, **returns `{"taskId": "<celery_task_id>"}`**.
*   **`tool_list`:** Returns empty list.

### 2. `/stream/{task_id}` (GET)

Streams status (`PENDING`, `STARTED`, `SUCCESS`, `FAILURE`) and final result of a Celery task via SSE. Uses the `taskId` from `/rpc`.

## Project Structure

*   `server.py`: Main FastAPI application.
*   `tasks.py`: Defines Celery task functions.
*   **`celery_app.py`**: Defines the Celery application instance.
*   `utils.py`: Helper functions.
*   `shared.py`: Shared state (LLM pipeline, model ID).
*   `models.py`: Pydantic models.
*   `inference_server.log`: Log file.
*   `(Suggested) requirements.txt`: Dependencies.
*   `README.md`: This file.
*   `Qwen-14B/`: Example local model folder.

## Future Considerations

*   Multiple Queues: Route different task types to different Celery queues/workers.
*   Monitoring: Use Celery Flower or integrate with Prometheus/Grafana.
*   Error Handling: More sophisticated retry logic in Celery tasks.

## Project Structure

*   `server.py`: Main FastAPI application (loads model, handles HTTP endpoints, manages RQ queue).
*   `tasks.py`: Defines the background task function (`run_llm_inference_task`) executed by RQ workers.
*   `utils.py`: Contains helper functions (prompt generation, response parsing).
*   `shared.py`: Holds shared state (like the loaded LLM pipeline) accessible by both server and workers.
*   `inference_server.log`: Log file where server and worker output is written (created on first run).
*   `(Suggested) requirements.txt`: Dependencies.
*   `README.md`: This file.
*   `Qwen-14B/` (Example local model folder):
    ```text
    Qwen-14B/
    ├── .cache/             # Optional cache files
    │   └── huggingface/
    ├── figures/            # Optional figure files from model repo
    ├── config.json
    ├── generation_config.json
    ├── model-*.safetensors
    ├── model.safetensors.index.json
    ├── tokenizer_config.json
    └── tokenizer.json
    ```

## Future Considerations

(Ideas inspired by generic dual-protocol patterns, not currently implemented)

*   **External State Management:** Using Redis or similar to manage conversation state if scaling to multiple server instances.
*   **Task Queues:** Offloading long-running tool executions or complex processing.
*   **Advanced Streaming Control:** Implementing backpressure if clients consume SSE streams too slowly.
*   **Enhanced Error Reporting:** More detailed error objects sent via SSE.
*   **Monitoring:** Integrating Prometheus/Grafana for monitoring request latency, SSE connections, resource usage, etc. 