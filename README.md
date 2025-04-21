# Weather MCP Server Example

This repository contains a simple FastAPI-based server demonstrating the Model Context Protocol (MCP) for retrieving weather information. It provides a backend that can be inspected and interacted with using the MCP Inspector tool.

## Architecture Overview

This server implements the Model Context Protocol (MCP) over standard input/output (STDIO) when launched via a tool like the MCP Inspector. It listens for JSON-RPC 2.0 requests conforming to the MCP specification and responds accordingly.

The core functionality involves:

1.  **Receiving MCP Requests:** Parsing JSON-RPC messages from standard input.
2.  **Handling `get_tools`:** Responding with the available `get_current_weather` tool definition.
3.  **Handling `call_tool`:** Executing the `get_current_weather` function based on the provided parameters (location, unit) and returning the result.
4.  **Sending MCP Responses:** Writing JSON-RPC responses back to standard output.

## Key Features

*   **MCP Implementation:** Adheres to the Model Context Protocol specification.
*   **Tool Definition:** Exposes a `get_current_weather` tool.
*   **Simple Tool Logic:** Provides hardcoded weather data for demonstration purposes.
*   **FastAPI & Pydantic:** Built using FastAPI for structure and Pydantic for data validation (though HTTP endpoints are not the primary interaction method when used with MCP Inspector via STDIO).
*   **STDIO Transport:** Designed to communicate via standard input/output when launched by an MCP client/inspector.

## Setup

1.  **Clone Repository:**
    ```bash
    git clone <repository-url> # Replace with your repo URL
    cd weather-mcp-server
    ```

2.  **Install Dependencies:**
    Use a virtual environment:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
    Install required packages:
    ```bash
    pip install -r requirements.txt
    # Or install manually:
    # pip install fastapi uvicorn pydantic
    # pip install uv # If you want to use uv as the server runner
    ```

## Running with MCP Inspector

The primary way to interact with this server is through the MCP Inspector.

1.  **Install MCP Inspector:**
    ```bash
    npm install -g @modelcontextprotocol/inspector
    ```

2.  **Launch Inspector:**
    ```bash
    npx @modelcontextprotocol/inspector
    # or simply `inspector` if installed globally
    ```

3.  **Configure Connection in Inspector UI:**
    *   **Transport Type:** Select `STDIO`.
    *   **Command:** Enter `uv` (or `python` if not using `uv`).
    *   **Arguments:** Enter `run src/weather/server.py` (if using `uv`) or `src/weather/server.py` (if using `python`).
    *   **Working Directory (CWD):** **Crucially, set this to the root directory of this project** (`D:\project\weather-mcp-server` or wherever you cloned it).
        *   *If the CWD option isn't directly available, you might need to use a wrapper script (see README troubleshooting section if needed).* 
    *   Click **Connect**.

4.  **Interact via Inspector:**
    *   Once connected, you can send MCP requests like `get_tools` or `call_tool` using the Inspector's UI.
    *   Example `call_tool` request body:
        ```json
        {
          "jsonrpc": "2.0",
          "method": "call_tool",
          "params": {
            "tool_name": "get_current_weather",
            "parameters": {
              "location": "London",
              "unit": "celsius"
            }
          },
          "id": 1
        }
        ```

## Running Standalone (for basic HTTP check - less relevant for MCP)

You can run the FastAPI server directly to check if the basic setup is working, but it won't respond to MCP requests this way.

```bash
# Make sure you are in the project root directory (weather-mcp-server)
# Using uv
uv run src.weather.server:app --reload 

# Using uvicorn (standard FastAPI runner)
# uvicorn src.weather.server:app --reload
```
This will start a standard HTTP server (e.g., on `http://127.0.0.1:8000`), but the MCP logic relies on STDIO communication when launched via the Inspector.

## Project Structure

*   `src/weather/server.py`: Main application logic, handles MCP requests via STDIO.
*   `src/weather/tools.py`: Defines the `get_current_weather` tool.
*   `src/weather/models.py`: Pydantic models for request/response validation.
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore.
*   `requirements.txt`: Lists project dependencies.
*   `README.md`: This file.

## Troubleshooting

*   **Inspector Connection Error ("File not found"):** Ensure the **Working Directory (CWD)** in the Inspector is set correctly to the project's root folder (`weather-mcp-server`). If the CWD option is unavailable, consider using a wrapper script (e.g., `run.bat` or `run.sh`) that first changes the directory and then executes the server command (`uv run ...` or `python ...`), and point the Inspector's **Command** field to this script. 