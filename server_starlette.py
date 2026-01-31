import asyncio
import json
import uuid
import logging
import os
from typing import Any, Optional, Union, Dict, List

# Starlette imports
from starlette.applications import Starlette
from starlette.routing import Route, WebSocketRoute
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.websockets import WebSocket, WebSocketDisconnect

# Keep our existing modules
import shared
from shared import setup_logging, MODEL_ID
from utils import generate_interactive_stream_ws
from models import *

setup_logging()
logging.info("*** Starlette WebSocket Server Starting Up - Logging Configured ***")

# --- Session Management (Keep) ---
SESSIONS: Dict[str, Dict[str, Any]] = {}

# --- Pydantic Models ---

# --- Helper to send JSON-RPC Error Response over WebSocket ---
async def send_jsonrpc_error(websocket: WebSocket, code: int, message: str, request_id: Optional[Union[str, int]] = None, data: Optional[Any] = None):
    error_resp = {
        "jsonrpc": "2.0",
        "error": {"code": code, "message": message},
        "id": request_id
    }
    if data:
        error_resp["error"]["data"] = data
    try:
        await websocket.send_text(json.dumps(error_resp))
    except Exception as e:
        # Log potential error during send, but might happen if WS closed
        logging.warning(f"Failed to send error response over WebSocket (WS might be closed): {e}")

# --- WebSocket Endpoint Handler ---
async def websocket_endpoint(websocket: WebSocket):
    session_id = websocket.path_params.get('session_id')
    ws_log_prefix = f"[WS Session {session_id}]"
    active_generation_task : Optional[asyncio.Task] = None # Track task for cancellation
    current_history: List[HistoryMessage] = [] # Maintain history within the connection
    current_tools: Optional[List[Dict[str, Any]]] = None # Store tools for the session
    current_iteration: int = 0 # Track iteration count within the session
    
    # 1. Accept Connection & Validate Session
    try:
        await websocket.accept()
        logging.info(f"{ws_log_prefix} WebSocket connected.")
        
        session_data = SESSIONS.get(session_id)
        if not session_data:
            logging.warning(f"{ws_log_prefix} Invalid session ID on connect.")
            # Send JSON-RPC error notification (no ID for connection errors)
            error_notif = {"jsonrpc": "2.0", "method": "error", "params": ErrorNotificationParams(error=JsonRpcError(code=-32000, message="Invalid session ID")).model_dump()}
            await websocket.send_text(json.dumps(error_notif))
            await websocket.close(code=1008) # Policy violation
            return
            
        # Retrieve session tools and initialize history/iteration
        session_tools = session_data.get("tools")
        current_tools = session_tools # Store for later use
        current_history = [] # Start with empty history for this connection
        current_iteration = 0
        logging.info(f"{ws_log_prefix} Session validated. Tools available: {bool(session_tools)}")

    except WebSocketDisconnect:
         logging.info(f"{ws_log_prefix} WebSocket disconnected during handshake.")
         return # Connection closed before we could validate/accept fully
    except Exception as accept_e:
        logging.exception(f"{ws_log_prefix} Error accepting WebSocket connection: {accept_e}")
        # Cannot send error if accept failed, just close if possible
        await websocket.close(code=1011)
        return
        
    # Main message handling loop
    try:
        while True:
            message_text = await websocket.receive_text()
            # ADDED: Log raw representation for encoding check
            logging.info(f"{ws_log_prefix} Received raw text. Type: {type(message_text)}, Repr: {repr(message_text[:50])}...")
            logging.debug(f"{ws_log_prefix} Received message: {message_text[:200]}...")

            request_id : Optional[Union[str,int]] = None # Keep track of ID for responses
            try:
                request_data = json.loads(message_text)
                rpc_request = JsonRpcRequest(**request_data)
                request_id = rpc_request.id

                if rpc_request.method == "generate":
                    if active_generation_task and not active_generation_task.done():
                         logging.warning(f"{ws_log_prefix} Received 'generate' request while another is running. Ignoring.")
                         await send_jsonrpc_error(websocket, JsonRpcErrorCode.INTERNAL_ERROR, "Generation already in progress", request_id)
                         continue

                    logging.info(f"{ws_log_prefix} Received 'generate' request (ID: {request_id}).")
                    try:
                        # Validate parameters using the Pydantic model
                        params = GenerateRequestParams(**(rpc_request.params or {}))

                        # Construct or reset history for this generate request
                        if params.history:
                            current_history = params.history
                            logging.info(f"{ws_log_prefix} Received history with {len(current_history)} messages.")
                        elif params.message:
                            # Start new history if only message provided
                            current_history = [HistoryMessage(role="user", content=params.message)]
                            logging.info(f"{ws_log_prefix} Starting new history with user message.")
                        else:
                            logging.error(f"{ws_log_prefix} 'generate' request rejected: Neither history nor message provided.")
                            await send_jsonrpc_error(websocket, JsonRpcErrorCode.INVALID_PARAMS, "Cannot generate without history or message", request_id)
                            continue

                        # Reset iteration count for a new generate request
                        current_iteration = 0

                        # *** Determine which tools to use for this specific request ***
                        request_tools = params.tools if params.tools is not None else current_tools
                        logging.info(f"{ws_log_prefix} Using tools for this request: {bool(request_tools)}")

                        # Send acknowledgment response for the request
                        ack_resp = {"jsonrpc": "2.0", "result": {"status": "processing"}, "id": request_id}
                        await websocket.send_text(json.dumps(ack_resp))

                        # Start generation task
                        active_generation_task = asyncio.create_task(
                            generate_interactive_stream_ws(
                                session_id=session_id,
                                websocket=websocket,
                                history=current_history, # Use connection's history
                                tools=request_tools,     # *** Use request-specific or session tools ***
                                generation_params=params # Pass validated params
                            )
                        )
                        # Add callback to handle completion/cancellation
                        def _generation_done_callback(fut: asyncio.Future):
                            nonlocal active_generation_task, current_history, current_iteration
                            active_generation_task = None
                            try:
                                result_status = fut.result() # Get status string from generate_interactive_stream_ws
                                logging.info(f"{ws_log_prefix} Generation task finished with status: {result_status}")
                                if result_status == "OK: Function call requested":
                                    # Update history based on LLM response (which should include the assistant turn with the tool call)
                                    # NOTE: generate_interactive_stream_ws needs to return the assistant message containing the tool call
                                    # For now, assume history is updated implicitly by the client/next step
                                    # We might need to explicitly get the last assistant turn from the generator
                                    current_iteration += 1 # Increment iteration count AFTER a successful tool call request
                                    logging.debug(f"{ws_log_prefix} Iteration count incremented to {current_iteration}")
                                elif result_status == "OK: Completed":
                                     # Final text response generated, history potentially updated (need final assistant message)
                                    # Reset iteration count? Or keep it for context?
                                    current_iteration = 0 # Reset on final answer
                                    logging.debug(f"{ws_log_prefix} Final response generated, resetting iteration count.")
                                elif "ERROR" in result_status:
                                    # Error occurred, iteration count not changed
                                    pass
                            except asyncio.CancelledError:
                                logging.info(f"{ws_log_prefix} Generation task was cancelled.")
                            except Exception as e:
                                logging.exception(f"{ws_log_prefix} Exception retrieving generation task result: {e}")
                        active_generation_task.add_done_callback(_generation_done_callback)

                    except Exception as param_e: # Catch Pydantic validation errors etc.
                         logging.error(f"{ws_log_prefix} Invalid params for 'generate': {param_e}")
                         await send_jsonrpc_error(websocket, JsonRpcErrorCode.INVALID_PARAMS, f"Invalid params: {param_e}", request_id)
                
                elif rpc_request.method == "tool_result":
                    logging.info(f"{ws_log_prefix} Received 'tool_result' request (ID: {request_id}).")
                    skip_tool_result = False
                    if active_generation_task and not active_generation_task.done():
                        logging.warning(f"{ws_log_prefix} Received 'tool_result' while generation is running. Ignoring.")
                        await send_jsonrpc_error(websocket, JsonRpcErrorCode.INTERNAL_ERROR, "Cannot process tool result while generation is active", request_id)
                        skip_tool_result = True
                    elif current_iteration == 0:
                        logging.warning(f"{ws_log_prefix} Received 'tool_result' when no tool call was expected (iteration is 0). Ignoring stale result.")
                        ack_resp = {"jsonrpc": "2.0", "result": {"status": "received_ignored"}, "id": request_id}
                        await websocket.send_text(json.dumps(ack_resp))
                        skip_tool_result = True

                    if not skip_tool_result:
                        try:
                            # Validate tool result parameters (Now expects ToolResultParams with results list)
                            params = ToolResultParams(**(rpc_request.params or {}))
                            original_task_id = params.task_id  # Extract the original task ID

                            # Process the results list (currently assuming only one result)
                            if not params.results:
                                logging.warning(f"{ws_log_prefix} Received 'tool_result' with empty results list.")
                                await send_jsonrpc_error(websocket, JsonRpcErrorCode.INVALID_PARAMS, "Received tool_result with no results data", request_id)
                                continue

                            # --- Corrected Logic ---
                            # For now, process only the first result in the list
                            first_result = params.results[0]
                            tool_name = first_result.tool_name
                            result_data = first_result.result
                            error_data = first_result.error_message if first_result.isError else None
                            # --- End Corrected Logic ---

                            logging.debug(f"{ws_log_prefix} Parsed tool result for tool: {tool_name}")

                            # Append tool result to the current history using extracted data
                            tool_message = HistoryMessage(
                                role="tool",
                                tool_name=tool_name,
                                tool_result=result_data,
                                tool_error=error_data,
                            )
                            current_history.append(tool_message)
                            logging.info(f"{ws_log_prefix} Appended tool result for '{tool_name}' to history. New history length: {len(current_history)}")

                            # Send acknowledgment for the tool result
                            ack_resp = {"jsonrpc": "2.0", "result": {"status": "received"}, "id": request_id}
                            await websocket.send_text(json.dumps(ack_resp))

                            # Add follow-up instruction and trigger next generation
                            summary_instruction = HistoryMessage(
                                role="user",
                                content="请基于上述工具调用结果，继续回答或进行总结。",
                            )
                            current_history.append(summary_instruction)
                            logging.info(f"{ws_log_prefix} Appended summary instruction to history. New history length: {len(current_history)}")

                            # Ensure we are not already running a task (safety check)
                            if active_generation_task and not active_generation_task.done():
                                logging.warning(f"{ws_log_prefix} Attempting to restart generation after tool result, but a task is still active. This should not happen.")
                            else:
                                generation_params_for_continuation = GenerateRequestParams(max_new_tokens=512)
                                session_data = SESSIONS.get(session_id, {})
                                tools_for_continuation = session_data.get("tools")
                                logging.info(f"{ws_log_prefix} Triggering generation continuation after tool result.")
                                active_generation_task = asyncio.create_task(
                                    generate_interactive_stream_ws(
                                        session_id=session_id,
                                        websocket=websocket,
                                        history=current_history,
                                        tools=tools_for_continuation,
                                        generation_params=generation_params_for_continuation,
                                    )
                                )
                                active_generation_task.add_done_callback(_generation_done_callback)

                            # Check for empty GitHub result and send direct response if needed
                            is_empty_github_search = False
                            if tool_name == "github@search_repositories" and isinstance(result_data, dict):
                                if not result_data.get("items") and not result_data.get("repositories"):
                                    is_empty_github_search = True
                                elif isinstance(result_data.get("items"), list) and not result_data.get("items"):
                                    is_empty_github_search = True
                                elif isinstance(result_data.get("repositories"), list) and not result_data.get("repositories"):
                                    is_empty_github_search = True

                            if is_empty_github_search:
                                logging.info(f"{ws_log_prefix} Detected empty result for {tool_name}. Sending direct response.")
                                final_text = "Sorry, the search for GitHub repositories didn't return any relevant results."
                                final_params = StreamEndParams(session_id=session_id, task_id=params.task_id, final_text=final_text)
                                final_text_notification = {
                                    "jsonrpc": "2.0",
                                    "method": "final_text",
                                    "params": final_params.model_dump(),
                                }
                                await websocket.send_text(json.dumps(final_text_notification))
                                end_params = StreamEndParams(session_id=session_id, task_id=params.task_id)
                                end_notification = {
                                    "jsonrpc": "2.0",
                                    "method": "end",
                                    "params": end_params.model_dump(),
                                }
                                await websocket.send_text(json.dumps(end_notification))
                                current_iteration = 0
                                logging.debug(f"{ws_log_prefix} Empty tool result handled directly, resetting iteration count.")
                                continue

                        except Exception as tool_e:
                            logging.error(f"{ws_log_prefix} Error processing 'tool_result': {tool_e}")
                            await send_jsonrpc_error(websocket, JsonRpcErrorCode.INVALID_PARAMS, f"Invalid tool_result params or internal error: {tool_e}", request_id)

                elif rpc_request.method == "cancel": # Add a way to cancel
                    logging.info(f"{ws_log_prefix} Received 'cancel' request (ID: {request_id}).")
                    if active_generation_task and not active_generation_task.done():
                        logging.info(f"{ws_log_prefix} Cancelling active generation task.")
                        active_generation_task.cancel()
                        # Send confirmation response
                        resp = {"jsonrpc": "2.0", "result": {"status": "cancelled"}, "id": request_id}
                        await websocket.send_text(json.dumps(resp))
                    else:
                        logging.warning(f"{ws_log_prefix} No active generation task to cancel.")
                        await send_jsonrpc_error(websocket, -32003, "No active generation to cancel", request_id)
                else:
                    logging.warning(f"{ws_log_prefix} Received unsupported method: {rpc_request.method}")
                    await send_jsonrpc_error(websocket, JsonRpcErrorCode.METHOD_NOT_FOUND, "Method not found", request_id)

            except json.JSONDecodeError:
                logging.error(f"{ws_log_prefix} Received non-JSON message: {message_text[:100]}...")
                await send_jsonrpc_error(websocket, JsonRpcErrorCode.PARSE_ERROR, "Parse error: Invalid JSON")
            except Exception as parse_e: 
                 logging.error(f"{ws_log_prefix} Invalid JSON-RPC message structure: {parse_e}")
                 # Attempt to send error if we could parse an ID, otherwise send notification
                 error_id = request_id if 'request_id' in locals() else None
                 await send_jsonrpc_error(websocket, -32600, f"Invalid Request: {parse_e}", error_id)

    except WebSocketDisconnect:
        logging.info(f"{ws_log_prefix} WebSocket disconnected by client.")
    except Exception as ws_e:
        logging.exception(f"{ws_log_prefix} Unhandled error in WebSocket handler: {ws_e}")
    finally:
        logging.info(f"{ws_log_prefix} Cleaning up WebSocket connection.")
        # Ensure any running task is cancelled
        if active_generation_task and not active_generation_task.done():
            logging.info(f"{ws_log_prefix} Cancelling generation task during cleanup.")
            active_generation_task.cancel()
        # Attempt to close websocket gracefully if not already closed
        try:
            await websocket.close(code=1000)
        except RuntimeError as e:
             if "Connection is closed" in str(e):
                 pass # Already closed, ignore
             else:
                 logging.warning(f"{ws_log_prefix} Error closing websocket during cleanup: {e}")
        except Exception as e:
             logging.warning(f"{ws_log_prefix} Unexpected error closing websocket during cleanup: {e}")

# --- Starlette App Definition ---
# Keep HTTP handlers for session creation and optional RPC
async def create_session_endpoint_starlette(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        params = CreateSessionParams(**(body or {}))
        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = {"tools": params.tools or []}
        logging.info(f"[Session {session_id}] Created via HTTP. Tools registered: {bool(params.tools)}")
        return JSONResponse(CreateSessionResponse(session_id=session_id).model_dump())
    except Exception as e:
         logging.error(f"[Create Session] Error: {e}")
         return JSONResponse({'detail': 'Invalid request body or internal error.'}, status_code=400)

async def json_rpc_endpoint_starlette(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        rpc_request = JsonRpcRequest(**body)
        if rpc_request.method == "tool_list":
            logging.info(f"[RPC] Received 'tool_list' request (ID: {rpc_request.id}). Returning empty list.")
            resp = JsonRpcResponse(result={"tools": []}, id=rpc_request.id)
            return JSONResponse(resp.model_dump(exclude_none=True))
        else:
            logging.warning(f"[RPC] Received unsupported method '{rpc_request.method}' (ID: {rpc_request.id}).")
            resp = JsonRpcResponse(error=JsonRpcError(code=-32601, message="Method not found"), id=rpc_request.id)
            return JSONResponse(resp.model_dump(exclude_none=True), status_code=200) # JSON-RPC errors usually return 200 OK
    except Exception as e:
         logging.error(f"[RPC] Invalid request: {e}")
         resp = JsonRpcResponse(error=JsonRpcError(code=-32600, message=f"Invalid Request: {e}"), id=None)
         return JSONResponse(resp.model_dump(exclude_none=True), status_code=400)

# Define routes
starlette_app = Starlette(
    debug=True,
    routes=[
        # WebSocket endpoint for main communication
        WebSocketRoute("/ws/{session_id:str}", endpoint=websocket_endpoint),
        # Keep HTTP endpoints for session creation and optional RPC compat
        Route("/create_session", endpoint=create_session_endpoint_starlette, methods=["POST"]),
        Route("/rpc", endpoint=json_rpc_endpoint_starlette, methods=["POST"]),
    ],
    on_startup=[shared.load_model],
    # on_shutdown=[shutdown_handler], # TODO: Add shutdown handler for SESSIONS cleanup if needed
)

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    import torch
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"GPU name: {torch.cuda.get_device_name(0)}")
    logging.info(f"Starting Uvicorn server with Starlette WebSocket app (model: {MODEL_ID} loaded via shared module)")
    uvicorn.run(starlette_app, host="0.0.0.0", port=8000, ws="auto") # Specify ws="auto" 