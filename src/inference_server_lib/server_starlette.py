import asyncio
import json
import uuid
import logging
import os
import time
from typing import Any, Optional, Union, Dict, List, Callable
from pydantic import ValidationError

# Starlette imports
from starlette.applications import Starlette
from starlette.routing import Route, WebSocketRoute
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.websockets import WebSocket, WebSocketDisconnect

# Keep our existing modules
from . import shared
from .shared import setup_logging
from .utils import generate_interactive_stream_ws
from .models import *

setup_logging()
logging.info("*** Starlette WebSocket Server Starting Up - Logging Configured ***")

# --- Global SESSIONS Store ---
SESSIONS: Dict[str, Dict[str, Any]] = {}

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
    active_generation_task : Optional[asyncio.Task] = None # Track task for cancellation - this might still be session-level for simplicity of a single active generator
    current_history: List[HistoryMessage] = [] # Maintain history within the connection - shared for now
    current_tools: Optional[List[Dict[str, Any]]] = None # Store tools for the session
    
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

                        # Send acknowledgment response for the request
                        ack_resp = {"jsonrpc": "2.0", "result": {"status": "processing"}, "id": request_id}
                        await websocket.send_text(json.dumps(ack_resp))

                        # --- ADDED: Create and store task_state --- 
                        current_task_id = str(uuid.uuid4()) # Generate a unique ID for this generation task/flow
                        task_state_data = {
                            "original_generation_params": params.model_dump(), 
                            "current_generation_task": None, 
                            "iteration_count": 0, # Initial iteration count for this task flow
                            "pending_ai_message": None # Initialize pending_ai_message for this task
                        }
                        
                        # ---- ADDED DIAGNOSTIC LOGS ----
                        logging.debug(f"{ws_log_prefix} About to assign to SESSIONS[session_id]['tasks'][current_task_id]. session_id: {session_id}, current_task_id: {current_task_id}")
                        if session_id in SESSIONS:
                            logging.debug(f"{ws_log_prefix} SESSIONS[session_id] exists. Keys: {list(SESSIONS[session_id].keys())}")
                            if "tasks" in SESSIONS[session_id] and isinstance(SESSIONS[session_id].get("tasks"), dict):
                                logging.debug(f"{ws_log_prefix} SESSIONS[session_id]['tasks'] exists and is a dict. Current tasks keys: {list(SESSIONS[session_id]['tasks'].keys())}")
                            elif "tasks" in SESSIONS[session_id]:
                                logging.error(f"{ws_log_prefix} CRITICAL: SESSIONS[session_id]['tasks'] exists BUT IS NOT A DICT! Type: {type(SESSIONS[session_id]['tasks'])}")
                            else:
                                logging.error(f"{ws_log_prefix} CRITICAL: SESSIONS[session_id] exists BUT 'tasks' key is MISSING before assignment!")
                        # --- END ADDED DIAGNOSTIC LOGS ---

                        SESSIONS[session_id]["tasks"][current_task_id] = task_state_data
                        logging.debug(f"{ws_log_prefix} Initialized task state for task_id: {current_task_id}")
                        # --- END ADDED ---

                        # Start generation task
                        active_generation_task = asyncio.create_task(
                            generate_interactive_stream_ws(
                                websocket=websocket,
                                session_id=session_id,
                                task_id=current_task_id,
                                history=current_history,
                                tools=current_tools,
                                generation_params=params
                            )
                        )
                        task_state_data["current_generation_task"] = active_generation_task

                        # Add callback to handle completion/cancellation
                        # Pass current_task_id to the callback
                        def _generation_done_callback(task_id_of_completed_task: str, result_dict: Dict[str, Any]):
                            nonlocal active_generation_task, current_history 
                            
                            session_obj = SESSIONS.get(session_id)
                            if not session_obj or "tasks" not in session_obj or task_id_of_completed_task not in session_obj["tasks"]:
                                logging.error(f"{ws_log_prefix} Task state not found for completed task {task_id_of_completed_task} in callback. Cannot store pending AI message or update iteration.")
                                if active_generation_task is result_dict.get("current_generation_task"): # Check if this is the session's active task
                                    active_generation_task = None
                                return

                            task_state_for_completed = session_obj["tasks"][task_id_of_completed_task]
                            
                            if active_generation_task is result_dict.get("current_generation_task"): # If this callback is for the session's tracked active task
                                active_generation_task = None
                            
                            try:
                                status = result_dict.get("status")
                                final_ai_message_chunk = result_dict.get("final_ai_message_chunk") # This is AIMessageChunk
                                aggregated_openai_tool_calls_from_result = result_dict.get("aggregated_openai_tool_calls") # List[OpenAIAssistantToolCall]
                                
                                logging.debug(f"{ws_log_prefix} _generation_done_callback for task {task_id_of_completed_task}. Status: {status}. final_ai_message_chunk is None: {final_ai_message_chunk is None}. aggregated_openai_tool_calls is None: {aggregated_openai_tool_calls_from_result is None}")
                                if final_ai_message_chunk:
                                    logging.debug(f"{ws_log_prefix} final_ai_message_chunk content: '{final_ai_message_chunk.content}', tool_calls from chunk: {final_ai_message_chunk.tool_calls}")
                                if aggregated_openai_tool_calls_from_result:
                                    logging.debug(f"{ws_log_prefix} aggregated_openai_tool_calls_from_result: {aggregated_openai_tool_calls_from_result}")

                                if status == "function_call_requested" and aggregated_openai_tool_calls_from_result:
                                    assistant_message_with_tool_calls = HistoryMessage(
                                        role="assistant",
                                        content=final_ai_message_chunk.content if final_ai_message_chunk and final_ai_message_chunk.content else None,
                                        tool_calls=aggregated_openai_tool_calls_from_result
                                    )
                                    current_history.append(assistant_message_with_tool_calls)
                                    task_state_for_completed["pending_ai_message"] = assistant_message_with_tool_calls
                                    task_state_for_completed["iteration_count"] = task_state_for_completed.get("iteration_count", 0) + 1
                                    logging.info(f"{ws_log_prefix} Stored pending AI tool call message (using aggregated_openai_tool_calls) for task {task_id_of_completed_task}. Iteration: {task_state_for_completed['iteration_count']}. History len: {len(current_history)}")
                                
                                elif final_ai_message_chunk and final_ai_message_chunk.tool_call_chunks and not aggregated_openai_tool_calls_from_result: 
                                    logging.warning(f"{ws_log_prefix} _generation_done_callback for task {task_id_of_completed_task}: Fallback to tool_call_chunks. This is unexpected.")
                                    processed_tool_calls_from_chunks = []
                                    for tc_chunk in final_ai_message_chunk.tool_call_chunks:
                                        if tc_chunk.get('name') and tc_chunk.get('id') and tc_chunk.get('args') is not None:
                                            processed_tool_calls_from_chunks.append(
                                                OpenAIAssistantToolCall(
                                                    id=tc_chunk['id'], type='function',
                                                    function=OpenAIFunctionSpec(name=tc_chunk['name'], arguments=tc_chunk['args'])
                                                )
                                            )
                                        else:
                                            logging.warning(f"{ws_log_prefix} Malformed tool_call_chunk in callback fallback: {tc_chunk}. Skipping.")
                                    if processed_tool_calls_from_chunks:
                                        pending_message = HistoryMessage(
                                            role="assistant",
                                            content=final_ai_message_chunk.content if final_ai_message_chunk.content else None,
                                            tool_calls=processed_tool_calls_from_chunks
                                        )
                                        current_history.append(pending_message)
                                        task_state_for_completed["pending_ai_message"] = pending_message
                                        task_state_for_completed["iteration_count"] = task_state_for_completed.get("iteration_count", 0) + 1
                                        logging.debug(f"{ws_log_prefix} Stored pending AI (fallback from chunks) for task {task_id_of_completed_task}.")
                                    else:
                                        task_state_for_completed["pending_ai_message"] = None
                                
                                elif final_ai_message_chunk and final_ai_message_chunk.content:
                                    final_text_message = HistoryMessage(role="assistant", content=final_ai_message_chunk.content)
                                    current_history.append(final_text_message)
                                    logging.info(f"{ws_log_prefix} Appended final text response to history for task {task_id_of_completed_task}. History len: {len(current_history)}")
                                    task_state_for_completed["pending_ai_message"] = None
                                
                                elif status == "function_call_requested" and not aggregated_openai_tool_calls_from_result:
                                     logging.error(f"{ws_log_prefix} _generation_done_callback: Status indicates tool calls, but aggregated_openai_tool_calls_from_result is missing/empty for task {task_id_of_completed_task}.")
                                     task_state_for_completed["pending_ai_message"] = None
                                else:
                                    logging.debug(f"{ws_log_prefix} Generation for task {task_id_of_completed_task} finished (callback); no tool calls and no text content in final_ai_message_chunk, or chunk was None and status didn't indicate tool calls.")
                                    task_state_for_completed["pending_ai_message"] = None

                            except AttributeError as ae:
                                logging.error(f"{ws_log_prefix} AttributeError in _generation_done_callback for task {task_id_of_completed_task}: {ae}. final_ai_message_chunk might be None or malformed.", exc_info=True)
                                task_state_for_completed["pending_ai_message"] = None
                            except Exception as e: 
                                logging.error(f"{ws_log_prefix} Exception in _generation_done_callback processing for task {task_id_of_completed_task}: {type(e).__name__} - {e}", exc_info=True)
                                task_state_for_completed["pending_ai_message"] = None # Ensure state is cleared on error

                        # --- MODIFIED: Corrected callback registration --- 
                        def task_done_wrapper(completed_task_future: asyncio.Future, task_id_for_callback: str):
                            # nonlocal ws_log_prefix # ws_log_prefix is available in the outer scope already
                            try:
                                if completed_task_future.cancelled():
                                    logging.info(f"{ws_log_prefix} Task {task_id_for_callback} was cancelled. Calling callback with cancelled status.")
                                    _generation_done_callback(task_id_of_completed_task=task_id_for_callback, result_dict={"status": "cancelled", "final_ai_message_chunk": None})
                                else:
                                    result = completed_task_future.result()
                                    logging.debug(f"{ws_log_prefix} Task {task_id_for_callback} completed. Result for callback: {result}")
                                    _generation_done_callback(task_id_of_completed_task=task_id_for_callback, result_dict=result)
                            except Exception as e:
                                logging.error(f"{ws_log_prefix} Exception in task_done_wrapper for task {task_id_for_callback}: {e}", exc_info=True)
                                _generation_done_callback(task_id_of_completed_task=task_id_for_callback, result_dict={"status": "error", "error_message": str(e), "final_ai_message_chunk": None})

                        active_generation_task.add_done_callback(
                            # --- RESTORED: Lambda now calls task_done_wrapper --- 
                            lambda completed_task_future: task_done_wrapper(completed_task_future, current_task_id)
                        )
                        # --- END MODIFICATION ---

                    except Exception as param_e: # Catch Pydantic validation errors etc.
                         logging.error(f"{ws_log_prefix} Invalid params for 'generate': {param_e}")
                         await send_jsonrpc_error(websocket, JsonRpcErrorCode.INVALID_PARAMS, f"Invalid params: {param_e}", request_id)
                
                elif rpc_request.method == "tool_result":
                    logging.info(f"{ws_log_prefix} Received 'tool_result' request (ID: {request_id}).")
                    
                    if active_generation_task and not active_generation_task.done():
                         logging.warning(f"{ws_log_prefix} Received 'tool_result' while generation is running. Ignoring.")
                         await send_jsonrpc_error(websocket, JsonRpcErrorCode.INTERNAL_ERROR, "Cannot process tool result while generation is active", request_id)
                         continue

                    # --- MODIFICATION START: Validate all pending tool calls are addressed ---
                    # Retrieve task_id from params, then use it to get task_state
                    try:
                        params = ToolResultParams(**(rpc_request.params or {}))
                        original_task_id = params.task_id # Task ID of the original generate request

                        task_state_for_tool_result = SESSIONS.get(session_id, {}).get("tasks", {}).get(original_task_id)

                        if not task_state_for_tool_result:
                            logging.warning(f"{ws_log_prefix} Received 'tool_result' for unknown or already cleaned up task_id {original_task_id}. Ignoring.")
                            ack_resp = {"jsonrpc": "2.0", "result": {"status": "received_ignored_unknown_task"}, "id": request_id}
                            await websocket.send_text(json.dumps(ack_resp))
                            continue

                        current_task_pending_ai_message = task_state_for_tool_result.get("pending_ai_message")
                        current_task_iteration_count = task_state_for_tool_result.get("iteration_count", 0)


                        if not current_task_pending_ai_message:
                            logging.warning(f"{ws_log_prefix} Received 'tool_result' for task {original_task_id}, but no AI message was expecting tool calls in its state. Iteration: {current_task_iteration_count}. Ignoring stale result.")
                            ack_resp = {"jsonrpc": "2.0", "result": {"status": "received_ignored_no_pending_call_in_task_state"}, "id": request_id}
                            await websocket.send_text(json.dumps(ack_resp))
                            continue
                        
                        if not params.results:
                            logging.warning(f"{ws_log_prefix} Received 'tool_result' with empty results list for task {original_task_id}.")
                            ack_resp = {"jsonrpc": "2.0", "result": {"status": "received_empty_results_unexpectedly"}, "id": request_id}
                            await websocket.send_text(json.dumps(ack_resp))
                            task_state_for_tool_result["pending_ai_message"] = None
                            continue
                        
                        required_tool_call_ids = set()
                        # --- MODIFICATION START: Correctly access attributes of OpenAIAssistantToolCall objects ---
                        for tool_call_obj in current_task_pending_ai_message.tool_calls: # Use current_task_pending_ai_message
                            if isinstance(tool_call_obj, OpenAIAssistantToolCall) and tool_call_obj.id:
                                required_tool_call_ids.add(tool_call_obj.id)
                            else:
                                # This path should ideally not be hit if current_task_pending_ai_message.tool_calls contains valid OpenAIAssistantToolCall objects
                                logging.error(f"{ws_log_prefix} Malformed tool_call entry in pending_ai_message for task {original_task_id}. Expected OpenAIAssistantToolCall, got: {type(tool_call_obj)}. Entry: {tool_call_obj}.")
                                await send_jsonrpc_error(websocket, JsonRpcErrorCode.INTERNAL_ERROR, "Internal error: Malformed tool call data structure in pending AI message for task.", request_id)
                                task_state_for_tool_result["pending_ai_message"] = None
                                raise ValueError("Malformed pending tool call data structure.")
                        # --- MODIFICATION END --

                        if not required_tool_call_ids:
                             logging.warning(f"{ws_log_prefix} AI message was pending for tool calls for task {original_task_id}, but its tool_calls list resolved to zero required IDs. This is unusual.")
                             ack_resp = {"jsonrpc": "2.0", "result": {"status": "received_no_tools_were_pending_in_task"}, "id": request_id}
                             await websocket.send_text(json.dumps(ack_resp))
                             task_state_for_tool_result["pending_ai_message"] = None
                             continue

                        received_tool_call_ids = {res.tool_call_id for res in params.results if res.tool_call_id}

                        if required_tool_call_ids != received_tool_call_ids:
                            missing_ids = list(required_tool_call_ids - received_tool_call_ids)
                            extra_ids = list(received_tool_call_ids - required_tool_call_ids)
                            logging.error(f"{ws_log_prefix} Mismatch in tool_call_ids for task {original_task_id}. Required: {required_tool_call_ids}, Received: {received_tool_call_ids}. Missing: {missing_ids}, Extra: {extra_ids}. Aborting continuation.")
                            error_detail = {
                                "required_ids": list(required_tool_call_ids),
                                "received_ids": list(received_tool_call_ids),
                                "missing_ids": missing_ids,
                                "extra_ids": extra_ids
                            }
                            await send_jsonrpc_error(websocket, JsonRpcErrorCode.INVALID_PARAMS, "Mismatch between required and received tool_call_ids. All required tool results must be provided in a single tool_result call.", request_id, data=error_detail)
                            # Client needs to send a `tool_result` message with all required IDs.
                            # We keep pending_ai_tool_call_message so they can retry.
                            continue 
                        
                        # If we reach here, all tool calls are accounted for.
                        # The pending AI message (that requested the tools) IS ALREADY IN current_history
                        # as it was added by _generation_done_callback. We just need to clear it from task_state.
                        task_state_for_tool_result["pending_ai_message"] = None # Clear it from task_state as it's now processed
                        logging.debug(f"{ws_log_prefix} Cleared pending AI tool call message from task state for task {original_task_id}. History length should reflect prior append by callback: {len(current_history)}.")

                        # Add the tool results to history
                        processed_any_tool_results = False # This check is somewhat redundant now due to above validation but keep for sanity
                        original_history_len = len(current_history) 

                        # Create a map of received results for easier lookup if needed, though iteration order is fine
                        results_map = {res.tool_call_id: res for res in params.results}

                        # Add tool messages in the order of required_tool_call_ids to maintain consistency if it ever matters,
                        # or simply iterate params.results as before if order is guaranteed by client or doesn't matter.
                        # For now, iterate params.results which should align with the validated received_tool_call_ids.
                        for tool_input_item in params.results: # Iterating what client sent, order might matter to client
                            tool_name = tool_input_item.tool_name
                            tool_call_id_from_client = tool_input_item.tool_call_id # Already validated to be in required_ids
                            
                            # Basic validation of tool_call_id format was implicitly done by set comparison.
                            # Explicit check was removed as part of streamlining.
                            # Ensure tool_call_id is not None or empty string if it passed set inclusion (it shouldn't be).

                            result_data = tool_input_item.result
                            error_data = tool_input_item.error

                            tool_message = HistoryMessage(
                                role="tool",
                                tool_name=tool_name, 
                                tool_result=result_data,
                                tool_error=error_data,
                                tool_call_id=tool_call_id_from_client
                            )
                            logging.debug(f"{ws_log_prefix} Created tool HistoryMessage with tool_call_id: '{tool_message.tool_call_id}' for tool_name: '{tool_message.tool_name}'. Task ID: {original_task_id}")
                            current_history.append(tool_message)
                            logging.info(f"{ws_log_prefix} Appended tool result for '{tool_name}' (ID: {tool_call_id_from_client}) to history. New history length: {len(current_history)}. Task ID: {original_task_id}")
                            processed_any_tool_results = True
                        
                        # This check is now less critical due to the strict ID matching above, 
                        # but can catch edge cases if params.results was empty but somehow passed earlier checks.
                        if not processed_any_tool_results and required_tool_call_ids: 
                             logging.warning(f"{ws_log_prefix} No tool results were processed despite matching ID sets. This is unexpected. Task ID: {original_task_id}")
                             # This state should ideally not be reached if required_tool_call_ids was non-empty.
                             ack_resp = {"jsonrpc": "2.0", "result": {"status": "received_empty_results_unexpectedly"}, "id": request_id}
                             await websocket.send_text(json.dumps(ack_resp))
                             continue # Avoid proceeding if something is very wrong

                        ack_resp = {"jsonrpc": "2.0", "result": {"status": "received_and_processed"}, "id": request_id}
                        await websocket.send_text(json.dumps(ack_resp))

                        # --- MODIFIED: Correctly retrieve task_state --- 
                        # task_state = SESSIONS[session_id]["tasks"].get(original_task_id) # Already have as task_state_for_tool_result
                        # --- END MODIFICATION ---
                        if not task_state_for_tool_result:
                            logging.error(f"{ws_log_prefix} Task state not found for task_id {original_task_id} after tool result (unexpected). Cannot continue.")
                            await send_jsonrpc_error(websocket, JsonRpcErrorCode.INTERNAL_ERROR, "Internal server error: Task state lost.", request_id)
                            continue
                            
                        # Prepare for continuation call
                        original_gen_params_dict = task_state_for_tool_result["original_generation_params"]
                        continuation_dict_for_params = original_gen_params_dict.copy()

                        # We need to pass the history to GenerateRequestParams for validation if its definition requires it,
                        # but we pass the actual objects (current_history) to the generation function.
                        # Let's just add the current history (as objects) to the dict for validation purposes.
                        continuation_dict_for_params["history"] = current_history 

                        continuation_dict_for_params["tools"] = session_data.get("tools") # Ensure tools are passed for continuation
                        # Use iteration count from task_state
                        new_iteration_count = task_state_for_tool_result.get("iteration_count", 0) # Already incremented in callback, or use as is
                        continuation_dict_for_params["iteration_count"] = new_iteration_count 

                        try:
                            generation_params_for_continuation = GenerateRequestParams(**continuation_dict_for_params)
                        except Exception as e:
                            logging.error(f"{ws_log_prefix} Pydantic validation error for GenerateRequestParams on continuation: {e}")
                            await send_jsonrpc_error(websocket, JsonRpcErrorCode.INTERNAL_ERROR, f"Internal error preparing for generation continuation: {e}", request_id)
                            SESSIONS[session_id]["tasks"].pop(original_task_id, None) # Clean up task state
                            continue

                            # DEBUG LOG 2 (Corrected to logging.debug)
                            if current_history:
                               last_hist_msg = current_history[-1] # This will be a HistoryMessage object
                               logging.debug(f"{ws_log_prefix} Last history message (object) just before task creation: role='{last_hist_msg.role}', content_preview='{str(last_hist_msg.content)[:50]}...', tool_call_id='{getattr(last_hist_msg, 'tool_call_id', 'N/A')}', tool_name='{getattr(last_hist_msg, 'tool_name', 'N/A')}'. Task ID: {original_task_id}")
                            else:
                                logging.debug(f"{ws_log_prefix} current_history is empty before continuation. Task ID: {original_task_id}") 
                            
                            # DEBUG LOG 3 (Keep) -> This log seems to reference 'history_as_dicts' which was removed. Let's update or remove.
                            # For now, removing this specific log block as history_as_dicts is no longer used here.
                            # logging.debug(f"{ws_log_prefix} Dumping current_history (list of dicts via history_as_dicts) before task creation:")
                            # for idx, hist_item_dict in enumerate(history_as_dicts):
                            #    # hist_item_dict is now a dict, so access keys directly
                            #    role = hist_item_dict.get('role', 'N/A')
                            #    tc_id = hist_item_dict.get('tool_call_id', 'N/A')
                            #    tn = hist_item_dict.get('tool_name', 'N/A')
                            #    logging.debug(f"{ws_log_prefix}   idx {idx}: role='{role}', tool_call_id='{tc_id}', tool_name='{tn}'")


                        # Schedule the next generation step
                        active_generation_task = asyncio.create_task(
                            generate_interactive_stream_ws(
                                websocket=websocket,
                                session_id=session_id,
                                task_id=original_task_id, 
                                history=current_history, 
                                tools=SESSIONS[session_id].get("tools"),
                                generation_params=generation_params_for_continuation
                            )
                        )
                        task_state_for_tool_result["current_generation_task"] = active_generation_task
                        
                        # --- MODIFIED: Corrected callback registration for continuation task --- 
                        active_generation_task.add_done_callback(
                            # --- RESTORED: Lambda now calls task_done_wrapper for continuation task --- 
                            lambda completed_task_future: task_done_wrapper(completed_task_future, original_task_id)
                        )
                        # --- END MODIFICATION --- 

                        logging.info(f"{ws_log_prefix} Scheduled continuation generation task (ID: {original_task_id}), task iteration: {new_iteration_count}")

                    except ValidationError as ve_tool_result:
                        logging.error(f"{ws_log_prefix} Invalid params for 'tool_result': {ve_tool_result}")
                        await send_jsonrpc_error(websocket, JsonRpcErrorCode.INVALID_PARAMS, f"Invalid tool_result params: {ve_tool_result}", request_id)
                    except ValueError as ve_internal: # Catch our specific ValueError for malformed pending data
                        logging.error(f"{ws_log_prefix} Internal error processing tool_result precondition: {ve_internal}. Task ID: {original_task_id if 'original_task_id' in locals() else 'Unknown'}")
                        # Error already sent if it was due to malformed pending_ai_tool_call_message.
                        # If not, send a generic internal error.
                        if "Malformed pending tool call data" not in str(ve_internal): # Avoid double sending if error already sent
                             await send_jsonrpc_error(websocket, JsonRpcErrorCode.INTERNAL_ERROR, f"Internal server error: {ve_internal}", request_id)
                    except Exception as tool_e:
                        logging.exception(f"{ws_log_prefix} Error processing 'tool_result': {tool_e}. Task ID: {original_task_id if 'original_task_id' in locals() else 'Unknown'}")
                        await send_jsonrpc_error(websocket, JsonRpcErrorCode.INTERNAL_ERROR, f"Internal server error processing tool_result: {tool_e}", request_id)

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
        # --- MODIFIED: Initialize "tasks" sub-dictionary --- 
        SESSIONS[session_id] = {
            "tools": params.tools or [], 
            "created_at": time.time(),
            "tasks": {} # Initialize tasks dictionary for the session
        }
        # --- END MODIFICATION ---
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
    on_startup=[shared.initialize_llm],
    # on_shutdown=[shutdown_handler], # TODO: Add shutdown handler for SESSIONS cleanup if needed
)

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    # import torch # Removed torch import as it's no longer directly used here for version printing
    # logging.info(f"PyTorch version: {torch.__version__}") # Removed
    # logging.info(f"CUDA available: {torch.cuda.is_available()}") # Removed
    # if torch.cuda.is_available(): # Removed
    #     logging.info(f"CUDA version: {torch.version.cuda}") # Removed
    #     logging.info(f"GPU name: {torch.cuda.get_device_name(0)}") # Removed
    # logging.info(f"Starting Uvicorn server with Starlette WebSocket app (model: {MODEL_ID} loaded via shared module)") # MODEL_ID is removed
    logging.info(f"Starting Uvicorn server with Starlette WebSocket app (using ChatOpenAI via shared module)") # Updated log message
    uvicorn.run(starlette_app, host="0.0.0.0", port=8000, ws="auto") # Specify ws="auto" 