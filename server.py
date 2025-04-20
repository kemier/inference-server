import asyncio
import json
# Remove direct model loading imports if no longer needed directly in server
# from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer 
# import torch 
from fastapi import FastAPI, HTTPException
import uvicorn
import os
from typing import Any, Optional, Union, Dict, List
import re
# Remove sse-starlette import
# from sse_starlette.sse import EventSourceResponse 
from celery.result import AsyncResult 
from fastapi.responses import StreamingResponse # Ensure this is imported

# Import the task function itself
from tasks import run_llm_inference_task

import shared # Keep shared import if needed for other shared state (though pipeline isn't used here)
import logging
from shared import setup_logging # Keep logging setup
from utils import generate_conversation_prompt, parse_llm_response_for_tool_calls
from models import *
from shared import MODEL_ID # Keep MODEL_ID import for logging if needed

setup_logging()
logging.info("*** Inference Server Starting Up - Logging Configured ***")

app = FastAPI(title="LLM MCP-Style Celery Server", version="1.1.0")

# --- Model Configuration --- MOVED TO shared.py ---

# --- Global Variables --- REMOVED MODEL/TOKENIZER ---
# model = None
# tokenizer = None
# --- END REMOVED GLOBALS ---

# --- Models --- (MOVED)

# --- Helper Functions --- (MOVED)

# --- Model Loading Function --- REMOVED ---
# def load_model():
#     ...
# --- END REMOVED FUNCTION ---

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    """Server startup event. (Removed model loading)"""
    logging.info("Server startup event triggered...")
    # load_model() # REMOVED
    logging.info("Server is ready (model loading handled by workers).")

# --- Single Endpoint for JSON-RPC --- (Keep as is, uses Celery task)
@app.post("/rpc")
async def json_rpc_endpoint(request: JsonRpcRequest) -> JsonRpcResponse:
    # ... (existing implementation sending task to Celery)
    req_id = request.id or "NO_ID"

    # --- Log Incoming Request ---
    try:
        logging.info(f"--- Incoming RPC Request --- ID: {req_id}, Method: {request.method}")
        if request.params:
             params_for_log = request.params.copy() if isinstance(request.params, dict) else request.params
             if isinstance(params_for_log, dict) and 'history' in params_for_log and isinstance(params_for_log.get('history'), list):
                 params_for_log['history'] = f"[History with {len(params_for_log['history'])} messages]"
             logging.debug(f"Params (summary): {params_for_log}")
    except Exception as log_err:
        logging.error(f"Error logging incoming request: {log_err}")

    response: JsonRpcResponse
    MAX_ITERATIONS = 30

    # --- Basic Validation ---
    if request.jsonrpc != "2.0":
        response = JsonRpcResponse(
            error=JsonRpcError(code=JsonRpcErrorCode.INVALID_REQUEST, message="Invalid JSON-RPC version"),
            id=req_id
        )
        return response

    # --- Handle 'tool_list' Method ---
    if request.method == "tool_list":
        logging.info(f"Handling tool_list request (ID: {req_id}).")
        tool_list_result = ToolListResult(tools=[])
        response = JsonRpcResponse(result=tool_list_result.dict(), id=req_id)

    # --- Handle 'create_message' Method (MODIFIED FOR CELERY) ---
    elif request.method == "create_message":
        logging.debug(f"Handling create_message request (ID: {req_id}).")

        # Validate Params
        try:
            logging.debug(f"Validating parameters (ID: {req_id})...")
            if request.params is None or not isinstance(request.params, dict):
                 raise ValueError("Params must be a JSON object for 'create_message' method")
            logging.debug(f"Attempting Pydantic validation for CreateMessageParams (ID: {req_id})...")
            params = CreateMessageParams(**request.params)
            logging.debug(f"Parameters validated successfully (ID: {req_id}).")
        except Exception as e:
            logging.exception(f"ERROR during parameter validation (ID: {req_id}). Exception: {e}")
            return JsonRpcResponse(
                error=JsonRpcError(code=JsonRpcErrorCode.INVALID_PARAMS, message=f"Invalid parameters for create_message: {e}"),
                id=req_id
            )

        # --- Core Logic: Generate Prompt and Send Task to Celery ---
        try:
            current_history: List[HistoryMessage] = []
            if params.history:
                current_history = params.history
            # --- Ensure latest user message is considered --- 
            # This logic seems flawed if history is provided. 
            # If history is provided, the latest user message should already be IN the history.
            # If ONLY message is provided, history should be [User(message)]
            # Let's refine this based on typical chat flow:
            
            # Start with the provided history, if any
            current_history = params.history if params.history else []

            # If there's a new message in this request (typical case for user turn)
            # ADD it to the history before generating the prompt.
            # Note: The client should ideally send the FULL history including the new message.
            # This server-side append is a fallback if client only sends new message + old history.
            if params.message:
                 # Avoid duplicating if client already included it in history
                 if not current_history or current_history[-1].role != 'user' or current_history[-1].content != params.message:
                    logging.debug(f"Appending new user message from params.message to history (ID: {req_id})")
                    current_history.append(HistoryMessage(role="user", content=params.message))
                 # else: # DEBUG
                 #    logging.debug(f"New user message already seems present in history[-1], not appending. (ID: {req_id})") # DEBUG
            
            if not current_history:
                 logging.error(f"History is empty after processing params! (ID: {req_id})") # Log error if history is empty
                 raise ValueError("Cannot generate prompt with empty history")

            # --- ADDED DETAILED HISTORY LOGGING --- 
            try:
                history_log_str = json.dumps([msg.model_dump() for msg in current_history], indent=2, ensure_ascii=False)
                logging.debug(f"History being passed to generate_conversation_prompt (ID: {req_id}):\n{history_log_str}")
            except Exception as log_hist_err:
                logging.error(f"Error logging history content (ID: {req_id}): {log_hist_err}")
            # --- END LOGGING ---

            current_iteration = params.iteration_count
            MAX_ITERATIONS = 30

            logging.debug(f"Generating LLM prompt (ID: {req_id})...")
            llm_input_prompt = generate_conversation_prompt(
                history=current_history,
                tools=params.tools,
                iteration_count=current_iteration,
                max_iterations=MAX_ITERATIONS
            )
            logging.debug(f"--- Generated Prompt Start (ID: {req_id}) ---")
            logging.debug(llm_input_prompt)
            logging.debug(f"--- Generated Prompt End --- (Length: {len(llm_input_prompt)} chars, ID: {req_id})")
            
            logging.info(f"Sending LLM inference task to Celery (ID: {req_id})...")
            task_result: AsyncResult = run_llm_inference_task.delay(
                prompt=llm_input_prompt,
                max_new_tokens=params.max_new_tokens,
                do_sample=params.do_sample,
                tools_definitions=params.tools,
                temperature=params.temperature
            )
            celery_task_id = task_result.id
            logging.info(f"Task sent to Celery with Task ID: {celery_task_id} (Request ID: {req_id})")

            response = JsonRpcResponse(
                result={"taskId": celery_task_id},
                id=req_id
            )
            logging.debug(f"Preparing Task ID response (ID: {req_id}).")
        
        except Exception as e:
             logging.exception(f"ERROR processing create_message for Celery task (ID: {req_id}): {e}")
             response = JsonRpcResponse(
                 error=JsonRpcError(code=JsonRpcErrorCode.INTERNAL_ERROR, message=f"Error processing create_message: {e}"),
                 id=req_id
             )

    # --- Handle Unknown Method ---
    else:
        logging.warning(f"ERROR: Method not found: {request.method} (ID: {req_id})")
        response = JsonRpcResponse(
            error=JsonRpcError(code=JsonRpcErrorCode.METHOD_NOT_FOUND, message=f"Method '{request.method}' not found"),
            id=req_id
        )

    # --- Log Outgoing Response ---
    try:
        outgoing_log = response.model_dump(exclude_none=True)
        logging.info(f"--- Outgoing RPC Response --- ID: {req_id}")
        logging.debug(json.dumps(outgoing_log, indent=2, ensure_ascii=False))
        logging.debug(f"--------------------------- ID: {req_id}")
    except Exception as log_err:
         logging.error(f"Error logging outgoing response (ID: {req_id}): {log_err}")

    return response

# --- Streaming Logic (MODIFIED FOR MANUAL SSE FORMATTING) ---
async def poll_task_and_stream_results_manual_sse(task_id: str):
    """Polls a Celery task and streams status/results via SSE (manual formatting)."""
    logging.info(f"[SSE:{task_id}] *** Manual SSE Stream started. Polling Celery task... ***")
    
    async_result: AsyncResult = AsyncResult(task_id)

    polling_interval = 1
    max_wait_cycles = 600
    cycles = 0
    last_reported_status = None

    try:
        while cycles < max_wait_cycles:
            status = async_result.state
            
            if status != last_reported_status:
                logging.debug(f"[SSE:{task_id}] Polling, status: {status}")
                # Manual SSE format: data: <json_string>\n\n
                yield f"data: {json.dumps({'type': 'status', 'taskId': task_id, 'status': status})}\n\n"
                last_reported_status = status

            if status == 'SUCCESS':
                logging.info(f"[SSE:{task_id}] Task finished successfully. Fetching result...")
                task_output = async_result.result
                logging.debug(f"[SSE:{task_id}] Result fetched from backend.")
                
                if not isinstance(task_output, dict):
                     logging.error(f"[SSE:{task_id}] ERROR: Worker returned unexpected result type: {type(task_output)}")
                     # Manual SSE format: event: <event_name>\ndata: <json_string>\n\n
                     yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': 'Worker Format Error', 'details': 'Worker returned invalid result format'})}\n\n"
                     break

                cleaned_response_text = task_output.get("cleaned_response_text", "")
                tools_definitions = task_output.get("tools_definitions")
                
                logging.debug(f"[SSE:{task_id}] Parsing worker result...")
                tool_calls = parse_llm_response_for_tool_calls(cleaned_response_text, tools_definitions)

                final_result = None
                if tool_calls:
                    logging.info(f"[SSE:{task_id}] Parsed tool calls successfully.")
                    final_result = {"type": "tool_calls", "content": tool_calls}
                    logging.debug(f"[SSE:{task_id}] Yielding final tool_calls data.")
                    yield f"data: {json.dumps(final_result)}\n\n"
                else:
                    logging.info(f"[SSE:{task_id}] No tool calls found in result.")
                    final_text_content = cleaned_response_text 
                    final_result = {"type": "final_text", "content": final_text_content}
                    logging.debug(f"[SSE:{task_id}] Yielding final_text data.")
                    yield f"data: {json.dumps(final_result)}\n\n"
                break

            elif status == 'FAILURE':
                logging.error(f"[SSE:{task_id}] ERROR: Task failed in worker.")
                try:
                    error_details = async_result.traceback or str(async_result.result)
                except Exception as tb_exc:
                    error_details = f"Could not retrieve traceback: {tb_exc}"
                logging.debug(f"[SSE:{task_id}] Failure details: {error_details}")
                yield f"event: error\ndata: {json.dumps({'type': 'error', 'taskId': task_id, 'error': 'Task failed in worker', 'details': str(error_details)})}\n\n"
                break
            
            elif status in ['PENDING', 'RECEIVED', 'STARTED', 'RETRY']:
                pass
            else:
                 logging.warning(f"[SSE:{task_id}] Unknown Celery task state encountered: {status}")
                 pass

            await asyncio.sleep(polling_interval)
            cycles += 1

        if cycles >= max_wait_cycles and status not in ['SUCCESS', 'FAILURE']:
            logging.error(f"[SSE:{task_id}] ERROR: Timeout waiting for Celery task to complete.")
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'taskId': task_id, 'error': 'Timeout', 'details': f'Timeout waiting for task completion. Last state: {status}'})}\n\n"
            
    except Exception as e:
        logging.exception(f"[SSE:{task_id}] ERROR: Exception during Celery polling loop: {e}")
        yield f"event: error\ndata: {json.dumps({'type': 'error', 'taskId': task_id, 'error': 'Streaming Error', 'details': f'Error during result streaming: {e}'})}\n\n"
    finally:
        logging.info(f"[SSE:{task_id}] *** Ending Manual SSE stream. ***")
        yield f"event: end\ndata: {json.dumps({'type': 'end', 'taskId': task_id, 'content': 'Stream ended'})}\n\n"
# --- END Polling Generator ---

# --- Streaming Endpoint (Using StreamingResponse) ---
@app.get("/stream/{task_id}")
async def stream_endpoint(task_id: str):
    """Handles SSE requests using FastAPI StreamingResponse."""
    return StreamingResponse(
        poll_task_and_stream_results_manual_sse(task_id), 
        media_type="text/event-stream"
    )

# --- Main Execution --- (Keep as is, ensure torch isn't imported if removed above)
if __name__ == "__main__":
    # logging.info(f"PyTorch version: {torch.__version__}") # Removed if torch not imported
    # logging.info(f"CUDA available: {torch.cuda.is_available()}") # Removed if torch not imported
    # if torch.cuda.is_available():
    #     logging.info(f"CUDA version: {torch.version.cuda}")
    #     logging.info(f"GPU name: {torch.cuda.get_device_name(0)}")
    logging.info(f"Starting Uvicorn server (model: {MODEL_ID} loaded by workers)") # Updated log
    uvicorn.run(app, host="0.0.0.0", port=8000)