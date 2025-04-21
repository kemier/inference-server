import asyncio
import json
import re
import logging
from typing import Any, Optional, Union, Dict, List, AsyncGenerator
import time
import threading
from transformers import TextIteratorStreamer
import shared
from models import HistoryMessage, ToolCallRequest, GenerateRequestParams, TextChunkParams, FunctionCallRequestParams, ErrorNotificationParams, StreamEndParams, BaseNotificationParams, JsonRpcError
from starlette.websockets import WebSocket
import traceback

# --- Logging configuration (moved near top) ---
logging.basicConfig(
    level=logging.DEBUG, # Adjust level as needed (DEBUG, INFO, WARNING, ERROR)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference_server.log"),
        logging.StreamHandler() # Log to console as well
    ]
)
logger = logging.getLogger(__name__) # Use a specific logger for utils

# --- JSON-RPC Helper to create notification strings ---
def create_jsonrpc_notification(method: str, params: Optional[Dict[str, Any]] = None) -> str:
    """Creates a JSON-RPC 2.0 notification string."""
    notification = {
        "jsonrpc": "2.0",
        "method": method,
    }
    if params is not None:
        notification["params"] = params
    try:
        return json.dumps(notification)
    except TypeError as e:
        logger.error(f"Failed to serialize JSON-RPC notification. Method: {method}, Params: {params}, Error: {e}")
        # Fallback for unserializable data - send an error notification instead
        fallback_params = {'type': 'error', 'error': 'Serialization Error', 'details': f'Could not serialize parameters for method {method}: {e}'}
        return json.dumps({"jsonrpc": "2.0", "method": "error", "params": fallback_params})

# --- Add FUNC_CALL_REGEX --- 
# Regex to find function calls like <function_call name="get_current_time">{format: "%H:%M"}</function_call>
# It captures the function name and the JSON arguments string.
FUNC_CALL_REGEX = re.compile(r"<function_call\s*name=['\"]([a-zA-Z_][a-zA-Z0-9_]*)['\"]\s*args=['\"](.*?)['\"]\s*/>", re.DOTALL)
# --- End Add --- 

def generate_conversation_prompt(
    history: List[HistoryMessage],
    tools: Optional[List[Dict[str, Any]]] = None,
    iteration_count: int = 0,
    max_iterations: int = 5, # Default max iterations
    session_id: Optional[str] = None, # Added
    task_id: Optional[str] = None      # Added
) -> str:
    # Create log prefix if IDs are available
    log_prefix = ""
    if session_id and task_id:
        log_prefix = f"[WS Session {session_id} / Task {task_id}] "
    elif session_id:
        log_prefix = f"[WS Session {session_id}] "

    logger.debug(f"{log_prefix}Generating prompt for iteration {iteration_count}/{max_iterations}") # Use prefix
    prompt_parts = []

    # Add history formatting (keep existing logic)
    prompt_parts.append("CONVERSATION HISTORY:")
    latest_user_question_index = -1
    last_message_role = history[-1].role if history else None # Check last message role
    for i, msg in reversed(list(enumerate(history))):
        if msg.role == 'user':
            latest_user_question_index = i
            break

    last_added_normalized_content = None # Track normalized content
    for idx, msg in enumerate(history):
        turn = ""
        role = msg.role
        content = msg.content if msg.content else "" # Ensure content is a string

        # Normalized content for deduplication check (strip whitespace and </s>)
        normalized_content = content.strip().removesuffix('</s>').strip()

        # Build the turn string based on role (only if content is not empty OR it's a tool call)
        if role == 'user':
            if idx == latest_user_question_index:
                turn += "\nLATEST USER QUESTION: USER: "
            else:
                turn += "\nUSER: "
            turn += content # Use original content here
        elif role == 'assistant':
            turn += "\nASSISTANT:"
            if content: # Use original content
                turn += f" {content}"
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                     try:
                         # Represent tool calls requested by assistant in history
                         # Keep XML format for history representation consistency
                         args_json_str = json.dumps(tool_call.parameters)
                         turn += f" <function_call name='{tool_call.tool}' args='{args_json_str}'/>"
                     except Exception as dump_err:
                         logging.error(f"{log_prefix}ERROR serializing tool_call in history: {dump_err}") # Use prefix
                         turn += " [Error displaying tool call]"
        elif role == 'tool':
            # Tool results aren't deduplicated based on simple content match
            normalized_content = f"TOOL_RESULT:{msg.tool_name}:{idx}" # Ensure uniqueness
            turn += f"\nTOOL_RESULT (for {msg.tool_name}):"
            if msg.tool_error:
                turn += f" Error: {msg.tool_error}"
            else:
                try:
                    # Keep result representation concise in history
                    result_str = json.dumps(msg.tool_result, separators=(',', ':'), ensure_ascii=False)
                except TypeError:
                    result_str = str(msg.tool_result)
                turn += f" Result: {result_str}"
        
        # Add turn to prompt only if it has content AND normalized content differs from last
        # Also skip if normalized content is empty (e.g. assistant message was only </s>)
        if turn and normalized_content and normalized_content != last_added_normalized_content:
            prompt_parts.append(turn)
            last_added_normalized_content = normalized_content # Update last added content
        elif not turn:
            logger.debug(f"{log_prefix}Skipping history message index {idx} as it resulted in empty turn string.") # Use prefix
        elif not normalized_content:
             logger.debug(f"{log_prefix}Skipping history message index {idx} as normalized content is empty (e.g., only '</s>').") # Use prefix
        else: # Normalized Content is identical to previous
            logger.debug(f"{log_prefix}Skipping history message index {idx} due to identical consecutive normalized content.") # Use prefix
            
    prompt_parts.append("\nEND HISTORY.")

    # Add Available Tools section (keep existing formatting logic)
    if tools:
        prompt_parts.append("\nAVAILABLE TOOLS:")
        formatted_tools = ""
        for tool in tools:
            tool_name = tool.get('name', 'Unnamed Tool')
            description = tool.get('description', 'No description.')
            parameters_spec = tool.get('parameters') or tool.get('inputSchema')
            formatted_tools += f"\n- Tool Name: `{tool_name}`"
            formatted_tools += f"\n  Description: {description}"
            if parameters_spec and isinstance(parameters_spec, dict):
                properties = parameters_spec.get('properties')
                required = parameters_spec.get('required', [])
                if properties and isinstance(properties, dict) and properties:
                    formatted_tools += "\n  Parameters:"
                    for param_name, param_info in properties.items():
                        param_type = param_info.get('type', 'any')
                        param_desc = param_info.get('description', 'No description.')
                        is_required = "(required)" if param_name in required else "(optional)"
                        formatted_tools += f"\n    - `{param_name}` ({param_type}) {is_required}: {param_desc}"
                else:
                    formatted_tools += "\n  Parameters: None specified."
            else:
                formatted_tools += "\n  Parameters: None defined."
            formatted_tools += "\n"
        prompt_parts.append(formatted_tools)
        prompt_parts.append("END AVAILABLE TOOLS.")
        # print(f"Formatted tools: {formatted_tools}") # Removed print

    # --- Updated Instructions --- 
    instruction_text = "\nINSTRUCTIONS:\n"

    # Case 1: Last message was a TOOL_RESULT
    if last_message_role == 'tool':
        instruction_text += (
            "A tool has just been executed and its result is available in the history above.\n"
            "- Analyze the TOOL_RESULT provided in the conversation history.\n"
            "- <think>Consider the key information from the TOOL_RESULT and the LATEST USER QUESTION.</think>\n"
            "- Generate a final, user-friendly text response that directly answers the LATEST USER QUESTION based on the TOOL_RESULT.\n"
            "- **Your response MUST contain ONLY the final answer for the user.** Do NOT include any prefix like 'Summary:' or 'Answer:'.\n"
            "- **Do NOT call any tools** in this response. **Do NOT output any JSON object.**\n"
        )
    # Case 2: Tools are available, but last message was not a tool result
    elif tools:
        example_json = json.dumps({ # Use json.dumps for robust example generation
            "tool": "some@tool",
            "parameters": {
                "query": "some value with \"quotes\" and \\backslashes\\"
            }
        })
        instruction_text += f"""
Based on the LATEST USER QUESTION and the conversation history, decide if using one of the AVAILABLE TOOLS is necessary to answer the question.

- If a tool is needed:
  - Select the single most relevant tool.
  - Respond ONLY with a single, valid JSON object representing the function call.
  - This JSON object MUST be the *only* content in your response. **Do NOT include ANY other text, reasoning, or commentary before or after the JSON object.**
  - The object must have a 'tool' key (string, exact tool name) and a 'parameters' key (object).
  - The 'parameters' object MUST map parameter names (string) to their values (string, number, boolean, list, or nested object), containing only the actual parameters defined for the tool.
  - Ensure all string values within the parameters object are properly escaped (e.g., use \" for quotes within strings, \\ for backslashes).
  - **Remember: ONLY the JSON object, nothing else!**
  - Example: {example_json}

- If no tool is needed, or no tool is suitable:
  - <think>Consider the relevant information from the conversation history that you are using to answer the question.</think>\n"
  - Generate a helpful and concise text response directly answering the LATEST USER QUESTION based on the history.\n"
  - **Your response MUST contain ONLY the final answer for the user.** Do NOT include any prefix like 'Summary:' or 'Answer:'.\n"
  - Do NOT output any JSON object in this case.
"""
    # Case 3: No tools available
    else:
        instruction_text += (
            "<think>Consider the relevant information from the conversation history that you are using to answer the question.</think>\n"
            "Generate a helpful and concise text response directly answering the LATEST USER QUESTION based on the history.\n"
            "**Your response MUST contain ONLY the final answer for the user.** Do NOT include any prefix like 'Summary:' or 'Answer:'.\n"
        )

    prompt_parts.append(instruction_text)
    # --- End Updated Instructions ---

    # Iteration limit check (keep this guardrail)
    if iteration_count >= max_iterations:
         logger.warning(f"{log_prefix}Maximum iteration count ({max_iterations}) reached. Forcing non-tool response.") # Use prefix
         prompt_parts.append("\nIMPORTANT: You have reached the maximum number of tool call iterations for this request. Do not call any more tools. Provide a final answer to the user based on the information available.")

    # Add final assistant marker
    prompt_parts.append("\n\nAssistant response:")

    final_prompt = "\n".join(prompt_parts)
    logger.debug(f"{log_prefix}Final prompt length: {len(final_prompt)} chars") # Use prefix
    return final_prompt


def parse_llm_response_for_tool_calls(response_text: str, tools_definitions: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Attempts to parse the LLM response, prioritizing JSON object within markdown, then first raw object using raw_decode."""
    response_text = response_text.strip()
    cleaned_response_text = response_text.replace("</think>", "").strip()
    # logging.debug(f"Parser will attempt on cleaned text: '''{cleaned_response_text}'''") # DEBUG

    if not tools_definitions:
        logging.warning("No tool definitions provided for validation.")

    potential_call_object = None
    parsing_method = None

    # Strategy 1: Look for JSON object within markdown code block
    # logging.debug("Attempting Strategy 1: Find JSON in markdown...") # DEBUG
    extracted_json_str_md = None
    try:
        match = re.search(r'```json\\s*(\\{.*?\\})\\s*```', cleaned_response_text, re.DOTALL)
        if match:
            extracted_json_str_md = match.group(1)
            parsing_method = "markdown"
            # logging.debug(f"Extracted JSON object from markdown: {extracted_json_str_md}") # DEBUG
            try:
                fixed_json_str = extracted_json_str_md.replace('{{"', '{"')
                potential_call_object = json.loads(fixed_json_str)
                # logging.debug("Successfully parsed JSON from markdown.") # DEBUG
            except json.JSONDecodeError as md_parse_err:
                # logging.debug(f"Failed to parse JSON extracted from markdown ({md_parse_err}). Trying next strategy...") # DEBUG
                potential_call_object = None 
                parsing_method = None        
                extracted_json_str_md = None 
        # else:
             # logging.debug("Could not find ```json { ... } ``` markdown block. Trying next strategy...") # DEBUG
    except Exception as regex_err:
        logging.error(f"ERROR during regex search for markdown JSON: {regex_err}. Trying next strategy...")

    # Strategy 2: If no valid object from markdown, use raw_decode
    if not potential_call_object:
        # logging.debug("Attempting Strategy 2: Use json.JSONDecoder().raw_decode()...") # DEBUG
        first_brace_index = cleaned_response_text.find('{')
        if first_brace_index != -1:
            try:
                decoder = json.JSONDecoder()
                decoded_object, end_index = decoder.raw_decode(cleaned_response_text[first_brace_index:])
                potential_call_object = decoded_object
                parsing_method = "raw_decode"
                # logging.debug(f"Successfully decoded first JSON object using raw_decode (stopped at index {first_brace_index + end_index}).") # DEBUG
            except json.JSONDecodeError as raw_decode_err:
                # logging.debug(f"raw_decode failed: {raw_decode_err}") # DEBUG
                potential_call_object = None 
            except Exception as e:
                 logging.error(f"Error during raw_decode processing: {e}")
                 potential_call_object = None 
        # else:
            # logging.debug("Could not find starting '{' for raw_decode.") # DEBUG

    if not potential_call_object:
        # logging.debug("Could not extract and parse a valid JSON object via any method.") # DEBUG
        return None

    # Validate the successfully parsed object
    # logging.debug(f"Attempting to validate parsed object (method: {parsing_method}): {potential_call_object}") # DEBUG
    try:
        if not (isinstance(potential_call_object, dict) and 
                isinstance(potential_call_object.get('tool'), str) and 
                isinstance(potential_call_object.get('parameters'), dict)):
            logging.warning(f"Parsed object is not a valid tool call structure: {potential_call_object}")
            return None

        tool_name = potential_call_object['tool']
        parameters = potential_call_object['parameters']
        call_is_valid = True 
        if tools_definitions:
            tool_def = next((t for t in tools_definitions if t.get('name') == tool_name), None)
            if not tool_def:
                logging.warning(f"No definition found for tool '{tool_name}'. Cannot validate required parameters.")
            else:
                param_schema = tool_def.get('parameters') or tool_def.get('inputSchema')
                if param_schema and isinstance(param_schema, dict):
                    required_params = param_schema.get('required', [])
                    if required_params:
                        missing_params = [p for p in required_params if p not in parameters]
                        if missing_params:
                            logging.warning(f"Validation failed for tool '{tool_name}': Missing required parameters: {missing_params}")
                            call_is_valid = False
                else:
                     logging.warning(f"No parameter schema found in definition for tool '{tool_name}'. Cannot validate required parameters.")
        
        if call_is_valid:
            validated_calls_list = [potential_call_object] # Wrap the dict in a list
            # logging.debug(f"Successfully validated tool call (method: {parsing_method}): {validated_calls_list}") # DEBUG
            return validated_calls_list
        else:
            return None 
    except Exception as e:
         logging.exception(f"Error during validation of parsed object (method: {parsing_method}, error: {e})") 
         return None 

# --- Modified generate_interactive_stream function --- 
async def generate_interactive_stream(session_id: str, 
                                      history: List[HistoryMessage], 
                                      tools: Optional[List[Dict[str, Any]]], 
                                      generation_params: 'GenerateRequestParams',
                                      queue: asyncio.Queue[Optional[str]], 
                                      loop: asyncio.AbstractEventLoop
                                      ) -> str: 
    """Generates LLM output using TextIteratorStreamer correctly, puts SSE messages onto a queue via the event loop."""
    log_prefix = f"[Task Runner / Session {session_id}]"
    
    if shared.model is None or shared.tokenizer is None:
        logging.error(f"{log_prefix} Model or Tokenizer not loaded! Cannot generate.")
        # Schedule the error put operation on the main loop
        asyncio.run_coroutine_threadsafe(queue.put(f"event: error\ndata: {json.dumps({'type': 'error', 'error': 'Server Configuration Error', 'details': 'Model not loaded.'})}\n\n"), loop)
        return "ERROR: Model not loaded"
        
    req_id = str(time.time_ns()) 
    task_log_prefix = f"[Task {req_id} / Session {session_id}]"
    logging.info(f"{task_log_prefix} Starting interactive generation (Corrected Streamer Usage)...")
    error_yielded_to_queue = False 
    function_call_request_yielded = False 

    try:
        prompt_text = generate_conversation_prompt(
            history=history, 
            tools=tools, 
            iteration_count=generation_params.iteration_count, 
            max_iterations=generation_params.iteration_count + 5 
        )
        # Modify existing log to include full prompt text (at DEBUG level)
        logging.debug(f"{task_log_prefix} Generated prompt:\n--- PROMPT START ---\n{prompt_text}\n--- PROMPT END ---")

        # --- Corrected TextIteratorStreamer Usage --- 
        streamer = TextIteratorStreamer(shared.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            max_new_tokens=generation_params.max_new_tokens,
            do_sample=generation_params.do_sample,
            temperature=generation_params.temperature if generation_params.do_sample and generation_params.temperature is not None else (0.6 if generation_params.do_sample else None),
            streamer=streamer, # Pass streamer here
            # Add other necessary generation params like top_p, top_k if desired
        )
        if not generation_params.do_sample:
            generation_kwargs.pop('temperature', None)

        inputs = shared.tokenizer(prompt_text, return_tensors="pt").to(shared.model.device)
        
        # model.generate will run in a background thread managed BY THE STREAMER
        # We need to start it non-blockingly. 
        # The typical way is to run model.generate in a separate thread 
        # and then iterate the streamer in the current thread/context.
        
        # Define the target for the thread that runs model.generate
        def generation_thread_target():
             logging.debug(f"{task_log_prefix} [Generate Thread] Starting model.generate...")
             try:
                 shared.model.generate(**inputs, **generation_kwargs)
                 logging.debug(f"{task_log_prefix} [Generate Thread] model.generate finished.")
             except Exception as gen_e:
                 logging.exception(f"{task_log_prefix} [Generate Thread] Exception during model.generate: {gen_e}")
                 # Put error on queue if generate fails catastrophically
                 sse_event = f"event: error\ndata: {json.dumps({'type': 'error', 'error': 'Generation Failed', 'details': f'Core model generation failed: {gen_e}'})}\n\n"
                 asyncio.run_coroutine_threadsafe(queue.put(sse_event), loop)
                 # Also signal end via None? The runner's finally should handle this.

        # Start the generation thread
        generator_thread = threading.Thread(target=generation_thread_target)
        generator_thread.start()
        logging.debug(f"{task_log_prefix} Generation thread started.")
        # --- End Corrected Usage ---

        # --- Now iterate the streamer in the current context (the asyncio task) --- 
        full_response_text = ""
        chunk_count = 0
        logging.debug(f"{task_log_prefix} [Streamer Loop] Entering streamer loop...")
        try:
            for chunk in streamer: # Iterate the streamer directly
                chunk_count += 1
                logging.debug(f"{task_log_prefix} [Streamer Loop] Received chunk {chunk_count}: '{chunk[:50]}...'")
                # Check if function call already handled to potentially stop early
                if function_call_request_yielded or error_yielded_to_queue:
                     logging.debug(f"{task_log_prefix} [Streamer Loop] Exiting loop early (terminal event sent).")
                     break 
                     
                full_response_text += chunk
                match = FUNC_CALL_REGEX.search(full_response_text)
                
                if match:
                    func_name = match.group(1)
                    args_str = match.group(2)
                    logging.info(f"{task_log_prefix} Function call marker detected: {func_name}")
                    try:
                        func_args = json.loads(args_str) if args_str else {}
                        tool_schema = None
                        if tools:
                            tool_schema = next((t for t in tools if t.get('name') == func_name), None)
                        
                        if tool_schema:
                            log_args_str = json.dumps(func_args) if func_args else "(No arguments)"
                            logging.info(f"{task_log_prefix} Valid function call '{func_name}' detected. Args: {log_args_str[:150]}{{'...' if len(log_args_str) > 150 else ''}}. Putting JSON-RPC notification onto queue.")
                            sse_event = f"event: function_call_request\ndata: {json.dumps({'jsonrpc': '2.0', 'method': func_name, 'params': func_args})}\n\n"
                            # This loop runs in the asyncio task, so we can await directly
                            await queue.put(sse_event)
                            function_call_request_yielded = True 
                        else:
                            logging.warning(f"{task_log_prefix} LLM hallucinated function call '{func_name}' not in provided tools for this session.")
                            sse_event = f"event: error\ndata: {json.dumps({'type': 'error', 'error': 'Invalid Function Call', 'details': f'Function {func_name} not found in provided tool list for session {session_id}.'})}\n\n"
                            await queue.put(sse_event)
                            error_yielded_to_queue = True
                    except json.JSONDecodeError as e:
                        logging.error(f"{task_log_prefix} Failed to parse JSON args for {func_name}: {e}")
                        sse_event = f"event: error\ndata: {json.dumps({'type': 'error', 'error': 'Function Call Parse Failed', 'details': f'Invalid JSON arguments for {func_name}: {e}'})}\n\n"
                        await queue.put(sse_event)
                        error_yielded_to_queue = True
                    
                    break # Exit loop after handling function call
                else:
                     sse_event = f"data: {json.dumps({'type': 'text_chunk', 'content': chunk})}\n\n"
                     await queue.put(sse_event)
            # --- End of streamer loop --- 
            logging.debug(f"{task_log_prefix} [Streamer Loop] Exited streamer loop after {chunk_count} chunks.")

            # If loop finished without function call and no error was put on queue
            if not error_yielded_to_queue and not function_call_request_yielded:
                 final_text = FUNC_CALL_REGEX.sub("", full_response_text).strip() 
                 log_final_text = final_text[:200] + ("..." if len(final_text) > 200 else "")
                 logging.info(f"{task_log_prefix} Final text generation completed. Response: '{log_final_text}'")
                 if final_text: 
                     sse_event = f"data: {json.dumps({'type': 'final_text', 'content': final_text})}\n\n"
                     await queue.put(sse_event)
                 else:
                     logging.debug(f"{task_log_prefix} No final text to yield after cleaning.")
        
        except Exception as loop_e:
            logging.exception(f"{task_log_prefix} Exception during streamer loop: {loop_e}")
            # Put error on queue
            sse_event = f"event: error\ndata: {json.dumps({'type': 'error', 'error': 'Stream Processing Error', 'details': f'Server error processing stream: {loop_e}'})}\n\n"
            await queue.put(sse_event)
            error_yielded_to_queue = True
        finally:
             # Wait for the generator thread to finish regardless of loop outcome
             if generator_thread.is_alive():
                 logging.debug(f"{task_log_prefix} Waiting for generation thread to join...")
                 # Need to run blocking join in executor
                 await loop.run_in_executor(None, generator_thread.join)
                 logging.debug(f"{task_log_prefix} Generation thread joined.")
             else:
                 logging.debug(f"{task_log_prefix} Generation thread already finished.")
        
        # Return status to the runner task
        if error_yielded_to_queue:
             return f"COMPLETED_WITH_ERROR"
        elif function_call_request_yielded:
             return f"COMPLETED_FUNCTION_CALL"
        else:
            return f"COMPLETED_OK"
            
    except Exception as e:
        logging.exception(f"{task_log_prefix} Error setting up stream generation: {e}")
        return f"ERROR: {e}"

# --- End Modified Function --- 

# --- Core Generation Logic (modified for WebSocket) ---

async def generate_interactive_stream_ws(
    session_id: str,
    websocket: WebSocket, # Accept WebSocket object
    history: List[HistoryMessage],
    tools: Optional[List[Dict[str, Any]]],
    generation_params: GenerateRequestParams,
    task_id: Optional[str] = None # <<< Added optional task_id parameter
) -> str: # Returns status string back to the caller (websocket handler)
    """Generates LLM output using TextIteratorStreamer, sends JSON-RPC notifications over WebSocket."""
    # --- MODIFIED: Use provided task_id or generate a new one --- 
    if task_id is None:
        task_id = f"task_{str(time.time_ns())[-9:]}" # Generate new ID only if not provided
        logger.debug(f"[WS Session {session_id}] No task_id provided, generated new one: {task_id}")
    else:
        logger.debug(f"[WS Session {session_id}] Using provided task_id: {task_id}")
    # --- End Modification ---
    task_log_prefix = f"[WS Session {session_id} / Task {task_id}]" # Use generated task ID in log

    if shared.model is None or shared.tokenizer is None:
        logger.error(f"{task_log_prefix} Model or Tokenizer not loaded!")
        # Use ErrorNotificationParams for the error payload
        error_params = ErrorNotificationParams(
            session_id=session_id,
            task_id=task_id, # Include task_id even for setup errors
            error=JsonRpcError(code=-32001, message="Server Configuration Error", data="Model not loaded.")
        )
        error_notif = create_jsonrpc_notification(
            method="error",
            params=error_params.model_dump()
        )
        # Use try-except for sending as WS might close unexpectedly
        try:
            await websocket.send_text(error_notif)
        except Exception as send_e:
            logger.warning(f"{task_log_prefix} Failed to send Model not loaded error notification: {send_e}")
        return "ERROR: Model not loaded"

    logger.info(f"{task_log_prefix} Starting interactive generation...")
    error_occurred = False
    function_call_request_sent = False
    generation_complete = False # Flag to ensure end notification is sent
    full_response_text = "" # Accumulate the full text
    tool_call_likely_detected = False # Flag to suppress text chunks once tool call starts

    # --- Helper to safely send notifications ---
    async def _send_notification(method: str, params_model: BaseNotificationParams):
        nonlocal error_occurred
        try:
            notif_str = create_jsonrpc_notification(method=method, params=params_model.model_dump())
            logger.debug(f"{task_log_prefix} Sending notification: {method} (Payload: {notif_str[:100]}...)") # Log before send
            await websocket.send_text(notif_str)
        except Exception as send_e:
            logger.warning(f"{task_log_prefix} Failed to send notification '{method}': {send_e}")
            # --- MODIFICATION: Set error_occurred on ANY send failure --- 
            # Original only set for error, end, function_call_request
            logger.error(f"{task_log_prefix} Setting error_occurred = True due to send failure for '{method}'")
            error_occurred = True

    # --- Error sending wrapper ---
    async def _send_error_notification(etype: str, error_msg: str, details: str):
        nonlocal error_occurred
        logger.error(f"{task_log_prefix} {etype}: {error_msg} - {details}") # Log the error
        error_occurred = True
        error_params = ErrorNotificationParams(
            session_id=session_id,
            task_id=task_id,
            error=JsonRpcError(code=-32000, message=error_msg, data=details)
        )
        await _send_notification(
            method="error",
            params_model=error_params
        )

    try:
        prompt_text = generate_conversation_prompt(
            history=history,
            tools=tools,
            iteration_count=generation_params.iteration_count,
            max_iterations=generation_params.iteration_count + 5,
            session_id=session_id, # Pass session_id
            task_id=task_id        # Pass task_id
        )
        logger.info(f"{task_log_prefix} Generated prompt:\n--- PROMPT START ---\n{prompt_text[:500]}... (truncated)\n--- PROMPT END ---")

        streamer = TextIteratorStreamer(shared.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            max_new_tokens=generation_params.max_new_tokens,
            do_sample=generation_params.do_sample,
            temperature=generation_params.temperature if generation_params.do_sample and generation_params.temperature is not None else (0.6 if generation_params.do_sample else None),
            streamer=streamer,
        )
        if not generation_params.do_sample:
            generation_kwargs.pop('temperature', None)

        inputs = shared.tokenizer(prompt_text, return_tensors="pt").to(shared.model.device)
        loop = asyncio.get_running_loop()

        def generation_thread_target():
            nonlocal generation_complete, error_occurred
            try:
                logger.debug(f"{task_log_prefix} [Generate Thread] Starting model.generate...")
                shared.model.generate(**inputs, **generation_kwargs)
                logger.debug(f"{task_log_prefix} [Generate Thread] model.generate finished.")
            except Exception as thread_e:
                logger.error(f"{task_log_prefix} [Generate Thread] Error during generation: {thread_e}")
                logger.error(traceback.format_exc())
                error_occurred = True
            finally:
                generation_complete = True
                logger.debug(f"{task_log_prefix} [Generate Thread] Generation thread finished.")

        generation_thread = threading.Thread(target=generation_thread_target)
        generation_thread.start()
        logger.debug(f"{task_log_prefix} Started generation thread. Starting to consume streamer...")

        # --- Modified Stream Consumption Loop ---
        for text_chunk in streamer:
            # --- ADDED: Log raw chunk --- 
            logger.debug(f"{task_log_prefix} Raw chunk received: {repr(text_chunk)}")

            if error_occurred: # Stop processing if an error was flagged
                 logger.warning(f"{task_log_prefix} Error occurred, stopping streamer consumption.")
                 break

            full_response_text += text_chunk # Always accumulate full response for post-parsing
            
            # --- ADDED: Log detection flag value --- 
            logger.debug(f"{task_log_prefix} tool_call_likely_detected = {tool_call_likely_detected}")

            # Check if a tool call structure seems likely based on accumulated text
            # Adjust these markers based on your LLM's actual output format for tool calls
            if not tool_call_likely_detected and ('<function_call' in full_response_text or '{"tool":' in full_response_text):
                 logger.debug(f"{task_log_prefix} Tool call structure likely detected. Suppressing further text chunks.")
                 tool_call_likely_detected = True

            # Only send text chunks if no tool call is detected and the chunk has content after cleaning
            if not tool_call_likely_detected and text_chunk:
                # 1. Clean <think> tags
                cleaned_chunk = re.sub(r"<think>.*?</think>", "", text_chunk, flags=re.DOTALL)
                # 2. Clean potential markdown code fences (start and end)
                #    Handles ```json ... ``` or just ``` ... ```
                cleaned_chunk = re.sub(r"^\s*```(?:json)?\n?", "", cleaned_chunk) # Remove opening fence
                cleaned_chunk = re.sub(r"\n?```\s*$", "", cleaned_chunk) # Remove closing fence

                # --- MODIFIED: Log FULL cleaned chunk --- 
                logger.debug(f"{task_log_prefix} Cleaned chunk (after think/markdown removal): {repr(cleaned_chunk)}") # Log full cleaned chunk

                # --- MODIFIED: Check if cleaned_chunk is non-empty OR just whitespace --- 
                if cleaned_chunk or cleaned_chunk.isspace(): 
                    logger.debug(f"{task_log_prefix} Attempting to send cleaned text chunk (non-empty or whitespace)...")
                    chunk_params = TextChunkParams(session_id=session_id, task_id=task_id, content=cleaned_chunk)
                    await _send_notification(method="text_chunk", params_model=chunk_params)
                    # Check if sending the chunk itself caused an error
                    if error_occurred:
                        logger.warning(f"{task_log_prefix} Send failure detected immediately after sending text chunk. Breaking loop.")
                        break # Exit loop if send failed
                else:
                     # This will now only skip truly empty chunks (after <think> and fence removal)
                     logger.debug(f"{task_log_prefix} Skipping send for truly empty cleaned chunk.") 
            elif tool_call_likely_detected:
                 logger.debug(f"{task_log_prefix} Skipping send because tool_call_likely_detected is True.") # Log skipped due to flag


        # --- End of Stream Consumption Loop ---

        # --- Post-Generation Parsing for Tool Call ---
        parsed_tool_call = None
        if not error_occurred and not function_call_request_sent: # Only parse if no error/call already handled
            logger.debug(f"{task_log_prefix} Stream finished. Attempting to parse full response for potential tool call. Length: {len(full_response_text)}")
            # Try finding the JSON object using raw_decode starting from the first brace
            first_brace_index = full_response_text.find('{')
            if first_brace_index != -1:
                # Extract the substring that might contain JSON
                potential_json_str = full_response_text[first_brace_index:]
                # Clean common invalid escape sequences often produced by LLMs
                cleaned_json_str = potential_json_str.replace('\\\"', '\"') # Replace \" with "
                logger.debug(f"{task_log_prefix} Cleaned potential JSON string (first 100 chars): {cleaned_json_str[:100]}")
                try:
                    decoder = json.JSONDecoder()
                    # Decode the cleaned string
                    potential_json_obj, _ = decoder.raw_decode(cleaned_json_str)

                    # Validate structure
                    if isinstance(potential_json_obj, dict) and \
                       isinstance(potential_json_obj.get('tool'), str) and \
                       isinstance(potential_json_obj.get('parameters'), dict):

                        tool_name = potential_json_obj['tool']
                        parameters = potential_json_obj['parameters']
                        logger.info(f"{task_log_prefix} Successfully parsed tool call JSON from full response: {tool_name}({parameters})" )
                        parsed_tool_call = ToolCallRequest(tool=tool_name, parameters=parameters)
                    else:
                        logger.debug(f"{task_log_prefix} Found JSON object, but not a valid tool call structure.")
                except json.JSONDecodeError as decode_error:
                    # Log the cleaned string that failed
                    logger.debug(f"{task_log_prefix} Failed to decode cleaned JSON from full response: {decode_error}. String was: {cleaned_json_str[:200]}...")
                except Exception as parse_err:
                     logger.error(f"{task_log_prefix} Unexpected error during post-generation JSON parsing: {parse_err}")
            else:
                logger.debug(f"{task_log_prefix} No opening brace '{{' found in full response.")

        # --- End Post-Generation Parsing ---

        logger.debug(f"{task_log_prefix} Waiting for generation thread to complete...")
        generation_thread.join(timeout=10.0)
        if generation_thread.is_alive():
             logger.warning(f"{task_log_prefix} Generation thread did not finish within timeout.")
             await _send_error_notification("Timeout Error", "Generation Thread Timeout", "The background generation task did not complete in time.")
             error_occurred = True
        if not generation_complete and not error_occurred:
             logger.warning(f"{task_log_prefix} Streamer finished, but generation thread did not signal completion cleanly.")

    except Exception as e:
        tb_str = traceback.format_exc()
        await _send_error_notification("Generation Error", f"Error during generation stream: {e}", tb_str)

    finally:
        logger.info(f"{task_log_prefix} Cleaning up generation...")
        if generation_thread.is_alive():
             logger.warning(f"{task_log_prefix} Attempting final join on generation thread...")
             generation_thread.join(timeout=1.0)

        # --- Modified Final Notification Logic --- 
        status = ""
        if not error_occurred:
            # Check if we successfully parsed a tool call AFTER the stream ended
            if parsed_tool_call:
                # Send the function call request now
                func_call_params = FunctionCallRequestParams(session_id=session_id, task_id=task_id, tool_call=parsed_tool_call)
                await _send_notification(method="function_call_request", params_model=func_call_params)
                function_call_request_sent = True # Mark as sent for status reporting
                logger.info(f"{task_log_prefix} Sent function call request notification (post-generation parse)." )
                status = "OK: Function call requested"
                # Do NOT send final_text if a tool call was extracted
            else:
                # No tool call parsed. The full response was already sent via text_chunk.
                # Do NOT send final_text again.
                logger.info(f"{task_log_prefix} No tool call parsed. Text response already streamed. Full length: {len(full_response_text)}")
                status = "OK: Completed"
        else:
            logger.warning(f"{task_log_prefix} Generation ended with error.")
            status = "ERROR: Generation failed"
        
        # Always send end notification (signals completion of the task)
        try:
            end_params = StreamEndParams(session_id=session_id, task_id=task_id) # final_text is None here
            await _send_notification(method="end", params_model=end_params)
        except Exception as final_send_e:
             logger.warning(f"{task_log_prefix} Failed to send final 'end' notification: {final_send_e}")

        return status

# --- Helper Functions (Keep generate_conversation_prompt and parse_llm_response_for_tool_calls) ---

def parse_llm_response_for_tool_calls(response_text: str) -> Optional[ToolCallRequest]:
    """
    Parses the LLM response to extract the *first* function call request
    using the specific XML-like format. Returns None if no valid call is found.
    """
    logger.debug(f"Attempting to parse response for tool call: '{response_text[:100]}...'") # Log input

    match = FUNC_CALL_REGEX.search(response_text)
    if not match:
        logger.debug("No function call pattern found in response.")
        return None

    func_name = match.group(1)
    args_str = match.group(2)
    logger.debug(f"Potential function call found: name='{func_name}', args_str='{args_str}'" )

    try:
        # Attempt to decode escaped characters before parsing JSON
        # Be cautious with this; complex escapes might need more robust handling
        try:
            args_str_decoded = bytes(args_str, "utf-8").decode("unicode_escape")
        except Exception as decode_e:
            logger.warning(f"Failed to decode args string '{args_str}' using unicode_escape: {decode_e}. Trying raw string.")
            args_str_decoded = args_str # Fallback to raw string

        args = json.loads(args_str_decoded)
        if not isinstance(args, dict):
            logger.error(f"Parsed arguments are not a dictionary: type={type(args)}, value={args}")
            return None # Arguments must be a dictionary

        logger.info(f"Successfully parsed tool call: {func_name}({args})")
        return ToolCallRequest(tool=func_name, parameters=args)

    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding failed for args: '{args_str_decoded}'. Error: {e}")
        return None
    except Exception as e: # Catch other potential errors
        logger.error(f"Unexpected error parsing tool call arguments: {e}")
        logger.error(traceback.format_exc())
        return None