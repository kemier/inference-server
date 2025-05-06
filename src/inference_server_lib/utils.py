import asyncio
import json
import re
import logging
from typing import Any, Optional, Union, Dict, List, AsyncGenerator, Tuple, Callable
import time
import threading
from . import shared
from .models import (
    HistoryMessage, ToolCallRequest, GenerateRequestParams, TextChunkParams,
    FunctionCallRequestParams, ErrorNotificationParams, StreamEndParams,
    BaseNotificationParams, JsonRpcError, ToolResultItem, OpenAIAssistantToolCall, OpenAIFunctionSpec, JsonRpcErrorCode
)
from starlette.websockets import WebSocket
from starlette.status import WS_1008_POLICY_VIOLATION
import traceback

# --- Langchain Imports ---
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessageChunk, AIMessageChunk
from langchain_core.exceptions import OutputParserException

# --- Logging configuration (moved near top) ---
logging.basicConfig(
    level=logging.DEBUG, # Adjust level as needed (DEBUG, INFO, WARNING, ERROR)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference_server.log", mode='w'), # Ensure fresh log on each run
        logging.StreamHandler() # Log to console as well
    ]
)
logger = logging.getLogger(__name__) # Use a specific logger for utils

# --- ADDED: Set httpx and httpcore loggers to DEBUG --- 
logging.getLogger("httpx").setLevel(logging.DEBUG)
logging.getLogger("httpcore").setLevel(logging.DEBUG)
# --- END ADDED ---

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

        # Normalized content for deduplication check (strip whitespace and ])
        normalized_content = content.strip().removesuffix(']').strip()

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
        # Also skip if normalized content is empty (e.g. assistant message was only ])
        if turn and normalized_content and normalized_content != last_added_normalized_content:
            prompt_parts.append(turn)
            last_added_normalized_content = normalized_content # Update last added content
        elif not turn:
            logger.debug(f"{log_prefix}Skipping history message index {idx} as it resulted in empty turn string.") # Use prefix
        elif not normalized_content:
             logger.debug(f"{log_prefix}Skipping history message index {idx} as normalized content is empty (e.g., only ']').") # Use prefix
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

    # Determine the effective last role, considering if the server just added an instruction after a tool result
    effective_last_role = last_message_role
    if len(history) >= 2 and history[-1].role == 'user' and history[-2].role == 'tool':
        # If the last message is a user instruction following a tool result, treat it like we just got the tool result
        effective_last_role = 'tool' 
        # We might want to make the instruction slightly different in this specific sub-case later if needed.

    # Case 1: Effective last message was a TOOL_RESULT (or server instruction after tool result)
    if effective_last_role == 'tool':
        instruction_text += (
            "A tool has just been executed and its result is available in the history above.\n"
            "- Analyze the TOOL_RESULT provided in the conversation history.\n"
            "- <think>Consider the key information from the TOOL_RESULT and the LATEST USER QUESTION (or the explicit instruction given right after the tool result).</think>\n"
            "- Generate a final, user-friendly text response that directly answers the LATEST USER QUESTION or follows the explicit instruction, based on the TOOL_RESULT.\n"
            "- **Your response MUST contain ONLY the final answer/summary for the user.** Do NOT include any prefix like 'Summary:' or 'Answer:'.\n"
            "- **Do NOT call any tools** in this response. **Do NOT output any JSON object.**\n"
        )
    # Case 2: Tools are available, not immediately after a tool result
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

# --- Core Generation Logic (modified for WebSocket with ChatOpenAI) ---
async def generate_interactive_stream_ws(
    websocket: WebSocket,
    session_id: str,
    task_id: str, 
    history: List[HistoryMessage],
    tools: Optional[List[Dict[str, Any]]],
    generation_params: GenerateRequestParams, # Contains tool_choice, max_tokens etc.
    # send_tool_definitions: bool, # This logic is now implicit (first call vs continuation)
    # callback_fn: Optional[Callable[[Dict[str, Any]], None]] = None # Callback is handled by add_done_callback
) -> Dict[str, Any]: # Return dict with status and final AIMessageChunk
    
    log_prefix = f"[WS Session {session_id} / Task {task_id}] "
    logger.debug(f"{log_prefix}[generate_interactive_stream_ws Entry] generation_params: {generation_params}")

    final_status = "unknown_error"
    # --- MODIFIED: This will become the aggregated chunk ---
    final_ai_message_chunk_for_callback: Optional[AIMessageChunk] = None 
    stream_has_ended_sent = False # Moved up for broader scope

    try:
        logger.debug(f"{log_prefix}Preparing tools and LLM bindings.")
        openai_tools_definitions, tool_name_map, reverse_tool_name_map = format_tools_for_openai(tools)
        
        llm_with_tools = shared.llm
        if openai_tools_definitions:
            tool_choice_to_use = generation_params.tool_choice
            logger.debug(f"{log_prefix}Binding tools. Tool choice: {tool_choice_to_use}")
            try:
                if isinstance(tool_choice_to_use, str) and tool_choice_to_use in ["auto", "any", "none"]:
                    llm_with_tools = shared.llm.bind_tools(tools=openai_tools_definitions, tool_choice=tool_choice_to_use)
                elif isinstance(tool_choice_to_use, dict) and tool_choice_to_use.get("type") == "function" and tool_choice_to_use.get("function", {}).get("name"):
                    llm_with_tools = shared.llm.bind_tools(tools=openai_tools_definitions, tool_choice=tool_choice_to_use)
                elif tool_choice_to_use:
                    logger.warning(f"{log_prefix}Potentially unhandled tool_choice format: {tool_choice_to_use}. Binding all tools.")
                    llm_with_tools = shared.llm.bind_tools(tools=openai_tools_definitions)
                else:
                    llm_with_tools = shared.llm.bind_tools(tools=openai_tools_definitions)
            except Exception as tool_bind_e:
                logger.error(f"{log_prefix}Error binding tools to LLM: {tool_bind_e}", exc_info=True)
                await _send_notification_ws(websocket, "error", ErrorNotificationParams(session_id=session_id, task_id=task_id, error=JsonRpcError(code=JsonRpcErrorCode.INTERNAL_ERROR, message=f"Server error binding tools: {tool_bind_e}")), log_prefix)
                return {"status": "tool_binding_error", "final_ai_message_chunk": None, "error_message": str(tool_bind_e)}
        else:
            logger.debug(f"{log_prefix}No tools to bind or no tools defined.")

        logger.debug(f"{log_prefix}Generating Langchain messages from history.")
        langchain_messages = generate_langchain_messages(
            history=history,
            tools=None, 
            iteration_count=generation_params.iteration_count,
            session_id=session_id,
            task_id=task_id
        )
        logger.debug(f"{log_prefix}Invoking LLM stream with {len(langchain_messages)} messages.")

        full_response_content = ""
        # --- MODIFIED: Collect all AIMessageChunks ---
        all_ai_chunks_from_stream: List[AIMessageChunk] = []
        aggregated_final_tool_calls: List[OpenAIAssistantToolCall] = []
        # stream_has_ended_sent = False # Moved up

        try:
            llm_stream = llm_with_tools.astream(langchain_messages, {
                "max_tokens": generation_params.max_new_tokens,
            })
            
            # --- ADDED: Log before entering the stream loop ---
            logger.debug(f"{log_prefix}LLM astream() called, llm_stream object: {llm_stream}. About to iterate...")
            # --- END ADDED ---

            async for chunk in llm_stream:
                # --- ADDED: Log immediately upon entering loop iteration ---
                logger.debug(f"{log_prefix}Entered stream loop iteration. Chunk received: {type(chunk)}")
                # --- END ADDED ---
                if not isinstance(chunk, AIMessageChunk):
                    logger.warning(f"{log_prefix}Received non-AIMessageChunk from stream: {type(chunk)} - {chunk}")
                    continue

                all_ai_chunks_from_stream.append(chunk) # Accumulate

                if chunk.content:
                    logger.debug(f"{log_prefix}Text chunk: '{chunk.content}'")
                    full_response_content += chunk.content
                    await _send_notification_ws(websocket, "text_chunk", TextChunkParams(session_id=session_id, task_id=task_id, content=chunk.content), log_prefix)
                
                if chunk.tool_call_chunks: # Log raw chunks for debugging if needed
                    for tc_chunk in chunk.tool_call_chunks:
                        logger.debug(f"{log_prefix}Raw Tool call chunk received: {tc_chunk}")
            
            logger.info(f"{log_prefix}LLM stream finished iterating. Total AIMessageChunks: {len(all_ai_chunks_from_stream)}")

        except OutputParserException as ope:
            logger.error(f"{log_prefix}OutputParserException during LLM stream: {ope}", exc_info=True)
            # Ensure the error code is an int
            error_code_val = JsonRpcErrorCode.INTERNAL_ERROR if isinstance(JsonRpcErrorCode.INTERNAL_ERROR, int) else JsonRpcErrorCode.INTERNAL_ERROR.value
            await _send_notification_ws(websocket, "error", ErrorNotificationParams(session_id=session_id, task_id=task_id, error=JsonRpcError(code=error_code_val, message=f"LLM Output Parsing Error: {ope}")), log_prefix)
            return {"status": "llm_output_parser_error", "final_ai_message_chunk": None, "error_message": str(ope), "aggregated_openai_tool_calls": []}
        except Exception as stream_e:
            logger.error(f"{log_prefix}Exception during LLM stream: {type(stream_e).__name__} - {stream_e}", exc_info=True)
            # Ensure the error code is an int
            error_code_val = JsonRpcErrorCode.INTERNAL_ERROR if isinstance(JsonRpcErrorCode.INTERNAL_ERROR, int) else JsonRpcErrorCode.INTERNAL_ERROR.value
            critical_error_params = ErrorNotificationParams(
                session_id=session_id, 
                task_id=task_id, 
                error=JsonRpcError(code=error_code_val, message=f"LLM Stream Error: {type(stream_e).__name__} - {stream_e}")
            )
            try:
                await _send_notification_ws(websocket, "error", critical_error_params, log_prefix)
            except Exception as send_e:
                logger.error(f"{log_prefix}Failed to send critical error notification: {send_e}", exc_info=True)
            # Make sure to return a dict that includes all expected keys by the callback
            return {
                "status": "critical_error_in_utils_generate_stream", 
                "final_ai_message_chunk": None, 
                "aggregated_openai_tool_calls": [], # Add this key
                "error_message": str(stream_e)
            }
        
        # --- MODIFIED: Post-Stream Processing using aggregated chunks ---
        if all_ai_chunks_from_stream:
            # Concatenate all received AIMessageChunks
            concatenated_ai_chunk = all_ai_chunks_from_stream[0]
            if len(all_ai_chunks_from_stream) > 1:
                for i in range(1, len(all_ai_chunks_from_stream)):
                    concatenated_ai_chunk = concatenated_ai_chunk + all_ai_chunks_from_stream[i]
            
            final_ai_message_chunk_for_callback = concatenated_ai_chunk # This is what callback gets
            logger.debug(f"{log_prefix}Concatenated AIMessageChunk ID: {final_ai_message_chunk_for_callback.id}, Content: '{final_ai_message_chunk_for_callback.content}', Tool Calls: {final_ai_message_chunk_for_callback.tool_calls}")

            if final_ai_message_chunk_for_callback.tool_calls:
                logger.debug(f"{log_prefix}Processing {len(final_ai_message_chunk_for_callback.tool_calls)} tool_calls from concatenated AIMessageChunk.")
                for complete_tc_dict in final_ai_message_chunk_for_callback.tool_calls: # This is List[Dict]
                    try:
                        tc_id = complete_tc_dict.get('id')
                        tc_name_llm = complete_tc_dict.get('name')
                        # --- MODIFIED: Expect args to be a dict now from concatenated chunk --- 
                        tc_args_dict = complete_tc_dict.get('args') 

                        # --- MODIFIED: Validation for tc_args_dict ---
                        if not (tc_id and tc_name_llm and isinstance(tc_args_dict, dict)):
                            logger.warning(f"{log_prefix}Skipping malformed tool_call from concatenated chunk (id, name, or args not valid type): {complete_tc_dict}")
                            continue

                        client_tool_name = reverse_tool_name_map.get(tc_name_llm, tc_name_llm)
                        logger.info(f"{log_prefix}Tool name '{tc_name_llm}' (LLM) mapped to '{client_tool_name}' (Client) for notification.")
                        
                        # tc_args_dict is already a dict, use it directly for notification parameters
                        tool_call_req = ToolCallRequest(id=tc_id, tool=client_tool_name, parameters=tc_args_dict)
                        await _send_notification_ws(websocket, "function_call_request", FunctionCallRequestParams(session_id=session_id, task_id=task_id, tool_call=tool_call_req), log_prefix)
                        logger.info(f"{log_prefix}Sent function_call_request for tool: {client_tool_name} (LLM used: {tc_name_llm}). LLM Tool Call ID: {tc_id}")
                        
                        # --- MODIFIED: Serialize tc_args_dict to JSON string for OpenAIFunctionSpec --- 
                        try:
                            tc_args_str_for_spec = json.dumps(tc_args_dict)
                        except TypeError as json_dump_err:
                            logger.error(f"{log_prefix}Could not serialize tool arguments dict to JSON string for OpenAIFunctionSpec (tool: {tc_name_llm}, id: {tc_id}): {json_dump_err}. Args: {tc_args_dict}")
                            # Decide on fallback: skip, or use a placeholder like "{}"
                            continue # Skip this tool call if args can't be serialized

                        aggregated_final_tool_calls.append(
                            OpenAIAssistantToolCall(
                                id=tc_id, type='function', 
                                function=OpenAIFunctionSpec(name=tc_name_llm, arguments=tc_args_str_for_spec)
                            )
                        )
                    except Exception as agg_tc_e:
                        logger.error(f"{log_prefix}Error processing an aggregated tool call dict ({complete_tc_dict}): {agg_tc_e}", exc_info=True)
                
                if aggregated_final_tool_calls:
                    final_status = "function_call_requested"
                    # Update the final_ai_message_chunk_for_callback to include our parsed OpenAIAssistantToolCall list
                    # This is if the callback expects this specific format directly.
                    # However, the AIMessageChunk's .tool_calls (List[Dict]) is standard.
                    # The _generation_done_callback needs to be robust to handle the List[Dict] from .tool_calls
                    # For consistency and if the callback needs OpenAIAssistantToolCall, we might need to adjust it
                    # or make final_ai_message_chunk_for_callback hold these.
                    # For now, the callback gets the AIMessageChunk with its raw .tool_calls (List[Dict]).
                    # Let's ensure the HistoryMessage in server_starlette.py is created with OpenAIAssistantToolCall list.
                    # So, `aggregated_final_tool_calls` is important for server_starlette.
                    # The `final_ai_message_chunk_for_callback` (an AIMessageChunk) is returned to server_starlette, which then uses its .content and .tool_calls.
                    # We need to ensure the callback processing in server_starlette uses this `aggregated_final_tool_calls` data.
                    # The simplest is if `final_ai_message_chunk_for_callback.tool_calls` (List[Dict]) is what it uses.
                    # The `_generation_done_callback` will receive `final_ai_message_chunk_for_callback`. It should then create
                    # the `HistoryMessage` using `OpenAIAssistantToolCall` based on `final_ai_message_chunk_for_callback.tool_calls`
                    # or we pass `aggregated_final_tool_calls` back separately.
                    # Let's pass `aggregated_final_tool_calls` in the return dict.
                    logger.info(f"{log_prefix}Final OpenAI-formatted tool calls for history: {len(aggregated_final_tool_calls)}")

            else: # No tool calls in the concatenated chunk
                logger.info(f"{log_prefix}No tool calls found in concatenated AIMessageChunk.")
                # final_status will be determined by content below

        else: # No AIMessageChunks received from stream
            logger.warning(f"{log_prefix}No AIMessageChunks were received from the LLM stream.")
            final_status = "completed_empty_response" # Or an error status if this is unexpected

        # Determine final status based on content and tool calls
        if final_status == "function_call_requested":
            pass # Already set
        elif full_response_content:
            final_status = "completed_with_text"
        elif not aggregated_final_tool_calls: # No content and no tool calls
             # if already an error status, keep it
            if final_status not in ["llm_stream_error", "llm_output_parser_error", "tool_binding_error", "tool_processing_error", "critical_error_in_utils_generate_stream"]:
                final_status = "completed_empty_response"
        
        # Send Final 'end' Notification
        if not stream_has_ended_sent:
            end_status_message = "Completed"
            if final_status == "function_call_requested":
                end_status_message = "Function call(s) requested"
            elif final_status == "completed_empty_response":
                end_status_message = "Completed (empty)"
            elif final_status.startswith("llm_") or final_status.startswith("tool_") or final_status.startswith("critical_"):
                 end_status_message = f"Error: {final_status}"


            # Ensure final_text is only sent if status is completed_with_text
            text_to_send_on_end = full_response_content if final_status == "completed_with_text" else None
            
            logger.debug(f"{log_prefix}Sending 'end' notification: session_id={session_id}, task_id={task_id}, final_text={text_to_send_on_end}, status_message='{end_status_message}'")
            await _send_notification_ws(websocket, "end", StreamEndParams(session_id=session_id, task_id=task_id, final_text=text_to_send_on_end, status_message=end_status_message), log_prefix)
            stream_has_ended_sent = True
        
        logger.info(f"{log_prefix}Exiting generate_interactive_stream_ws. Final status: {final_status}")
        # --- MODIFIED: Return aggregated_final_tool_calls (List[OpenAIAssistantToolCall]) for server_starlette ---
        return {
            "status": final_status, 
            "final_ai_message_chunk": final_ai_message_chunk_for_callback, # This is the concatenated AIMessageChunk
            "aggregated_openai_tool_calls": aggregated_final_tool_calls, # List[OpenAIAssistantToolCall]
            "error_message": None
        }

    except Exception as e_outer:
        logger.critical(f"{log_prefix}CRITICAL UNHANDLED EXCEPTION in generate_interactive_stream_ws: {type(e_outer).__name__} - {e_outer}", exc_info=True)
        final_status = "critical_error_in_utils_generate_stream"
        try:
            if not stream_has_ended_sent:
                await _send_notification_ws(websocket, "error", ErrorNotificationParams(session_id=session_id, task_id=task_id, error=JsonRpcError(code=JsonRpcErrorCode.INTERNAL_ERROR.value, message=f"Critical Server Error: {e_outer}")), log_prefix)
                await _send_notification_ws(websocket, "end", StreamEndParams(session_id=session_id, task_id=task_id, status_message=f"Error: {final_status}"), log_prefix)
                # stream_has_ended_sent = True # Not strictly needed here as it's inside an except
        except Exception as e_notify_critical:
            logger.error(f"{log_prefix}Failed to send critical error notification: {e_notify_critical}")
        return {
            "status": final_status, 
            "final_ai_message_chunk": None, 
            "aggregated_openai_tool_calls": [],
            "error_message": str(e_outer)
        }
    finally:
        logger.debug(f"{log_prefix}Executing FINALLY block of generate_interactive_stream_ws.")
        if not stream_has_ended_sent: # Should ideally be sent before finally if all goes well or caught error
            logger.warning(f"{log_prefix}Stream end notification was not sent prior to finally block. Sending now with status: {final_status}.")
            try:
                # Determine appropriate message for this forced end
                final_end_status_msg = f"Ended ({final_status})"
                if final_status == "unknown_error" and full_response_content: # If outer error but we had content
                    final_end_status_msg = "Completed (potentially with errors)"
                elif final_status == "unknown_error":
                    final_end_status_msg = "Ended (unknown error)"
                
                await _send_notification_ws(websocket, "end", StreamEndParams(session_id=session_id, task_id=task_id, status_message=final_end_status_msg), log_prefix)
            except Exception as e_notify_finally:
                 logger.error(f"{log_prefix}Failed to send final 'end' notification in finally block: {e_notify_finally}")
        logger.info(f"{log_prefix}generate_interactive_stream_ws finished execution. Status for caller: {final_status}")


async def _send_notification_ws(
    websocket: WebSocket,
    method: str,
    params_model: BaseNotificationParams,
    log_prefix: str = ""
):
    """Helper to send JSON-RPC notifications over WebSocket."""
    notification_payload = params_model.model_dump()
    # The 'method' for JSON-RPC is distinct from the 'event type' some clients might use.
    # Here, 'method' is the JSON-RPC method name.
    # Original code used "text_chunk", "function_call_request", "end", "error" as methods.
    # This seems fine.
    notif_str = create_jsonrpc_notification(method=method, params=notification_payload)
    try:
        await websocket.send_text(notif_str)
        logger.debug(f"{log_prefix} Sent '{method}' notification: {notification_payload}")
    except Exception as e:
        logger.warning(f"{log_prefix} Failed to send WebSocket notification '{method}': {e}")

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

# --- Tool Definition Formatting for OpenAI ---
def format_tools_for_openai(tools_definitions: Optional[List[Dict[str, Any]]]) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, str], Dict[str, str]]:
    if not tools_definitions:
        return None, {}, {}

    formatted_tools = []
    original_to_sanitized_map: Dict[str, str] = {}
    sanitized_to_original_map: Dict[str, str] = {}
    valid_char_pattern = re.compile(r"[^a-zA-Z0-9_-]")

    for tool_def_item in tools_definitions: # Iterate through the list of tool definitions
        # Each item in tools_definitions is expected to be a dict like:
        # {"type": "function", "function": {"name": "actual_tool_name", ...}}
        if not isinstance(tool_def_item, dict) or tool_def_item.get("type") != "function":
            logger.warning(f"Skipping tool, not a function type or malformed: {tool_def_item}")
            continue
        
        func_dict = tool_def_item.get("function")
        if not isinstance(func_dict, dict) or "name" not in func_dict:
            logger.warning(f"Skipping tool, 'function' key missing, not a dict, or 'name' missing in function: {tool_def_item}")
            continue

        original_name = func_dict["name"]
        description = func_dict.get("description", "")
        parameters = func_dict.get("parameters", {"type": "object", "properties": {}})

        sanitized_name = valid_char_pattern.sub("_", original_name)
        if not sanitized_name:
            sanitized_name = f"tool_{len(original_to_sanitized_map)}"

        temp_sanitized_name = sanitized_name
        collision_counter = 0
        while temp_sanitized_name in sanitized_to_original_map and sanitized_to_original_map[temp_sanitized_name] != original_name:
            collision_counter += 1
            temp_sanitized_name = f"{sanitized_name}_{collision_counter}"
        sanitized_name = temp_sanitized_name
        
        original_to_sanitized_map[original_name] = sanitized_name
        sanitized_to_original_map[sanitized_name] = original_name
        
        logger.debug(f"Tool name sanitization: Original='{original_name}', Sanitized='{sanitized_name}'")

        formatted_tools.append({
            "type": "function",
            "function": {
                "name": sanitized_name,
                "description": description,
                "parameters": parameters
            }
        })
    return formatted_tools if formatted_tools else None, original_to_sanitized_map, sanitized_to_original_map

# --- Updated generate_conversation_prompt ---
def generate_langchain_messages(
    history: List[HistoryMessage],
    tools: Optional[List[Dict[str, Any]]] = None, # Will be formatted for OpenAI
    iteration_count: int = 0,
    max_iterations: int = 5,
    session_id: Optional[str] = None,
    task_id: Optional[str] = None
) -> List[Union[SystemMessage, HumanMessage, AIMessage, ToolMessage]]:
    log_prefix = f"[Session {session_id} / Task {task_id}] " if session_id and task_id else ""
    messages: List[Union[SystemMessage, HumanMessage, AIMessage, ToolMessage]] = []

    # System instruction part - simplified for ChatOpenAI
    system_prompt_text = "You are a helpful assistant."
    if tools:
        system_prompt_text += " You have access to the following tools. Use them when appropriate by outputting a JSON object with 'tool' and 'parameters'."
        system_prompt_text += " Only use the tools provided. If a tool is needed, respond *only* with the JSON for the tool call."
        system_prompt_text += " If a tool is not needed, or after a tool has been called and you have its result, respond to the user directly with a natural language message."

    # Add system message if it's the first turn or if strategy requires it
    if not any(isinstance(msg, SystemMessage) for msg in messages): # crude check, history might have it
         messages.append(SystemMessage(content=system_prompt_text))


    for h_msg in history:
        captured_tool_call_id = getattr(h_msg, 'tool_call_id', None) 
        logger.debug(f"{log_prefix} Processing history message: role='{h_msg.role}', captured_tool_call_id='{captured_tool_call_id!r}'")

        if h_msg.role == "user":
            messages.append(HumanMessage(content=h_msg.content or ""))
        elif h_msg.role == "assistant":
            ai_content = h_msg.content or ""
            langchain_tool_calls_for_ai_message = []
            if h_msg.tool_calls:
                logger.debug(f"{log_prefix} Processing assistant message with h_msg.tool_calls: {h_msg.tool_calls!r}")
                for tc_openai_fmt in h_msg.tool_calls:
                    if isinstance(tc_openai_fmt, OpenAIAssistantToolCall) and tc_openai_fmt.type == "function":
                        try:
                            parsed_args = json.loads(tc_openai_fmt.function.arguments)
                            if not isinstance(parsed_args, dict):
                                logger.error(f"{log_prefix} Parsed tool call arguments is not a dict: {parsed_args}. Using raw string as fallback.")
                                parsed_args = { "_raw_args": tc_openai_fmt.function.arguments } 
                            langchain_tool_calls_for_ai_message.append({
                                "id": tc_openai_fmt.id,
                                "name": tc_openai_fmt.function.name,
                                "args": parsed_args
                            })
                        except json.JSONDecodeError as e:
                            logger.error(f"{log_prefix} Failed to parse JSON arguments for tool call {tc_openai_fmt.function.name} (ID: {tc_openai_fmt.id}): {e}. Raw args: {tc_openai_fmt.function.arguments}")
                        except Exception as json_e:
                            logger.error(f"{log_prefix} Failed to parse JSON arguments for tool call {tc_openai_fmt.function.name} (ID: {tc_openai_fmt.id}): {json_e}. Raw args: {tc_openai_fmt.function.arguments}")
                    else:
                        logger.warning(f"{log_prefix} Encountered unexpected tool_call structure in history: {tc_openai_fmt}")
            
            logger.debug(f"{log_prefix} Creating AIMessage with tool_calls: {langchain_tool_calls_for_ai_message!r}")
            messages.append(AIMessage(content=ai_content, tool_calls=langchain_tool_calls_for_ai_message))
        elif h_msg.role == "tool":
            current_tool_call_id = captured_tool_call_id 
            logger.debug(f"{log_prefix} Inside role='tool' block. h_msg type: {type(h_msg)}, current_tool_call_id: {current_tool_call_id!r}, h_msg.tool_name: {h_msg.tool_name!r}")

            if not current_tool_call_id or not isinstance(current_tool_call_id, str):
                 logger.error(f"{log_prefix} CRITICAL: Captured tool_call_id is invalid ('{current_tool_call_id!r}') for role='tool'. Skipping ToolMessage creation.")
                 continue 

            tool_content_str = ""
            if h_msg.tool_result is not None:
                try:
                    tool_content_str = json.dumps(h_msg.tool_result)
                except TypeError:
                    tool_content_str = str(h_msg.tool_result)
            elif h_msg.tool_error:
                tool_content_str = f"Error: {h_msg.tool_error}"
                logger.warning(f"{log_prefix} Tool call for '{h_msg.tool_name}' (ID: {current_tool_call_id}) had an error: {h_msg.tool_error}")
            else:
                tool_content_str = "Error: Tool result and error are both missing."
            
            messages.append(ToolMessage(content=tool_content_str, tool_call_id=current_tool_call_id, name=h_msg.tool_name))
        elif h_msg.role == "system":
            if h_msg.content:
                # If a system message is already at the start, replace its content or add if different
                # This handles cases where history might contain multiple system messages.
                # For now, let's assume the first system message (if any) is the one we set initially.
                # If subsequent system messages appear in history, add them.
                if messages and isinstance(messages[0], SystemMessage) and messages[0].content == system_prompt_text:
                    if h_msg.content != system_prompt_text: # Add if different from initial one
                         messages.append(SystemMessage(content=h_msg.content))
                else:
                    messages.append(SystemMessage(content=h_msg.content))
            else:
                logger.warning(f"{log_prefix} System message in history has no content. Skipping.")

    if iteration_count >= max_iterations:
         logger.warning(f"{log_prefix}Maximum iteration count ({max_iterations}) reached. Adding instruction to respond without tools.")
         messages.append(HumanMessage(content="IMPORTANT: You have reached the maximum number of tool call iterations for this request. Do not call any more tools. Provide a final answer to the user based on the information available."))

    # --- ADDED: History Truncation Logic ---
    # Approx. token counting (char length) and truncation
    # DeepSeek context is 65536 tokens. Let's aim for a buffer.
    # OpenAI recommends counting tokens with their tokenizer for accuracy.
    # This is a simpler character-based heuristic.
    # 1 token ~= 4 chars in English. Max target chars = 60000 (approx 15k tokens, well within 65k)
    MAX_TOTAL_CHARS = 60000 
    
    current_total_chars = sum(
        len(str(msg.content)) + 
        sum(
            len(str(tc.get("args", ""))) 
            for tc in getattr(msg, 'tool_calls', []) if isinstance(tc, dict) # Ensure tc is a dict before .get
        ) 
        for msg in messages
    )

    if current_total_chars > MAX_TOTAL_CHARS:
        logger.warning(f"{log_prefix}Approx. message length ({current_total_chars} chars) exceeds target ({MAX_TOTAL_CHARS} chars). Truncating...")
        
        truncated_messages: List[Union[SystemMessage, HumanMessage, AIMessage, ToolMessage]] = []
        # Preserve the first message if it's a SystemMessage (our main prompt)
        has_initial_system_message = False
        if messages and isinstance(messages[0], SystemMessage):
            truncated_messages.append(messages[0])
            current_total_chars = len(str(messages[0].content)) # Recalculate with preserved system message
            messages_to_truncate = messages[1:]
            has_initial_system_message = True
        else:
            messages_to_truncate = list(messages) # Operate on a copy
            current_total_chars = 0 # Start from scratch if no system message to preserve

        # Add messages from the end until the limit is met
        temp_reversed_messages = []
        for msg in reversed(messages_to_truncate):
            msg_len = len(str(msg.content))
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    if isinstance(tc, dict) and isinstance(tc.get("args"), dict):
                         msg_len += len(json.dumps(tc.get("args", {}))) # more accurate for dicts
                    elif isinstance(tc, dict) and isinstance(tc.get("args"), str):
                         msg_len += len(tc.get("args",""))
            
            if current_total_chars + msg_len <= MAX_TOTAL_CHARS:
                temp_reversed_messages.append(msg)
                current_total_chars += msg_len
            else:
                logger.info(f"{log_prefix}Truncation: Dropping older messages. Current char count: {current_total_chars}, next msg length: {msg_len}. Limit: {MAX_TOTAL_CHARS}.")
                break # Stop adding messages
        
        # Combine preserved system message (if any) with the selected messages (reversed back to original order)
        if has_initial_system_message:
            messages = [messages[0]] + list(reversed(temp_reversed_messages))
        else:
            messages = list(reversed(temp_reversed_messages))

        final_char_count = sum(
            len(str(m.content)) + 
            sum(
                len(str(tc.get("args", ""))) 
                for tc in getattr(m, 'tool_calls', []) if isinstance(tc, dict) # Ensure tc is a dict
            ) 
            for m in messages
        )
        logger.info(f"{log_prefix}History truncated. New approx. length: {final_char_count} chars. Number of messages: {len(messages)}.")

    # --- END History Truncation Logic ---

    logger.debug(f"{log_prefix}Generated {len(messages)} Langchain messages for LLM.")
    # for i, msg in enumerate(messages):
    #    logger.debug(f"{log_prefix} Msg {i}: type={type(msg).__name__}, content='{str(msg.content)[:100]}...', tool_calls={getattr(msg, 'tool_calls', None)}")

    return messages