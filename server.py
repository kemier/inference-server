import asyncio
import json
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import os
import torch
from typing import Any, Optional, Union, Dict, List
import re # Add import for regex
from threading import Thread # For streaming
from transformers import TextIteratorStreamer # For streaming
from fastapi.responses import StreamingResponse # For streaming


# --- Model Configuration ---
# Choose how to specify the model:
# Option 1: Use a model ID (requires download on first run if not cached)
# MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" # Comment this out
# Option 2: Use a local path (if downloaded with --local-dir)
MODEL_ID = "./Qwen-14B" # Uncomment this line and ensure the path is correct

# --- Global Variables ---
llm_pipeline = None
model = None # ADDED for direct generation
tokenizer = None # ADDED for direct generation
# --- REMOVE or Comment Out Global Tool Definition ---
# AVAILABLE_TOOLS = [ ... ]

# --- JSON-RPC 2.0 Models ---

# Standard JSON-RPC 2.0 Error Codes (subset)
class JsonRpcErrorCode:
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

class JsonRpcError(BaseModel):
    code: int
    message: str
    data: Optional[Any] = None

class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Union[Dict[str, Any], List[Any]]] = None
    id: Optional[Union[str, int]] = None # Notification if null/missing, Request otherwise

class JsonRpcResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[JsonRpcError] = None
    id: Union[str, int, None] # Must be same as request id

# --- Tool Related Models (Can be reused or refined) ---
class ToolCallRequest(BaseModel): # Model for server requesting client to call a tool
    tool: str
    parameters: Dict[str, Any]

class ToolResultInput(BaseModel): # Model for client providing tool execution result
    tool_name: str = Field(..., alias="toolName") # Use ... for required field
    result: Any
    error: Optional[str] = None # Added error field for tool execution errors

    class Config:
        populate_by_name = True
        extra = "ignore"

# --- Message History Model ---
class HistoryMessage(BaseModel):
    role: str # 'user', 'assistant', 'tool'
    content: Optional[Union[str, List[Dict[str, Any]]]] = None # Text for user/assistant, tool calls for assistant, text result for tool
    tool_calls: Optional[List[ToolCallRequest]] = None # LLM requested tool calls
    tool_name: Optional[str] = None # Name of the tool that was called (for role='tool')
    tool_result: Optional[Any] = None # Result from tool execution (for role='tool')
    tool_error: Optional[str] = None # Error from tool execution (for role='tool')

# --- Method Specific Params/Result Models ---

# Params for 'create_message' method - Modified for multi-turn
class CreateMessageParams(BaseModel):
    # If history is provided, message is ignored (or treated as the latest user message)
    message: Optional[str] = None
    history: Optional[List[HistoryMessage]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[ToolResultInput]] = None # Results for calls requested in *previous* turn
    iteration_count: int = 0 # Track iteration depth
    max_new_tokens: int = 4096 # <-- Increased default value significantly
    do_sample: bool = True

# Result for 'create_message' method - Modified for multi-turn
class CreateMessageResult(BaseModel):
    type: str  # "tool_calls" or "final_text"
    content: Union[str, List[ToolCallRequest]] # Final text or list of tool calls requested for *next* turn
    history: List[HistoryMessage] # The complete updated history
    iteration_count: int # The updated iteration count

# (No specific params model needed for 'tool_list')

# Result for 'tool_list' method
class ToolListResult(BaseModel):
    tools: List[Dict[str, Any]] # Return the list of available tools

# --- Helper Functions ---

def generate_conversation_prompt(history: List[HistoryMessage], tools: Optional[List[Dict[str, Any]]] = None, iteration_count: int = 0, max_iterations: int = 30) -> str:
    """Generates the prompt for the LLM based on conversation history and available tools."""
    prompt = """
This is a conversation between a user and an assistant. The assistant has access to tools.

CONVERSATION HISTORY:
"""

    # --- ADDED: Track already called tools --- 
    already_called_tools = set()
    # --- MODIFIED: Iterate with index for last message check ---
    for idx, msg in enumerate(history):
        if msg.role == 'assistant' and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.tool if isinstance(tc, ToolCallRequest) else tc.get('tool')
                if tool_name:
                    already_called_tools.add(tool_name)
    # --- END Tracking ---

    # --- MODIFIED: Format history with index to mark last user message ---
    for idx, msg in enumerate(history):
        is_last_message = (idx == len(history) - 1)
        if msg.role == 'user':
            # Add special marker if it's the very last message in the history
            prefix = "\nLATEST USER QUESTION: User: " if is_last_message else "\nUser: "
            prompt += f"{prefix}{msg.content}"
        elif msg.role == 'assistant':
            if msg.content:
                prompt += f"\nAssistant: {msg.content}"
            if msg.tool_calls:
                try:
                    tool_calls_list_of_dicts = [tc.model_dump() for tc in msg.tool_calls]
                    calls_str = json.dumps(tool_calls_list_of_dicts)
                    prompt += f"\nAssistant (requests tool calls): {calls_str}"
                except Exception as dump_err:
                    print(f"ERROR serializing tool_calls in history: {dump_err}")
                    prompt += "\nAssistant (requests tool calls): [Error serializing tool calls]"
        elif msg.role == 'tool':
            prompt += f"\nTool ({msg.tool_name}): "
            if msg.tool_error:
                prompt += f"Error: {msg.tool_error}"
            else:
                try:
                    result_str = json.dumps(msg.tool_result, separators=(',', ':'))
                except TypeError:
                    result_str = str(msg.tool_result)
                prompt += f"Result: {result_str}"
        prompt += "\n"
    # --- END Modified Formatting ---

    prompt += "\nCURRENT TASK:\n"
    # Add Tool Definitions if available
    if tools:
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

        prompt += f"You have access to the following tools:\n{formatted_tools}\n"
    else:
        prompt += "You do not have access to any tools for this request.\n"

    # Add Instructions (Reverted to prioritize history check and include constraint)
    if iteration_count >= max_iterations:
        prompt += f"\nINSTRUCTIONS: You have reached the maximum tool iteration limit ({max_iterations}). Generate the final response for the user based *only* on the conversation history above. Do not request any more tool calls."
    elif tools:
        # Reverted instruction structure
        instruction_text = f"""\
INSTRUCTIONS: Based on the **entire conversation history** (especially the LATEST USER QUESTION) and the available tools (including their parameters), **carefully consider the user's overall goal and decide the next step.**
**First, review the 'Tool (...) Result:' entries in the history. Does the information needed to answer the user's latest request *already exist* there?**
**Your primary goal is to answer the user directly using information from the history if possible.** Only consider using a tool if the answer cannot be found in the conversation history and the tool is *essential* to fulfill the user's request.\n"
1.  If the necessary information **is present in the history's tool results**, generate the final response text summarizing that information. **Do NOT call a tool again if the answer is already in the history.**\n"
2.  If, **and only if**, the information is *not* in the history and you *absolutely need* external information or an action performed, choose the **single most relevant tool**. Then, respond with *only* a **single JSON object**. "
"""
        # Re-add constraint about already called tools
        if already_called_tools:
            called_tools_str = ", ".join(f'`{t}`' for t in sorted(list(already_called_tools)))
            instruction_text += f"**IMPORTANT CONSTRAINT: You MUST NOT request a call to any of the following tools again, as they have already been called in this conversation: {called_tools_str}.** If you need one of these tools again, you MUST generate a final text response instead.\n"
        
        # Add common JSON format instruction
        instruction_text += f"""\
This JSON object MUST be the *only* content in your response. **Do NOT include ANY other text, reasoning, or commentary before or after the JSON object.**
The object must have 'tool' (exact name) and 'parameters' (object mapping required params, and optional ones if extracted).
The 'parameters' object MUST only contain the actual parameters defined for the tool; do NOT include commentary or thought fields within it.
The JSON MUST be strictly valid. Ensure all string values within the parameters object are properly escaped (e.g., \" for quotes, \\ for backslashes).
Example: {{"tool": "some@tool", "parameters": {{"query": "some value with \\\"quotes\\\"\"}}}}
"""
        prompt += instruction_text
    else:
        # No tools available, must respond (Reverted)
        prompt += f"\nINSTRUCTIONS: Generate the final response for the user based on the conversation history (especially the LATEST USER QUESTION)."

    # Ensure the prompt suffix is just the label
    prompt += "\n\nAssistant response:"
    return prompt

# --- REMOVE OLD HELPER FUNCTIONS ---
# (We will remove generate_initial_prompt and generate_final_prompt later if confirmed)

# --- Existing parse_llm_response_for_tool_calls remains the same for now ---

def parse_llm_response_for_tool_calls(response_text: str, tools_definitions: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Attempts to parse the LLM response, prioritizing JSON object within markdown, then first raw object using raw_decode."""
    response_text = response_text.strip()
    cleaned_response_text = response_text.replace("</think>", "").strip()
    # print(f"Parser will attempt on cleaned text: '''{cleaned_response_text}'''") # DEBUG

    if not tools_definitions:
        print("Warning: No tool definitions provided for validation.")

    potential_call_object = None
    parsing_method = None

    # --- Strategy 1: Look for JSON object within markdown code block ---
    # print("Attempting Strategy 1: Find JSON in markdown...") # DEBUG
    extracted_json_str_md = None
    try:
        match = re.search(r'```json\\s*(\\{.*?\\})\\s*```', cleaned_response_text, re.DOTALL)
        if match:
            extracted_json_str_md = match.group(1)
            parsing_method = "markdown"
            # print(f"Extracted JSON object from markdown: {extracted_json_str_md}") # DEBUG
            # Try parsing this extracted string first
            try:
                fixed_json_str = extracted_json_str_md.replace('{{"', '{"')
                potential_call_object = json.loads(fixed_json_str)
                # print("Successfully parsed JSON from markdown.") # DEBUG
            except json.JSONDecodeError as md_parse_err:
                # print(f"Failed to parse JSON extracted from markdown ({md_parse_err}). Trying next strategy...") # DEBUG
                potential_call_object = None # Ensure reset if markdown parsing fails
                parsing_method = None        # Ensure reset
                extracted_json_str_md = None # Ensure reset
        # else:
             # print("Could not find ```json { ... } ``` markdown block. Trying next strategy...") # DEBUG
    except Exception as regex_err:
        print(f"ERROR during regex search for markdown JSON: {regex_err}. Trying next strategy...")

    # --- Strategy 2: If no valid object from markdown, use raw_decode on cleaned text ---
    if not potential_call_object:
        # print("Attempting Strategy 2: Use json.JSONDecoder().raw_decode()...") # DEBUG
        # Find the first opening brace to give raw_decode a starting point
        first_brace_index = cleaned_response_text.find('{')
        if first_brace_index != -1:
            try:
                decoder = json.JSONDecoder()
                # Decode starting from the first brace
                decoded_object, end_index = decoder.raw_decode(cleaned_response_text[first_brace_index:])
                potential_call_object = decoded_object # Use the directly decoded object
                parsing_method = "raw_decode"
                # print(f"Successfully decoded first JSON object using raw_decode (stopped at index {first_brace_index + end_index}).") # DEBUG
            except json.JSONDecodeError as raw_decode_err:
                # print(f"raw_decode failed: {raw_decode_err}") # DEBUG
                potential_call_object = None # Ensure reset
            except Exception as e:
                 print(f"Error during raw_decode processing: {e}")
                 potential_call_object = None # Ensure reset
        # else:
            # print("Could not find starting '{' for raw_decode.") # DEBUG

    # --- If no valid object found by either method, return None ---
    if not potential_call_object:
        # print("Could not extract and parse a valid JSON object via any method.") # DEBUG
        return None

    # --- Validate the successfully parsed object ---
    # print(f"Attempting to validate parsed object (method: {parsing_method}): {potential_call_object}") # DEBUG
    try:
        # Basic validation: is it a dictionary with required keys?
        if not (isinstance(potential_call_object, dict) and
                isinstance(potential_call_object.get('tool'), str) and
                isinstance(potential_call_object.get('parameters'), dict)):
            print(f"Parsed object is not a valid tool call structure.")
            return None

        # --- Validate required parameters ---
        tool_name = potential_call_object['tool']
        parameters = potential_call_object['parameters']
        call_is_valid = True
        if tools_definitions:
            tool_def = next((t for t in tools_definitions if t.get('name') == tool_name), None)
            if not tool_def:
                print(f"Warning: No definition found for tool '{tool_name}'. Cannot validate required parameters.")
            else:
                param_schema = tool_def.get('parameters') or tool_def.get('inputSchema')
                if param_schema and isinstance(param_schema, dict):
                    required_params = param_schema.get('required', [])
                    if required_params:
                        missing_params = [p for p in required_params if p not in parameters]
                        if missing_params:
                            print(f"Validation failed for tool '{tool_name}': Missing required parameters: {missing_params}")
                            call_is_valid = False
                else:
                     print(f"Warning: No parameter schema found in definition for tool '{tool_name}'. Cannot validate required parameters.")

        # --- Return result ---
        if call_is_valid:
            validated_calls_list = [potential_call_object] # Wrap the dict in a list
            # print(f"Successfully validated tool call (method: {parsing_method}): {validated_calls_list}") # DEBUG
            return validated_calls_list
        else:
            # Validation failed (missing required params)
            return None
    except Exception as e:
         # Catch errors during validation step
         print(f"Error during validation of parsed object (method: {parsing_method}, error: {e})")
         return None

# --- FastAPI App Initialization ---
app = FastAPI(title="LLM MCP-Style JSON-RPC Server", version="1.0.0")

# --- Model Loading Function ---
def load_model():
    """Loads the LLM model and tokenizer and assigns them to global variables."""
    # Make sure we modify the global variables
    global llm_pipeline, model, tokenizer 
    try:
        print(f"Attempting to load model: {MODEL_ID}")

        # Load the tokenizer
        print("Loading tokenizer...")
        # Assign to global tokenizer variable
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        print("Tokenizer loaded.")

        # Load the 4-bit quantized model
        print("Loading model (this may take time and significant RAM/VRAM)...")
        # Assign to global model variable
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True,
            # torch_dtype=torch.bfloat16 
        )
        model.eval() # Set model to evaluation mode
        print("Model loaded.")

        # Create the pipeline (for non-streaming endpoint)
        print("Creating text-generation pipeline...")
        llm_pipeline = pipeline(
            "text-generation",
            model=model,       # Use the loaded model
            tokenizer=tokenizer, # Use the loaded tokenizer
        )
        print(f"Pipeline for model {MODEL_ID} created successfully.")
        print(f"Global model and tokenizer also loaded for streaming.")

    except Exception as e:
        print(f"FATAL: Failed to load model {MODEL_ID}. Error: {e}")
        print("Please ensure the model ID/path is correct, necessary libraries are installed")
        print("('transformers', 'torch', 'accelerate', 'bitsandbytes'), and you have sufficient hardware resources.")
        raise SystemExit(f"Model loading failed: {e}")

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    """Load the model when the server starts."""
    print("Server starting up...")
    load_model()
    print("Model loading complete. Server is ready.")

# Remove or comment out the old /generate and /health endpoints
# @app.post("/generate", ...) ...
# @app.get("/health", ...) ...

# --- Single Endpoint for JSON-RPC ---
@app.post("/rpc")
async def json_rpc_endpoint(request: JsonRpcRequest) -> JsonRpcResponse:
    """Handles JSON-RPC 2.0 requests based on MCP concepts, now with multi-turn capability."""

    # --- Log Incoming Request ---
    # try:
    #     incoming_log = request.dict()
    #     print(f"\n--- Incoming JSON-RPC Request (ID: {request.id}) ---")
    #     print(json.dumps(incoming_log, indent=2, ensure_ascii=False))
    #     print("----------------------------------------------------\n")
    # except Exception as log_err:
    #     print(f"Error logging incoming request: {log_err}")
    # --- End Log Incoming Request ---

    response: JsonRpcResponse
    MAX_ITERATIONS = 30 # Define max iterations

    # --- Basic Validation ---
    if request.jsonrpc != "2.0":
        response = JsonRpcResponse(
            error=JsonRpcError(code=JsonRpcErrorCode.INVALID_REQUEST, message="Invalid JSON-RPC version"),
            id=request.id
        )
        # --- Log Outgoing Response ---
        # print(f"\n--- Outgoing JSON-RPC Response (ID: {request.id}) ---")
        # print(json.dumps(response.dict(), indent=2, ensure_ascii=False))
        # print("----------------------------------------------------\n")
        # --- End Log Outgoing Response ---
        return response

    # --- Handle 'tool_list' Method ---
    if request.method == "tool_list":
        # print(f"Handling 'tool_list' request (id: {request.id}) - Note: Tools should be provided in create_message") # DEBUG
        # Return empty list or a specific message
        tool_list_result = ToolListResult(tools=[])
        response = JsonRpcResponse(result=tool_list_result.dict(), id=request.id)

    # --- Handle 'create_message' Method (REWRITTEN FOR MULTI-TURN) ---
    elif request.method == "create_message":
        # print(f"Handling 'create_message' request (id: {request.id})") # DEBUG
        if llm_pipeline is None:
            response = JsonRpcResponse(
                error=JsonRpcError(code=JsonRpcErrorCode.INTERNAL_ERROR, message="Model pipeline is not available."),
                id=request.id
            )
            # Log and return...
            # print(f"\n--- Outgoing JSON-RPC Response (ID: {request.id}) ---") # DEBUG
            # print(json.dumps(response.dict(), indent=2, ensure_ascii=False)) # DEBUG
            # print("----------------------------------------------------\n") # DEBUG
            return response

        # --- Validate Params ---
        try:
            if request.params is None or not isinstance(request.params, dict):
                 raise ValueError("Params must be a JSON object for 'create_message' method")
            params = CreateMessageParams(**request.params)
        except Exception as e:
             response = JsonRpcResponse(
                error=JsonRpcError(code=JsonRpcErrorCode.INVALID_PARAMS, message=f"Invalid parameters for create_message: {e}"),
                id=request.id
            )
             # Log and return...
             # print(f"\n--- Outgoing JSON-RPC Response (ID: {request.id}) ---") # DEBUG
             # print(json.dumps(response.dict(), indent=2, ensure_ascii=False)) # DEBUG
             # print("----------------------------------------------------\n") # DEBUG
             return response

        # --- Core Multi-Turn Logic ---
        try:
            current_history: List[HistoryMessage] = []
            current_iteration = params.iteration_count # Keep for server-side loop prevention

            # 1. Initialize or Update History
            if params.history:
                current_history = params.history
            elif params.message:
                current_history = [HistoryMessage(role="user", content=params.message)]
            else:
                 raise ValueError("Request must contain either 'message' or 'history'")

            # --- REINSTATED: Calculation of already called tools --- 
            already_called_tools_before_llm = set()
            for msg in current_history:
                if msg.role == 'assistant' and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_name = tc.tool if isinstance(tc, ToolCallRequest) else tc.get('tool')
                        if tool_name:
                            already_called_tools_before_llm.add(tool_name)
            # --- END REINSTATED --- 

            # 2. Append Tool Results (if any) to History from PREVIOUS iteration
            if params.tool_results:
                # print(f"Received tool results for iteration {current_iteration} (id: {request.id})") # DEBUG
                # --- Normalize tool names from results before adding to history ---
                prefix = "time@" # Example prefix, adjust if needed for other tools
                for tr_input in params.tool_results:
                     # Add normalized name back if needed (depends on tool implementation)
                    if tr_input.tool_name == "get_current_time":
                        normalized_name = f"{prefix}{tr_input.tool_name}"
                        # print(f"Normalizing tool name in result: '{tr_input.tool_name}' -> '{normalized_name}'") # DEBUG
                    else:
                         normalized_name = tr_input.tool_name # Use as is

                    current_history.append(HistoryMessage(
                        role="tool",
                        tool_name=normalized_name, # Use potentially normalized name
                        tool_result=tr_input.result,
                        tool_error=tr_input.error
                    ))
                # --- End Normalization ---
                current_iteration += 1 # Still useful for server-side MAX_ITERATIONS check
                # print(f"Appended tool results to history. New iteration count: {current_iteration} (id: {request.id})") # DEBUG

            # 3. Check Iteration Limit (Server-side safety)
            force_final_response = False
            if current_iteration >= MAX_ITERATIONS:
                print(f"Maximum iteration limit ({MAX_ITERATIONS}) reached. Forcing final response. (id: {request.id})")
                force_final_response = True

            # 4. Generate Prompt and Call LLM
            # --- REVERTED: Call generate_conversation_prompt WITH iteration counts ---
            llm_input_prompt = generate_conversation_prompt(
                history=current_history, 
                tools=params.tools,
                iteration_count=current_iteration,
                max_iterations=MAX_ITERATIONS
            )
            # --- END REVERTED ---
            
            # print(f"Generating LLM response (iteration: {current_iteration}, force_final: {force_final_response}, id: {request.id})...") 
            results = llm_pipeline(llm_input_prompt, max_new_tokens=params.max_new_tokens, do_sample=params.do_sample)
            print(f"LLM generation complete (iteration: {current_iteration}, id: {request.id})") 

            # 5. Process LLM Response
            llm_response_text = "" # Default empty
            raw_llm_output = ""
            if results and isinstance(results, list) and 'generated_text' in results[0]:
                raw_llm_output = results[0]['generated_text']
                # print(f"Raw LLM Output (iteration: {current_iteration}) (id: {request.id}):\\n>>>\\n{raw_llm_output}\\n<<<") # DEBUG
                # --- Simple Cleaning: Remove prompt if present ---
                # More robust cleaning might be needed depending on the model
                if raw_llm_output.startswith(llm_input_prompt):
                    llm_response_text = raw_llm_output[len(llm_input_prompt):].strip()
                else:
                     llm_response_text = raw_llm_output.strip() # Fallback
                     # print(f"Warning: Input prompt not detected at start of LLM output. Using raw output. (id: {request.id}) ") # DEBUG
                # print(f"Cleaned LLM Response Text (before tool parse) (id: {request.id}): {llm_response_text}") # DEBUG
                # --- End Cleaning ---
            else:
                print(f"Error: Unexpected LLM pipeline output format. (id: {request.id})") # Keep this
                # Handle error, maybe return an internal error response
                raise ValueError("LLM pipeline returned unexpected output")

            # --- REINSTATED: Server-side check for repeated tool calls --- 
            tool_call_blocked = False
            if current_iteration > 0 and not force_final_response:
                potential_tool_match = re.search(r'"tool":\s*"([^"{}]+)"', llm_response_text) # Simplified regex
                if potential_tool_match:
                    potential_tool_name = potential_tool_match.group(1)
                    if potential_tool_name in already_called_tools_before_llm:
                        print(f"BLOCKING repeated tool call request for '{potential_tool_name}' (id: {request.id}) based on history.")
                        tool_call_blocked = True
            # --- END REINSTATED --- 
            
            # 6. Attempt to Parse Tool Calls (unless forcing final response OR blocked)
            tool_calls = None
            # --- REVERTED: Add tool_call_blocked back to condition --- 
            if not force_final_response and not tool_call_blocked:
                 tool_calls = parse_llm_response_for_tool_calls(llm_response_text, params.tools)
            # --- END REVERTED ---

            # 7. Determine Response Type and Build Response
            # --- REVERTED: Add logging check for tool_call_blocked --- 
            if tool_calls: 
                # --- Tool Call Path ---
                print(f"Requesting tool calls for next iteration (id: {request.id}): {tool_calls}")
                current_history.append(HistoryMessage(role="assistant", tool_calls=tool_calls))
                result_content = CreateMessageResult(
                    type="tool_calls",
                    content=tool_calls,
                    history=current_history,
                    iteration_count=current_iteration 
                )
                response = JsonRpcResponse(result=result_content.model_dump(), id=request.id)
            else:
                # --- Final Text Path (Includes blocked calls) ---
                if tool_call_blocked:
                    print(f"Proceeding with final text response due to BLOCKED repeated tool call (id: {request.id})")
                elif force_final_response:
                    pass 
                else:
                     print(f"Proceeding with final text response (no valid/allowed tool call) (id: {request.id})")

                # --- ADDED CLEANING for final_text: Remove </think> and preceding content ---
                final_text_content = llm_response_text # Start with potentially unclean text
                think_tag_marker = "</think>"
                think_tag_index = final_text_content.find(think_tag_marker)
                if think_tag_index != -1:
                    # Take text *after* the tag, strip leading whitespace/newlines
                    cleaned_final_text = final_text_content[think_tag_index + len(think_tag_marker):].lstrip()
                    # print(f"Original final text had </think>. Cleaned: \\"'{cleaned_final_text}'\\"") # DEBUG
                    final_text_content = cleaned_final_text
                # --- END CLEANING ---

                # print(f"Generating final text response (id: {request.id})") # Keep this - Redundant with above logs?
                # Add assistant's final response to history (using cleaned text)
                current_history.append(HistoryMessage(role="assistant", content=final_text_content))
                result_content = CreateMessageResult(
                    type="final_text",
                    content=final_text_content,
                    history=current_history,
                    iteration_count=current_iteration
                )
                response = JsonRpcResponse(result=result_content.model_dump(), id=request.id)
            # --- END REVERTED ---

        except Exception as e:
             print(f"Error processing create_message (id: {request.id}): {e}")
             response = JsonRpcResponse(
                 error=JsonRpcError(code=JsonRpcErrorCode.INTERNAL_ERROR, message=f"Error processing create_message: {e}"),
                 id=request.id
             )
        # --- End Core Multi-Turn Logic ---

    # --- Handle Unknown Method ---
    else:
        print(f"Method not found (id: {request.id}): {request.method}") # Keep this
        response = JsonRpcResponse(
            error=JsonRpcError(code=JsonRpcErrorCode.METHOD_NOT_FOUND, message=f"Method '{request.method}' not found"),
            id=request.id
        )

    # --- Log Outgoing Response ---
    # try:
    #     outgoing_log = response.model_dump(exclude_none=True) # Exclude none to keep log cleaner
    #     print(f"\n--- Outgoing JSON-RPC Response (ID: {request.id}) ---")
    #     print(json.dumps(outgoing_log, indent=2, ensure_ascii=False))
    # except Exception as log_err:
    #      print(f"Error logging outgoing response: {log_err}") # Keep this
    # --- End Log Outgoing Response ---

    return response

# --- Streaming Logic ---
async def stream_generation(prompt: str, streamer: TextIteratorStreamer, params: CreateMessageParams):
    """Runs model generation in a separate thread and yields tokens."""
    # Prepare inputs
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Generation arguments - Corrected using dictionary unpacking
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": params.max_new_tokens,
        "do_sample": params.do_sample,
        # Add other relevant generation parameters if needed (e.g., temperature, top_p)
    }

    # Run generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Yield generated tokens
    try:
        for new_text in streamer:
            if new_text: # Ensure not yielding empty strings
                # Format as SSE message: data: {...}\n\n
                yield f"data: {json.dumps({'delta': new_text})}\n\n"
    except Exception as e:
        print(f"Error during streaming: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        # Ensure thread is joined
        thread.join()
        print("Streaming generation thread finished.")

# --- Streaming Endpoint ---
@app.post("/stream")
async def stream_endpoint(params: CreateMessageParams):
    """Handles requests for streamed LLM responses."""
    global model, tokenizer # Ensure access to global model/tokenizer

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded")

    # --- Simplified History Handling for Streaming (Client manages full state) ---
    current_history: List[HistoryMessage] = []
    if params.history:
        current_history = params.history
        # print(f"Stream request using provided history (ID: N/A - Streaming)") # DEBUG
    elif params.message:
        current_history = [HistoryMessage(role="user", content=params.message)]
        # print(f"Stream request starting new history from message (ID: N/A - Streaming)") # DEBUG
    else:
         raise HTTPException(status_code=400, detail="Stream request must contain either 'message' or 'history'")

    # Note: Tool results are NOT processed server-side in this stream endpoint
    # The client should incorporate them into the history before sending the request

    # --- Generate Prompt ---
    # We don't track iterations server-side for streaming endpoint
    prompt = generate_conversation_prompt(
        history=current_history,
        tools=params.tools
        # iteration_count and max_iterations are omitted for pure streaming
    )
    # print(f"Generated prompt for streaming:\\n{prompt}") # DEBUG

    # --- Setup Streamer ---
    # skip_prompt=True avoids yielding the input prompt itself
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # --- Return Streaming Response ---
    # Pass streamer and params to the generator function
    return StreamingResponse(
        stream_generation(prompt, streamer, params),
        media_type="text/event-stream"
    )

# --- Main Execution ---
if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    # Make sure the MODEL_ID is set correctly above (either path or HF ID)
    print(f"Starting Uvicorn server for model: {MODEL_ID}")
    # Run the FastAPI app using Uvicorn
    # Listen on all available IPs (0.0.0.0) on port 8000
    # You can change the host and port as needed.
    uvicorn.run(app, host="0.0.0.0", port=8000)