import json
import re
import logging
from typing import Any, Optional, Union, Dict, List

# Import necessary Pydantic models from the new models.py file
from models import HistoryMessage, ToolCallRequest 


def generate_conversation_prompt(history: List[HistoryMessage], tools: Optional[List[Dict[str, Any]]] = None, iteration_count: int = 0, max_iterations: int = 30) -> str:
    """Generates the prompt for the LLM based on conversation history and available tools."""
    
    # --- Simplified History Formatting --- 
    prompt_parts = []
    prompt_parts.append("This is a conversation between a user and an assistant. The assistant may use tools.")
    prompt_parts.append("\nCONVERSATION HISTORY:")

    latest_user_question_index = -1
    for i, msg in reversed(list(enumerate(history))):
        if msg.role == 'user':
            latest_user_question_index = i
            break

    for idx, msg in enumerate(history):
        turn = ""
        if msg.role == 'user':
            # Apply special prefix ONLY if it's the very last user message identified
            if idx == latest_user_question_index:
                turn += "\nLATEST USER QUESTION: USER: "
            else:
                turn += "\nUSER: "
            turn += msg.content if msg.content else ""
        elif msg.role == 'assistant':
            turn += "\nASSISTANT:"
            if msg.content:
                turn += f" {msg.content}"
            if msg.tool_calls:
                try:
                    # Keep tool calls concise
                    calls_str = json.dumps([tc.model_dump() for tc in msg.tool_calls])
                    turn += f" [Requested Tool Calls: {calls_str}]"
                except Exception as dump_err:
                    logging.error(f"ERROR serializing tool_calls in history: {dump_err}")
                    turn += " [Error serializing tool calls]"
        elif msg.role == 'tool':
            # Format tool results clearly but concisely
            turn += f"\nTOOL_RESULT (for {msg.tool_name}):"
            if msg.tool_error:
                turn += f" Error: {msg.tool_error}"
            else:
                try:
                    result_str = json.dumps(msg.tool_result, separators=(',', ':'))
                except TypeError:
                    result_str = str(msg.tool_result) # Fallback for non-serializable results
                turn += f" Result: {result_str}"
        
        if turn: # Avoid adding empty turns
            prompt_parts.append(turn)
            
    prompt_parts.append("\nEND HISTORY.")
    # --- End Simplified History Formatting --- 

    prompt_parts.append("\nCURRENT TASK:")
    already_called_tools = set() # Recalculate based on history
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
            already_called_tools.add(tool_name) # Assume all listed tools might have been called if history unclear
        # Add tool descriptions to prompt_parts
        prompt_parts.append(f"You have access to the following tools:\n{formatted_tools}\n")
    else:
        prompt_parts.append("You do not have access to any tools for this request.\n")

    # --- Instructions --- (Keep the refined instructions from the previous step)
    if iteration_count >= max_iterations:
        prompt_parts.append(f"""\

INSTRUCTIONS: You have reached the maximum tool iteration limit ({max_iterations}).
Focus EXCLUSIVELY on the LATEST USER QUESTION.
Generate a final response for the user based *only* on the information available in the conversation history that is relevant to answering THIS specific question.
**Do not request any more tool calls.**
Ignore any limitations stated in previous assistant turns unless they directly prevent answering the *current* user request.
If the history does not contain enough information to directly answer the LATEST USER QUESTION, state that clearly.""" )
    elif tools:
        # ...(Keep the complex instruction logic for tool use / history response generation)
        instruction_text = f"""\
INSTRUCTIONS: Based on the **entire conversation history** (especially the LATEST USER QUESTION) and the available tools (including their parameters), **carefully consider the user's overall goal and decide the next step.**
**First, review the 'Tool (...) Result:' entries in the history. Does the information needed to answer the user's latest request *already exist* there?**
**Your primary goal is to answer the user directly using information from the history if possible.** Only consider using a tool if the answer cannot be found in the conversation history and the tool is *essential* to fulfill the user's request.

1.  If the necessary information **is present in the history's tool results OR can be directly answered using only the conversation history**, generate the final response text for the user. **Do NOT call a tool if the answer is already available or derivable from the history.**
2.  If, **and only if**, the information is *not* in the history and you *absolutely need* external information or an action performed using a tool, choose the **single most relevant tool**. Then, respond with *only* a **single JSON object** containing the tool call request."""

        if already_called_tools:
            called_tools_str = ", ".join(f'`{t}`' for t in sorted(list(already_called_tools)))
            instruction_text += f"""\

**IMPORTANT CONSTRAINT: You MUST NOT request a call to any of the following tools again, as they have already been called in this conversation: {called_tools_str}.** If you need one of these tools again, you MUST generate a final text response instead based *only* on the history. If the history is insufficient, state that."""
        
        instruction_text += f"""\

**If making a tool call:**
The JSON object MUST be the *only* content in your response. **Do NOT include ANY other text, reasoning, or commentary before or after the JSON object.**
The object must have 'tool' (exact name) and 'parameters' (object mapping required params, and optional ones if extracted).
The 'parameters' object MUST only contain the actual parameters defined for the tool; do NOT include commentary or thought fields within it.
The JSON MUST be strictly valid. Ensure all string values within the parameters object are properly escaped (e.g., \" for quotes, \\ for backslashes).
Example: {{"tool": "some@tool", "parameters": {{"query": "some value with \\\"quotes\\\""}}}}"""

        # --- Add specific instruction for fulfilling stated intentions --- 
        instruction_text += f"""\

**If generating a final text response (because the answer is in history or no suitable tool call is needed/allowed):**
Focus EXCLUSIVELY on the LATEST USER QUESTION.
Generate your response based *only* on the information available in the conversation history that is relevant to answering THIS specific question.
*However*, if the PREVIOUS assistant turn stated an intention to provide specific information (like a list, explanation, or recommendation) AND the LATEST USER QUESTION is a simple confirmation or request to proceed (e.g., 'Okay', 'Go ahead', '来吧', 'Yes', '继续'), then fulfill that previously stated intention NOW in your response. Use the conversation history for context if needed.
Otherwise (if not fulfilling a prior intention), ignore any limitations stated in previous assistant turns unless they directly prevent answering the *current* user request. Do not make up information.
If the history does not contain enough information to directly answer the LATEST USER QUESTION (and you weren't fulfilling a prior intention), clearly state that you cannot answer the specific question based on the provided conversation context."""
        prompt_parts.append(instruction_text)
    else:
        prompt_parts.append(f"""\

INSTRUCTIONS: Generate the final response for the user.
Focus EXCLUSIVELY on the LATEST USER QUESTION.
Generate your response based *only* on the information available in your knoledge base that is relevant to answering THIS specific question.
*However*, if the PREVIOUS assistant turn stated an intention to provide specific information (like a list, explanation, or recommendation) AND the LATEST USER QUESTION is a simple confirmation or request to proceed (e.g., 'Okay', 'Go ahead', '来吧', 'Yes', '继续'), then fulfill that previously stated intention NOW in your response. Use the conversation history for context if needed.
Otherwise (if not fulfilling a prior intention), ignore any limitations stated in previous assistant turns unless they directly prevent answering the *current* user request. Do not make up information.
If the history does not contain enough information to directly answer the LATEST USER QUESTION (and you weren't fulfilling a prior intention), clearly state that you cannot answer the specific question based on the provided conversation context.""" )
    # --- End Instructions --- 

    prompt_parts.append("\n\nAssistant response:")
    
    final_prompt = "\n".join(prompt_parts)
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