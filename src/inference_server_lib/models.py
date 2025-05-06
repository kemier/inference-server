from pydantic import BaseModel, Field, model_validator
from typing import Any, Optional, Union, Dict, List, Literal

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

# --- Tool Related Models (Moved Up) ---

# Model for a tool call FROM the LLM/Assistant (OpenAI format)
# Needs to be defined before HistoryMessage
class OpenAIFunctionSpec(BaseModel):
    name: str
    arguments: str # JSON string of arguments

class OpenAIAssistantToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: OpenAIFunctionSpec

# Model for a tool call request TO the client (MCP)
class ToolCallRequest(BaseModel): # Model for server requesting client to call a tool
    id: str # This is the tool_call_id from the LLM, passed through
    tool: str # Original (unsanitized) tool name for the client
    parameters: Dict[str, Any]

    class Config:
        populate_by_name = True
        extra = "ignore"

# Model for an individual tool result item FROM the client (MCP)
# Needs to be defined before ToolResultParams
class ToolResultItem(BaseModel):
    tool_call_id: str # ID of the tool call this result corresponds to
    tool_name: str # Original (unsanitized) name of the tool that was called
    result: Optional[Any] = None
    error: Optional[str] = None

# --- WebSocket Notification Parameter Models --- 

class BaseNotificationParams(BaseModel):
    """Base model for common fields in WebSocket notification parameters."""
    session_id: str
    task_id: str

class TextChunkParams(BaseNotificationParams):
    """Params for 'text_chunk' notification."""
    content: str

class FunctionCallRequestParams(BaseModel):
    """Params for 'function_call_request' notification."""
    session_id: str
    task_id: str
    tool_call: ToolCallRequest # Contains the actual tool, params, and id for client

class ErrorNotificationParams(BaseNotificationParams):
    """Params for 'error' notification."""
    error: JsonRpcError # Reusing the existing JsonRpcError model

class StreamEndParams(BaseNotificationParams):
    """Params for 'stream_end' notification."""
    final_text: Optional[str] = None # The final aggregated text, if applicable
    status_message: Optional[str] = None # e.g., "Completed", "Function call requested", "Error"

# --- Message History Model (Now after its dependencies) --- 
class HistoryMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[List[OpenAIAssistantToolCall]] = None # USES OpenAIAssistantToolCall
    
    # Fields specific to role 'tool'
    tool_call_id: Optional[str] = None # ID of the tool call that this is a result for
    tool_name: Optional[str] = None    # Original (unsanitized) name of the tool (for 'tool' role messages)
    tool_result: Optional[Any] = None  # For 'tool' role if successful
    tool_error: Optional[str] = None   # For 'tool' role if error occurred

    class Config:
        extra = 'ignore' # Or 'forbid' if you want to be strict

    @model_validator(mode='after')
    def check_tool_fields_are_present_if_role_is_tool(cls, values: Any) -> Any:
        # Ensure 'values' is treated as a dict-like object, Pydantic v2 passes the model instance
        # For Pydantic v2, accessing fields can be done via values.fieldname or getattr(values, 'fieldname')
        # If 'values' is the model instance itself:
        role = values.role
        tool_call_id = values.tool_call_id
        tool_name = values.tool_name

        if role == "tool":
            if not tool_call_id or not isinstance(tool_call_id, str) or not tool_call_id.strip():
                raise ValueError("If role is 'tool', 'tool_call_id' must be a non-empty string.")
            if not tool_name or not isinstance(tool_name, str) or not tool_name.strip():
                # While tool_name is not strictly required by OpenAI for the ToolMessage itself (it uses id),
                # our internal HistoryMessage for role 'tool' should probably have it for consistency
                # and because our generate_langchain_messages uses it for ToolMessage(name=...).
                raise ValueError("If role is 'tool', 'tool_name' must be a non-empty string.")
        return values

# --- Method Specific Params/Result Models --- 

# Model for client providing tool execution result (used in GenerateRequestParams)
# This model seems unused by GenerateRequestParams currently. Review if needed or can be removed.
class ToolResultInput(BaseModel):
    tool_name: str = Field(..., alias="toolName") # Use ... for required field
    result: Any
    error: Optional[str] = None # Added error field for tool execution errors

    class Config:
        populate_by_name = True
        extra = "ignore"

# Params for the initial 'generate' WebSocket request
class GenerateRequestParams(BaseModel): # Renamed from CreateMessageParams
    history: List[HistoryMessage] # USES HistoryMessage
    tools: Optional[List[Dict[str, Any]]] = None # Tools available for this turn (OpenAI format)
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None # ADDED: OpenAI tool_choice parameter
    max_new_tokens: Optional[int] = Field(default=4096, alias="max_tokens")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    iteration_count: int = 0

# Params for 'tool_result' method (from client)
class ToolResultParams(BaseModel):
    session_id: str
    task_id: str
    results: List[ToolResultItem] # USES ToolResultItem

# --- HTTP Endpoint Models --- 

# Params for POST /create_session
class CreateSessionParams(BaseModel):
    tools: Optional[List[Dict[str, Any]]] = None

# Response for POST /create_session
class CreateSessionResponse(BaseModel):
    session_id: str

# (No specific params model needed for 'tool_list')
# Removed ToolListResult as tool discovery is assumed to be handled separately

# --- WebSocket Specific Models ---

# (No specific params model needed for 'tool_list')
# Removed ToolListResult as tool discovery is assumed to be handled separately 