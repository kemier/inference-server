from pydantic import BaseModel, Field
from typing import Any, Optional, Union, Dict, List

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
# Define ToolCallRequest first as it's used by notification params
class ToolCallRequest(BaseModel): # Model for server requesting client to call a tool
    tool: str
    parameters: Dict[str, Any]

    class Config:
        populate_by_name = True
        extra = "ignore"

# --- WebSocket Notification Parameter Models ---

class BaseNotificationParams(BaseModel):
    """Base model for common fields in WebSocket notification parameters."""
    session_id: str
    task_id: str

class TextChunkParams(BaseNotificationParams):
    """Params for 'text_chunk' notification."""
    content: str

class FunctionCallRequestParams(BaseNotificationParams):
    """Params for 'function_call_request' notification."""
    tool_call: ToolCallRequest # Reusing the existing ToolCallRequest model

class ErrorNotificationParams(BaseNotificationParams):
    """Params for 'error' notification."""
    error: JsonRpcError # Reusing the existing JsonRpcError model

class StreamEndParams(BaseNotificationParams):
    """Params for 'stream_end' notification."""
    final_text: Optional[str] = None # The final aggregated text, if applicable

# --- Message History Model ---
class HistoryMessage(BaseModel):
    role: str # 'user', 'assistant', 'tool'
    content: Optional[Union[str, List[Dict[str, Any]]]] = None # Text for user/assistant, tool calls for assistant, text result for tool
    tool_calls: Optional[List[ToolCallRequest]] = None # LLM requested tool calls
    tool_name: Optional[str] = None # Name of the tool that was called (for role='tool')
    tool_result: Optional[Any] = None # Result from tool execution (for role='tool')
    tool_error: Optional[str] = None # Error from tool execution (for role='tool')

# --- Method Specific Params/Result Models ---

# Model for client providing tool execution result (used in GenerateRequestParams)
class ToolResultInput(BaseModel):
    tool_name: str = Field(..., alias="toolName") # Use ... for required field
    result: Any
    error: Optional[str] = None # Added error field for tool execution errors

    class Config:
        populate_by_name = True
        extra = "ignore"

# Params for the initial 'generate' WebSocket request
class GenerateRequestParams(BaseModel): # Renamed from CreateMessageParams
    # If history is provided, message is ignored (or treated as the latest user message)
    message: Optional[str] = None
    history: Optional[List[HistoryMessage]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[ToolResultInput]] = None # Results for calls requested in *previous* turn
    iteration_count: int = 0 # Track iteration depth
    max_new_tokens: int = 4096 # <-- Increased default value significantly
    do_sample: bool = True
    temperature: Optional[float] = Field(None, ge=0.01, le=2.0) # Add temperature with validation (0.01 to 2.0)

# Define the structure for a single result item within the list
class ToolResultItem(BaseModel):
    tool_call_id: str # ID provided by server in function_call_request (or client generated)
    tool_name: str    # Name of the tool that was executed
    result: Optional[Any] = None # Result from the tool execution
    isError: bool = False # Flag indicating if the result is an error
    error_message: Optional[str] = None # Error message if isError is True

# Params for 'tool_result' method (from client)
# This now correctly expects a list of results and inherits session/task IDs
class ToolResultParams(BaseNotificationParams):
    results: List[ToolResultItem]

# --- HTTP Endpoint Models ---

# Params for POST /create_session
class CreateSessionParams(BaseModel):
    tools: Optional[List[Dict[str, Any]]] = None

# Response for POST /create_session
class CreateSessionResponse(BaseModel):
    session_id: str

# --- WebSocket Specific Models ---

# (No specific params model needed for 'tool_list')
# Removed ToolListResult as tool discovery is assumed to be handled separately 