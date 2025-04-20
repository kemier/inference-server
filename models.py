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
    temperature: Optional[float] = Field(None, ge=0.01, le=2.0) # Add temperature with validation (0.01 to 2.0)

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