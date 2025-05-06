import logging
import sys
import os # Added for environment variables
from typing import Optional
# from transformers import Pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig # Removed
# import torch # Removed
from logging.handlers import RotatingFileHandler
from langchain_core.language_models.chat_models import BaseChatModel # Generic Langchain chat model
from langchain_openai import ChatOpenAI # Use ChatOpenAI
# from langchain_anthropic import ChatAnthropic # Removed
# from langchain_deepseek import ChatDeepSeek # Removed

# --- Model Configuration ---
# MODEL_ID = "./Qwen-14B" # Removed
# --- End Model Configuration ---

# --- Logging Setup --- 
LOG_FILE = "inference_server.log"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5

def setup_logging():
    # Configure root logger
    # Set level to DEBUG to capture info, debug messages
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s - %(levelname)s - [%(process)d] %(message)s', # Include process ID
        handlers=[
            RotatingFileHandler(LOG_FILE, maxBytes=MAX_LOG_SIZE, backupCount=LOG_BACKUP_COUNT, encoding='utf-8'), # Rotating file handler with encoding
            logging.StreamHandler(sys.stdout)        # Also log to console
        ]
    )
    logging.info("Logging configured: output to rotating file (UTF-8) and stdout.")

# Define shared state variables here
# llm_pipeline: Optional[Pipeline] = None # Removed

# --- Add Global Model/Tokenizer Variables --- 
# model = None # Removed
# tokenizer = None # Removed
llm: Optional[BaseChatModel] = None # Type changed to BaseChatModel
# --- End Add Globals ---

# --- Remove load_model function ---

# --- Add initialize_llm function ---
def _clean_model_name(name: str) -> str:
    """Strips leading/trailing whitespace, quotes (single or double), and comments from a model name."""
    name = name.strip()
    # Remove comments first
    name = name.split('#')[0].strip()
    
    # Iteratively strip quotes
    while (name.startswith('"') and name.endswith('"')) or \
          (name.startswith("'") and name.endswith("'")):
        if not name:
            break # Should not happen if we stripped correctly
        name = name[1:-1].strip() # Strip one layer and internal whitespace revealed
    return name

# Simplified initialize_llm using ChatOpenAI for compatible APIs
def initialize_llm(
    default_model: str = "deepseek-chat", 
    default_api_key_env_var: str = "DEEPSEEK_API_KEY", # Env var for the API key
    default_base_url: str = "https://api.deepseek.com/v1", # Default to DeepSeek endpoint
    temperature: float = 0.7
):
    """Initializes ChatOpenAI client for an OpenAI-compatible API (like DeepSeek)."""
    global llm

    model_name_from_env = os.getenv("LLM_MODEL_NAME")
    # Allow overriding API key env var name (e.g., could use OPENAI_API_KEY for actual OpenAI)
    api_key_env_var = os.getenv("API_KEY_ENV_VAR", default_api_key_env_var)
    # Allow overriding base URL (e.g., for OpenAI default or other compatible APIs)
    base_url_from_env = os.getenv("LLM_BASE_URL", default_base_url)
    
    # Determine the actual model name to use
    specific_model_name = model_name_from_env or default_model
    cleaned_model_name = _clean_model_name(specific_model_name)

    api_key = os.getenv(api_key_env_var)

    if not api_key:
        logging.error(f"{api_key_env_var} not found in environment variables. LLM initialization failed.")
        return
    
    if not base_url_from_env:
         logging.error(f"LLM Base URL is not set (checked LLM_BASE_URL env var and default). LLM initialization failed.")
         return

    try:
        logging.info(f"Attempting to initialize ChatOpenAI client")
        logging.info(f"Target API Base URL: '{base_url_from_env}'")
        logging.info(f"Using API Key from env var: '{api_key_env_var}'")
        logging.info(f"Initializing model: '{cleaned_model_name}' with temperature: {temperature}")
        
        llm = ChatOpenAI(
            model=cleaned_model_name,
            temperature=temperature,
            streaming=True,
            openai_api_key=api_key, # Pass the key explicitly
            base_url=base_url_from_env # Set the target API endpoint
        )

        logging.info(f"LLM (ChatOpenAI targeting '{base_url_from_env}' with model '{cleaned_model_name}') initialized and ready.")
    except Exception as e:
        logging.error(f"Failed to initialize ChatOpenAI for URL {base_url_from_env} with model {cleaned_model_name}: {e}")
        logging.error(f"Make sure '{api_key_env_var}' and model name are correct for the target API.")
        llm = None
# --- End Add initialize_llm function ---

# Ensure logging is set up when the module is imported (or call it explicitly in main app setup)
# setup_logging() # This is called in server_starlette.py