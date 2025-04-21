import logging
import sys
from typing import Optional
from transformers import Pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from logging.handlers import RotatingFileHandler

# --- Model Configuration ---
# Choose how to specify the model:
# Option 1: Use a model ID (requires download on first run if not cached)
# MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" # Comment this out
# Option 2: Use a local path (if downloaded with --local-dir)
MODEL_ID = "./Qwen-14B" # Uncomment this line and ensure the path is correct
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
# Initialize them to None
llm_pipeline: Optional[Pipeline] = None 

# --- Add Global Model/Tokenizer Variables --- 
model = None
tokenizer = None
# --- End Add Globals ---

# --- Add load_model function --- 
def load_model():
    """Loads the LLM model and tokenizer into the shared module."""
    global model, tokenizer 
    try:
        logging.info(f"Attempting to load model: {MODEL_ID}")
        logging.debug("Loading tokenizer...")
        # Ensure trust_remote_code=True if needed by the model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        logging.info("Tokenizer loaded.")

        # Create BitsAndBytes configuration for 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16 # Specify compute dtype here
        )

        logging.debug("Loading model (this may take time and RAM/VRAM)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config, # Use quantization_config instead
            device_map="auto", # Use auto device mapping
            # max_memory={0: "12GiB"}, # Optional: specify memory limits if needed
            trust_remote_code=True, # Ensure trust_remote_code=True if needed
        )
        model.eval() # Set model to evaluation mode
        logging.info("Model loaded and ready in shared module.")

    except Exception as e:
        logging.exception(f"FATAL: Failed to load model {MODEL_ID}. Error: {e}")
        # Exit if model loading fails, as the server cannot function
        raise SystemExit(f"Model loading failed: {e}")
# --- End Add load_model function --- 