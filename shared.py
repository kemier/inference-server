import logging
import sys
from typing import Optional
from transformers import Pipeline

# --- Model Configuration ---
# Choose how to specify the model:
# Option 1: Use a model ID (requires download on first run if not cached)
# MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" # Comment this out
# Option 2: Use a local path (if downloaded with --local-dir)
MODEL_ID = "./Qwen-14B" # Uncomment this line and ensure the path is correct
# --- End Model Configuration ---

# --- Logging Setup --- 
LOG_FILE = "inference_server.log"

def setup_logging():
    # Configure root logger
    # Set level to DEBUG to capture info, debug messages
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s - %(levelname)s - [%(process)d] %(message)s', # Include process ID
        handlers=[
            logging.FileHandler(LOG_FILE, mode='a'), # Append to file
            logging.StreamHandler(sys.stdout)        # Also log to console
        ]
    )
    logging.info("Logging configured: output to file and stdout.")
# --- End Logging Setup ---


# Define shared state variables here
# Initialize them to None
llm_pipeline: Optional[Pipeline] = None 