import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from transformers import Pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from logging.handlers import RotatingFileHandler

# Load .env from project root so INFERENCE_* vars apply without exporting
load_dotenv(Path(__file__).resolve().parent / ".env")

# --- Model Configuration (env overrides) ---
# INFERENCE_MODEL_ID: Hugging Face model id or local path (e.g. ./Qwen-14B or Qwen/Qwen2.5-7B-Instruct)
# INFERENCE_ATTN_IMPLEMENTATION: sdpa (built-in, no extra dep) | flash_attention_2 (requires flash-attn, faster)
# INFERENCE_BNB_COMPUTE_DTYPE: float16 | bfloat16 (for 4-bit quant compute dtype)
MODEL_ID: str = os.environ.get("INFERENCE_MODEL_ID", "./Qwen-14B")
ATTN_IMPLEMENTATION: str = os.environ.get("INFERENCE_ATTN_IMPLEMENTATION", "sdpa")
BNB_COMPUTE_DTYPE_STR: str = os.environ.get("INFERENCE_BNB_COMPUTE_DTYPE", "float16")
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

def _get_bnb_compute_dtype():
    if BNB_COMPUTE_DTYPE_STR.lower() == "bfloat16":
        return torch.bfloat16
    return torch.float16


# --- Add load_model function --- 
def load_model():
    """Loads the LLM model and tokenizer into the shared module."""
    global model, tokenizer
    if ATTN_IMPLEMENTATION not in ("sdpa", "flash_attention_2", "eager"):
        raise ValueError(
            f"INFERENCE_ATTN_IMPLEMENTATION must be one of sdpa, flash_attention_2, eager; got {ATTN_IMPLEMENTATION!r}"
        )
    try:
        logging.info(f"Attempting to load model: {MODEL_ID} (attn={ATTN_IMPLEMENTATION}, bnb_dtype={BNB_COMPUTE_DTYPE_STR})")
        logging.debug("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        logging.info("Tokenizer loaded.")

        bnb_dtype = _get_bnb_compute_dtype()
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=bnb_dtype,
        )

        logging.debug("Loading model (this may take time and RAM/VRAM)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",
            attn_implementation=ATTN_IMPLEMENTATION,
            trust_remote_code=True,
        )
        model.eval()
        logging.info("Model loaded and ready in shared module.")

    except Exception as e:
        logging.exception(f"FATAL: Failed to load model {MODEL_ID}. Error: {e}")
        raise SystemExit(f"Model loading failed: {e}")
# --- End Add load_model function --- 