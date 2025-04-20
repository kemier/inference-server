from typing import Any, Optional, Union, Dict, List
import logging
import traceback
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os

# Import the shared state module
import shared
# Import and run the logging setup from SHARED module
from shared import setup_logging, MODEL_ID

# Import the Celery app instance
from celery_app import app as celery_app

setup_logging() # Logging setup remains important

# --- Worker Initialization --- 
def initialize_worker_pipeline():
    """Loads model and pipeline if not already loaded in this worker process."""
    if shared.llm_pipeline is None:
        logging.info(f"[Worker {os.getpid()}] Initializing LLM pipeline for the first time...")
        try:
            logging.debug(f"[Worker {os.getpid()}] Loading tokenizer for {MODEL_ID}...")
            # Note: We don't need the global tokenizer/model in the worker, only the pipeline
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
            logging.info(f"[Worker {os.getpid()}] Tokenizer loaded.")

            logging.debug(f"[Worker {os.getpid()}] Loading model {MODEL_ID}... (worker)")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                load_in_4bit=True,
                device_map="auto",
                max_memory={0: "12GiB"},
                trust_remote_code=True
            )
            # Add log immediately after model loading attempt
            logging.debug(f"[Worker {os.getpid()}] AutoModelForCausalLM.from_pretrained call completed.") 
            model.eval()
            logging.info(f"[Worker {os.getpid()}] Model loaded.")

            logging.debug(f"[Worker {os.getpid()}] Creating text-generation pipeline...")
            shared.llm_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
            logging.info(f"[Worker {os.getpid()}] Pipeline created and stored in shared state for this worker.")

        except Exception as e:
            logging.exception(f"[Worker {os.getpid()}] FATAL: Failed to load model/pipeline in worker. Error: {e}")
            # Keep shared.llm_pipeline as None, the task will fail
    # else:
        # logging.debug(f"[Worker {os.getpid()}] LLM pipeline already initialized.") # Optional: log reuse


# --- Celery Task Function --- 
@celery_app.task 
# Add temperature parameter to the function signature
def run_llm_inference_task(prompt: str, max_new_tokens: int, do_sample: bool, tools_definitions: Optional[List[Dict[str, Any]]], temperature: Optional[float] = None):
    """Task executed by Celery worker to run LLM inference and return cleaned text."""
    logging.info(f"*** Task received. Processing... ***")

    initialize_worker_pipeline()

    logging.debug(f"Accessing shared LLM pipeline...")
    if shared.llm_pipeline is None:
        logging.error(f"LLM pipeline is still None after initialization attempt!")
        raise RuntimeError("LLM pipeline could not be initialized in worker.")
    
    # Prepare pipeline arguments
    pipeline_args = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample
    }

    # Add temperature only if sampling is enabled
    if do_sample:
        # Use provided temperature or default to 0.6 if None
        pipeline_args["temperature"] = temperature if temperature is not None else 0.6
        logging.debug(f"Sampling enabled. Using temperature: {pipeline_args['temperature']}")
    else:
        logging.debug("Sampling disabled. Temperature parameter ignored.")
        # Ensure temperature is not passed if do_sample is False
        # (Though pipeline might ignore it anyway, it's cleaner not to pass it)

    logging.debug(f"Generating response via pipeline with args: {pipeline_args}...")
    llm_response_text = ""
    raw_llm_output = ""
    results = None
    try:
        # Call the pipeline with dynamic arguments
        logging.debug(f"Calling shared.llm_pipeline('{prompt[:50]}...', **pipeline_args)...")
        results = shared.llm_pipeline(prompt, **pipeline_args)
        logging.debug(f"Pipeline call complete. Result type: {type(results)}")
        
        # Process results immediately to clean the prompt
        logging.debug(f"Processing raw result to extract and clean text...")
        if results and isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict) and 'generated_text' in results[0]:
            raw_llm_output = results[0]['generated_text']
            # Clean prompt from the beginning
            if raw_llm_output.startswith(prompt):
                llm_response_text = raw_llm_output[len(prompt):].strip()
                logging.debug(f"Cleaned text extracted (prompt removed).")
            else:
                 llm_response_text = raw_llm_output.strip() # Fallback
                 logging.warning(f"Raw LLM output did not start with the prompt. Using full output.")

            # --- Log before cleaning --- 
            logging.debug(f"LLM Response BEFORE cleaning <think> tag: >>>{llm_response_text}<<<" ) 
            # -------------------------

            # --- ADDED: Clean internal thought markers --- 
            think_tag_marker = "</think>"
            think_tag_pos = llm_response_text.find(think_tag_marker)
            if think_tag_pos != -1:
                logging.debug(f"Found '{think_tag_marker}' tag at position {think_tag_pos}. Truncating response.")
                llm_response_text = llm_response_text[:think_tag_pos].strip()
            # --------------------------------------------- 

        else:
             logging.error(f"Unexpected LLM pipeline output format: {results}")
             raise ValueError("LLM pipeline returned unexpected output in worker")

        logging.info(f"Task processing finished. Preparing return dictionary.")
        # Return cleaned text and tool definitions for streamer to parse
        return_data = {
            "cleaned_response_text": llm_response_text,
            "tools_definitions": tools_definitions
        }
        logging.debug(f"Returning data: {{'cleaned_response_text': '...', 'tools_definitions': ...}}")
        return return_data # Celery task returns the result

    except Exception as e:
        logging.exception(f"ERROR during generation or processing: {e}") 
        raise # Re-raise the exception so Celery marks the task as failed
# --- END Celery Task Function --- 