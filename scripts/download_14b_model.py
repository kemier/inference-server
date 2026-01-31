#!/usr/bin/env python3
"""Download a 14B model from Hugging Face to a local directory for inference."""
import argparse
import os
import sys
from pathlib import Path

# Project root = parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import snapshot_download

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_LOCAL_DIR = "Qwen2.5-14B-Instruct"


def main():
    parser = argparse.ArgumentParser(description="Download a 14B model for inference.")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--local-dir",
        default=DEFAULT_LOCAL_DIR,
        help=f"Local directory name under project root (default: {DEFAULT_LOCAL_DIR})",
    )
    args = parser.parse_args()

    local_path = PROJECT_ROOT / args.local_dir
    local_path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.model_id} to {local_path} ...")
    snapshot_download(
        repo_id=args.model_id,
        local_dir=str(local_path),
    )
    print(f"Done. Model saved to {local_path}")
    rel_path = Path(args.local_dir)
    print(f"\nSet INFERENCE_MODEL_ID to one of:")
    print(f"  {rel_path}")
    print(f"  {local_path}")
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists() and (PROJECT_ROOT / ".env.example").exists():
        example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        new_content = example.replace(
            "INFERENCE_MODEL_ID=./Qwen-14B",
            f"INFERENCE_MODEL_ID={rel_path.as_posix()}",
        )
        env_path.write_text(new_content, encoding="utf-8")
        print(f"\nCreated .env with INFERENCE_MODEL_ID={rel_path.as_posix()}")
    else:
        print(f"\nTo use this model, set in .env: INFERENCE_MODEL_ID={rel_path.as_posix()}")


if __name__ == "__main__":
    main()
