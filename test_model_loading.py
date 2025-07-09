#!/usr/bin/env python3
"""
Example script demonstrating direct model loading without checkpoint APIs.
This loads model configuration directly from config.json files.
"""

import os
import sys
from pathlib import Path

# Add the src directory to Python path so we can import the llama modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llama.generation import Llama
from llama.model import ModelArgs

def main():
    # Example model directory and tokenizer path
    model_dir = "models/Llama-2-7b-hf"  # Directory containing config.json
    tokenizer_path = "models/tokenizer.model"  # Adjust path as needed
    
    # Model parameters
    max_seq_len = 512
    max_batch_size = 1
    
    print("Loading Llama model directly from configuration files...")
    print(f"Model directory: {model_dir}")
    print(f"Tokenizer path: {tokenizer_path}")
    
    try:
        # Check if model directory exists
        if not os.path.isdir(model_dir):
            print(f"Error: Model directory '{model_dir}' does not exist.")
            print("Available models:")
            models_dir = Path("models")
            if models_dir.exists():
                for item in models_dir.iterdir():
                    if item.is_dir():
                        print(f"  - {item.name}")
            return 1
        
        # Check if config.json exists
        config_path = Path(model_dir) / "config.json"
        if not config_path.exists():
            print(f"Error: Configuration file not found: {config_path}")
            return 1
        
        # Create a dummy tokenizer file for testing if it doesn't exist
        if not os.path.isfile(tokenizer_path):
            print(f"Warning: Tokenizer file '{tokenizer_path}' does not exist.")
            print("Creating a dummy tokenizer file for testing...")
            os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
            with open(tokenizer_path, "w") as f:
                f.write("# Dummy tokenizer file for testing\n")
        
        # Build the Llama model
        llama = Llama.build(
            ckpt_dir=model_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size
        )
        
        print("✓ Model loaded successfully!")
        print(f"Model configuration:")
        print(f"  - Hidden size: {llama.model.model_args.hidden_size}")
        print(f"  - Vocab size: {llama.model.model_args.vocab_size}")
        print(f"  - Max sequence length: {llama.model.model_args.max_seq_len}")
        print(f"  - Number of layers: {llama.model.model_args.num_hidden_layers}")
        print(f"  - Number of attention heads: {llama.model.model_args.num_attention_heads}")
        
        print("\nNote: This loads the model configuration only.")
        print("Actual model weights would be loaded by the Rust backend.")
        
        return 0
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
