#!/usr/bin/env python3
"""
Example demonstrating the Python API for the Rust-based Transformer implementation.
This shows how to use the ModelArgs and Transformer classes that interface with
the Rust backend.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llama.model import ModelArgs, Transformer, create_transformer_from_config
from llama.generation import Llama
from llama.tokenizer import Tokenizer


def example_direct_model_usage():
    """Example of using ModelArgs and Transformer directly"""
    print("=== Direct Model Usage Example ===")
    
    # Create model configuration
    model_args = ModelArgs(
        hidden_size=4096,
        vocab_size=32000,
        num_hidden_layers=32,
        num_attention_heads=32,
        max_seq_len=2048,
        max_batch_size=1
    )
    
    print(f"Model configuration:")
    print(f"  - Hidden size: {model_args.hidden_size}")
    print(f"  - Vocab size: {model_args.vocab_size}")
    print(f"  - Layers: {model_args.num_hidden_layers}")
    print(f"  - Attention heads: {model_args.num_attention_heads}")
    print(f"  - Head size: {model_args.attention_head_size}")
    
    # Create transformer model
    model = Transformer(model_args)
    
    # Example forward pass with mock data
    sequences = [[1, 2, 3, 4, 5]]  # Mock token sequences
    logits = model.forward(sequences)
    print(f"Forward pass completed, output shape: {len(logits)}x{len(logits[0])}")
    
    # Example generation
    generated = model.generate(sequences, max_gen_len=10)
    print(f"Generated sequences: {generated}")


def example_config_file_usage():
    """Example of loading configuration from file"""
    print("\n=== Config File Usage Example ===")
    
    config_path = "models/Llama-2-7b-hf/config.json"
    
    if Path(config_path).exists():
        print(f"Loading config from: {config_path}")
        
        # Load from config file
        model_args = ModelArgs.from_config_file(config_path)
        print(f"Loaded configuration:")
        print(f"  - Model type: {model_args.model_type}")
        print(f"  - Hidden size: {model_args.hidden_size}")
        print(f"  - Vocab size: {model_args.vocab_size}")
        print(f"  - Layers: {model_args.num_hidden_layers}")
        
        # Create transformer using factory function
        model = create_transformer_from_config(config_path, max_seq_len=512)
        print("Transformer created from config file")
        
    else:
        print(f"Config file not found: {config_path}")
        print("Skipping config file example")


def example_llama_api_usage():
    """Example of using the high-level Llama API"""
    print("\n=== High-level Llama API Example ===")
    
    model_dir = "models/Llama-2-7b-hf"
    tokenizer_path = "models/tokenizer.model"
    
    print(f"Model directory: {model_dir}")
    print(f"Tokenizer path: {tokenizer_path}")
    
    if Path(model_dir).exists():
        try:
            # Build Llama instance
            llama = Llama.build(
                ckpt_dir=model_dir,
                tokenizer_path=tokenizer_path if Path(tokenizer_path).exists() else "/dev/null",
                max_seq_len=512,
                max_batch_size=1
            )
            
            print("Llama model built successfully!")
            
            # Example generation
            prompt_tokens = [[1, 10, 20, 30]]  # Mock prompt tokens
            generated_tokens, logprobs = llama.generate(
                prompt_tokens=prompt_tokens,
                max_gen_len=5,
                temperature=0.7,
                top_p=0.9,
                logprobs=True
            )
            
            print(f"Generated tokens: {generated_tokens}")
            if logprobs:
                print(f"Log probabilities shape: {len(logprobs)}x{len(logprobs[0])}")
                
        except Exception as e:
            print(f"Error building Llama model: {e}")
    else:
        print(f"Model directory not found: {model_dir}")
        print("Skipping Llama API example")


def example_rust_integration():
    """Example showing Rust integration points"""
    print("\n=== Rust Integration Points ===")
    
    print("The Python API provides interfaces to the Rust backend:")
    print("1. ModelArgs -> Config struct in Rust")
    print("2. Transformer.forward() -> Rust Transformer::forward()")
    print("3. Weight loading -> Rust tensor operations")
    print("4. Generation -> Rust generation algorithms")
    
    print("\nKey benefits of Rust backend:")
    print("- High performance tensor operations")
    print("- Memory efficient f16 computations")
    print("- Optimized attention mechanisms")
    print("- SafeTensors model loading")
    print("- CPU-optimized SIMD kernels")


def main():
    """Run all examples"""
    print("Rust-backed Transformer Python API Examples")
    print("=" * 50)
    
    example_direct_model_usage()
    example_config_file_usage()
    example_llama_api_usage()
    example_rust_integration()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo use with real models:")
    print("1. Place model files in the models/ directory")
    print("2. Ensure config.json and tokenizer files are available")
    print("3. The Rust backend will handle the actual computations")


if __name__ == "__main__":
    main()
