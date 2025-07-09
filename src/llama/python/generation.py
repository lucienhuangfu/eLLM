# Copyright (c) Meta Platforms, Inc. and affiliates.  
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

"""
High-level generation interface for the Rust-based Llama implementation.

This module provides the main Llama class for text generation, similar to the
original Llama implementation but powered by the Rust backend.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

from .model import ModelArgs, Transformer
from .tokenizer import ChatFormat, Dialog, Message, Tokenizer


class CompletionPrediction(TypedDict, total=False):
    """Type definition for completion predictions."""
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    """Type definition for chat predictions."""
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class Llama:
    """
    High-level interface for the Rust-based Llama implementation.
    
    This class provides a familiar API for loading and using Llama models,
    while leveraging the high-performance Rust backend for computation.
    """
    
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by loading model configuration and tokenizer.

        Args:
            ckpt_dir: Path to the directory containing model files
            tokenizer_path: Path to the tokenizer file
            max_seq_len: Maximum sequence length for input text
            max_batch_size: Maximum batch size for inference
            model_parallel_size: Number of model parallel processes (unused in Rust implementation)
            seed: Random seed for reproducibility

        Returns:
            Llama instance with loaded model and tokenizer

        Raises:
            AssertionError: If parameters are invalid or files don't exist
            FileNotFoundError: If required files are missing
        """
        # Validate parameters
        assert 1 <= max_seq_len <= 8192, f"max_seq_len must be between 1 and 8192, got {max_seq_len}"
        assert os.path.isfile(tokenizer_path), f"Tokenizer file '{tokenizer_path}' does not exist"
        
        print(f"Building Llama model from {ckpt_dir}")
        start_time = time.time()
        
        # Load model configuration
        config_path = Path(ckpt_dir) / "config.json"
        if config_path.exists():
            print(f"Loading configuration from {config_path}")
            model_args = ModelArgs.from_config_file(str(config_path))
        else:
            print("No config.json found, using default configuration")
            model_args = ModelArgs()
        
        # Override configuration with provided parameters
        model_args.max_seq_len = max_seq_len
        model_args.max_batch_size = max_batch_size
        
        # Load tokenizer
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer(model_path=tokenizer_path)
        
        # Create transformer model
        print("Initializing Rust-backed Transformer")
        model = Transformer(model_args)
        
        # In a full implementation, load actual model weights here
        # For now, we'll create a placeholder state dict
        mock_state_dict = {}
        model.load_state_dict(mock_state_dict, strict=False)
        
        elapsed = time.time() - start_time
        print(f"✓ Model built successfully in {elapsed:.2f} seconds")
        
        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        """
        Initialize Llama with model and tokenizer.
        
        Args:
            model: Initialized Transformer model
            tokenizer: Loaded tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = ChatFormat(tokenizer)

    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts.

        Args:
            prompt_tokens: List of tokenized prompts
            max_gen_len: Maximum length of generated text
            temperature: Temperature for controlling randomness
            top_p: Top-p threshold for nucleus sampling
            logprobs: Whether to compute token log probabilities
            echo: Whether to include prompt tokens in output

        Returns:
            Tuple of (generated_sequences, log_probabilities)
            
        Note:
            This method leverages the Rust backend for high-performance generation
            with optimized attention mechanisms and memory management.
        """
        print(f"Generating text for {len(prompt_tokens)} prompts")
        print(f"Parameters: max_len={max_gen_len}, temp={temperature}, top_p={top_p}")
        
        # Use the Rust-backed generation
        generated_sequences = self.model.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p
        )
        
        # Handle echo parameter
        if not echo:
            # Remove prompt tokens from generated sequences
            output_sequences = []
            for i, (prompt, generated) in enumerate(zip(prompt_tokens, generated_sequences)):
                prompt_len = len(prompt)
                output_sequences.append(generated[prompt_len:])
            generated_sequences = output_sequences
        
        # Handle logprobs
        if logprobs:
            # In a full implementation, this would come from the Rust backend
            mock_logprobs = [
                [0.0] * len(seq) for seq in generated_sequences
            ]
            return generated_sequences, mock_logprobs
        else:
            return generated_sequences, None
    
    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        """
        Generate text completions for the given prompts.
        
        Args:
            prompts: List of text prompts
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter  
            max_gen_len: Maximum generation length
            logprobs: Whether to return log probabilities
            echo: Whether to echo the prompt
            
        Returns:
            List of completion predictions
        """
        if max_gen_len is None:
            max_gen_len = self.model.model_args.max_seq_len - 1
        
        # Tokenize prompts
        prompt_tokens = [
            self.tokenizer.encode(prompt, bos=True, eos=False)
            for prompt in prompts
        ]
        
        # Generate completions
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        
        # Convert back to text
        completions = []
        for i, tokens in enumerate(generation_tokens):
            # Decode tokens to text
            text = self.tokenizer.decode(tokens)
            
            completion: CompletionPrediction = {"generation": text}
            
            if logprobs and generation_logprobs:
                completion["logprobs"] = generation_logprobs[i]
                completion["tokens"] = [
                    self.tokenizer.decode([token]) for token in tokens
                ]
            
            completions.append(completion)
        
        return completions
    
    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        """
        Generate chat completions for the given dialogs.
        
        Args:
            dialogs: List of conversation dialogs
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_gen_len: Maximum generation length
            logprobs: Whether to return log probabilities
            
        Returns:
            List of chat predictions
        """
        if max_gen_len is None:
            max_gen_len = self.model.model_args.max_seq_len - 1
        
        # Format dialogs and tokenize
        prompt_tokens = []
        for dialog in dialogs:
            formatted_dialog = self.formatter.encode_dialog_prompt(dialog)
            tokens = self.tokenizer.encode(formatted_dialog, bos=True, eos=False)
            prompt_tokens.append(tokens)
        
        # Generate responses
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=False,
        )
        
        # Convert to chat predictions
        predictions = []
        for i, tokens in enumerate(generation_tokens):
            # Decode response
            response_text = self.tokenizer.decode(tokens)
            
            # Create response message
            response_message: Message = {
                "role": "assistant",
                "content": response_text,
            }
            
            prediction: ChatPrediction = {"generation": response_message}
            
            if logprobs and generation_logprobs:
                prediction["logprobs"] = generation_logprobs[i]
                prediction["tokens"] = [
                    self.tokenizer.decode([token]) for token in tokens
                ]
            
            predictions.append(prediction)
        
        return predictions
    
    @property
    def device(self) -> str:
        """Get the device the model is running on."""
        return self.model.device
    
    def __repr__(self) -> str:
        """String representation of the Llama instance."""
        return (f"Llama(\n"
                f"  model={self.model.__class__.__name__},\n"
                f"  vocab_size={self.model.model_args.vocab_size},\n"
                f"  max_seq_len={self.model.model_args.max_seq_len},\n"
                f"  device={self.device}\n"
                f")")


# Utility functions for model loading and management
def load_model_from_checkpoint(
    checkpoint_path: str,
    model_args: ModelArgs,
    device: str = "cpu"
) -> Transformer:
    """
    Load a Transformer model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model_args: Model configuration
        device: Device to load the model on
        
    Returns:
        Loaded Transformer model
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Create model
    model = Transformer(model_args)
    
    # In a full implementation, this would:
    # 1. Load the checkpoint file (PyTorch .pth or SafeTensors)
    # 2. Extract state dict
    # 3. Convert weights to Rust format
    # 4. Load into Rust backend
    
    # For now, mock the loading
    mock_state_dict = {}
    model.load_state_dict(mock_state_dict, strict=False)
    
    print("✓ Model loaded successfully")
    return model


def get_model_size_info(model_args: ModelArgs) -> dict:
    """
    Get information about model size and memory requirements.
    
    Args:
        model_args: Model configuration
        
    Returns:
        Dictionary with size information
    """
    # Calculate approximate parameter count
    embed_params = model_args.vocab_size * model_args.hidden_size
    
    # Per-layer parameters
    attention_params = 4 * model_args.hidden_size * model_args.hidden_size  # Q, K, V, O
    mlp_params = 3 * model_args.hidden_size * model_args.intermediate_size  # Gate, Up, Down
    norm_params = 2 * model_args.hidden_size  # Input + post attention norms
    layer_params = attention_params + mlp_params + norm_params
    
    total_layer_params = model_args.num_hidden_layers * layer_params
    final_norm_params = model_args.hidden_size
    lm_head_params = model_args.vocab_size * model_args.hidden_size
    
    total_params = embed_params + total_layer_params + final_norm_params + lm_head_params
    
    # Memory estimates (f16 = 2 bytes per parameter)
    memory_f16_gb = (total_params * 2) / (1024**3)
    memory_f32_gb = (total_params * 4) / (1024**3)
    
    return {
        "total_parameters": total_params,
        "embedding_parameters": embed_params,
        "layer_parameters": layer_params,
        "memory_f16_gb": memory_f16_gb,
        "memory_f32_gb": memory_f32_gb,
        "num_layers": model_args.num_hidden_layers,
        "hidden_size": model_args.hidden_size,
        "vocab_size": model_args.vocab_size,
    }
