# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

"""
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
"""

from .model import ModelArgs, Transformer
from .tokenizer import ChatFormat, Dialog, Message, Tokenizer


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class Llama:
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
        Build a Llama instance by initializing and loading a model checkpoint.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.
        """
        assert 1 <= max_seq_len <= 8192, f"max_seq_len must be between 1 and 8192, got {max_seq_len}."
        # assert os.path.isdir(ckpt_dir), f"Checkpoint directory '{ckpt_dir}' does not exist."
        assert os.path.isfile(tokenizer_path), f"Tokenizer file '{tokenizer_path}' does not exist."
        
        # if not torch.distributed.is_initialized():
        #    torch.distributed.init_process_group("nccl")
        # if not model_parallel_is_initialized():
        #    if model_parallel_size is None:
        #        model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        #    initialize_model_parallel(model_parallel_size)

        # local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        #torch.manual_seed(seed)

        #if local_rank > 0:
        #    sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        # checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        # assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        # assert model_parallel_size == len(
        #    checkpoints
        # ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        # ckpt_path = checkpoints[get_model_parallel_rank()]
        # checkpoint = torch.load(ckpt_path, map_location="cpu")
        # with open(Path(ckpt_dir) / "params.json", "r") as f:
        #     params = json.loads(f.read())

        # Load model configuration from config.json
        config_path = Path(ckpt_dir) / "config.json"
        if config_path.exists():
            model_args = ModelArgs.from_config_file(str(config_path))
        else:
            # Create default model args if config doesn't exist
            model_args = ModelArgs()
        
        # Override with provided parameters
        model_args.max_seq_len = max_seq_len
        model_args.max_batch_size = max_batch_size
        
        tokenizer = Tokenizer(model_path=tokenizer_path)
        # assert model_args.vocab_size == tokenizer.n_words
        # if torch.cuda.is_bf16_supported():
        #     torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        # else:
        #    torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        # model.load_state_dict(checkpoint, strict=False)
        
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = ChatFormat(tokenizer)

    # @torch.inference_mode()
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
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        prev_pos = 0
        
        # Use the Rust-backed Transformer implementation
        # Convert the input format and call the Rust backend
        generated_sequences = self.model.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p
        )
        
        if logprobs:
            # Mock logprobs for now - in real implementation would come from Rust
            mock_logprobs = [[0.0] * len(seq) for seq in generated_sequences]
            return generated_sequences, mock_logprobs
        else:
            return generated_sequences, None
            



