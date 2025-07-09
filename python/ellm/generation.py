"""
High-level generation interface for the Llama model.
"""

from typing import List, Optional, Union
from .model import Transformer, ModelArgs

try:
    from ._lowlevel import Tokenizer as RustTokenizer
except ImportError:
    # 如果 Rust 模块未编译，提供模拟类
    class RustTokenizer:
        def __init__(self, tokenizer_path):
            self.vocab_size = 32000

        def encode(self, text, add_bos=True, add_eos=False):
            # 简单的模拟编码
            return [1] + [ord(c) % 1000 for c in text] + ([2] if add_eos else [])

        def decode(self, tokens):
            return " ".join(f"tok_{t}" for t in tokens)


class Llama:
    """
    High-level interface for the Llama model that handles both tokenization and generation.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        max_seq_len: int = 2048,
        max_batch_size: int = 1,
    ):
        """
        Initialize Llama model.

        Args:
            model_path: Path to model configuration or weights
            tokenizer_path: Path to tokenizer files
            max_seq_len: Maximum sequence length
            max_batch_size: Maximum batch size
        """
        # 加载配置
        if model_path.endswith(".json"):
            self.model_args = ModelArgs.from_config_file(model_path)
        else:
            # 假设是模型目录，查找 config.json
            import os

            config_path = os.path.join(model_path, "config.json")
            self.model_args = ModelArgs.from_config_file(config_path)

        # 设置序列长度和批处理大小
        self.model_args.max_seq_len = max_seq_len
        self.model_args.max_batch_size = max_batch_size

        # 初始化模型和分词器
        self.model = Transformer(self.model_args)
        self.tokenizer = RustTokenizer(tokenizer_path)

        print(f"Initialized Llama model:")
        print(
            f"  - Model: {self.model_args.num_hidden_layers} layers, {self.model_args.hidden_size} hidden size"
        )
        print(f"  - Tokenizer: {self.tokenizer.vocab_size} vocab size")
        print(f"  - Max sequence length: {max_seq_len}")

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_gen_len: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> Union[str, List[str]]:
        """
        Generate text from prompts.

        Args:
            prompts: Input text prompt(s)
            max_gen_len: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text(s)
        """
        # 处理单个字符串输入
        single_input = isinstance(prompts, str)
        if single_input:
            prompts = [prompts]

        # 分词
        prompt_tokens = []
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
            prompt_tokens.append(tokens)

        print(f"Generating for {len(prompt_tokens)} prompts, max_gen_len={max_gen_len}")

        # 生成
        generated_tokens = self.model.generate(
            prompt_tokens, max_gen_len, temperature, top_p
        )

        # 解码
        generated_texts = []
        for tokens in generated_tokens:
            text = self.tokenizer.decode(tokens)
            generated_texts.append(text)

        # 返回单个字符串或列表
        return generated_texts[0] if single_input else generated_texts

    def chat_completion(
        self,
        messages: List[dict],
        max_gen_len: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> str:
        """
        Chat completion interface compatible with OpenAI API format.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_gen_len: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated response text
        """
        # 构建聊天提示
        prompt_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)

        # 生成回复
        response = self.generate(prompt, max_gen_len, temperature, top_p)

        # 提取助手回复部分
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        return response

    def load_weights(self, weights_path: str):
        """
        Load model weights from file.

        Args:
            weights_path: Path to weights file (.safetensors, .bin, etc.)
        """
        print(f"Loading weights from: {weights_path}")

        # TODO: 实现实际的权重加载
        # 这里需要根据文件格式（safetensors, pickle等）加载权重
        mock_state_dict = {
            "model.embed_tokens.weight": "tensor_data",
            "model.norm.weight": "tensor_data",
            "lm_head.weight": "tensor_data",
        }

        self.model.load_state_dict(mock_state_dict)
        print("✓ Weights loaded successfully")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Llama(\n"
            f"  model={self.model_args.num_hidden_layers} layers,\n"
            f"  vocab_size={self.tokenizer.vocab_size},\n"
            f"  max_seq_len={self.model_args.max_seq_len}\n"
            f")"
        )
