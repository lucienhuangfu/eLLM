"""
High-level model interface that wraps the Rust implementation.
"""

from typing import Dict, List, Optional, Any, Union
import json
from pathlib import Path
from dataclasses import dataclass, field

try:
    from ._lowlevel import Config as RustConfig, Transformer as RustTransformer
except ImportError:
    # 如果 Rust 模块未编译，提供模拟类
    class RustConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class RustTransformer:
        def __init__(self, config):
            self.config = config

        def forward(self, input_ids, start_pos=0):
            raise NotImplementedError("Rust backend not available")


@dataclass
class ModelArgs:
    """
    Model configuration arguments that correspond to the Rust Config struct.
    """

    # Core model architecture
    hidden_size: int = 4096
    vocab_size: int = 32000
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    intermediate_size: int = 11008
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6

    # Sequence and batch configuration
    batch_size: int = 1
    max_seq_len: int = 2048
    max_batch_size: int = 32

    # Architecture details
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: Optional[int] = None
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    model_type: str = "llama"
    pretraining_tp: int = 1
    rope_scaling: Optional[Dict[str, Any]] = None
    tie_word_embeddings: bool = False
    torch_dtype: str = "float16"
    transformers_version: str = "4.21.0.dev0"
    use_cache: bool = True

    # Computed fields
    attention_head_size: Optional[int] = field(init=False, default=None)

    def __post_init__(self):
        """Post-initialization processing."""
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.attention_head_size is None:
            self.attention_head_size = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_config_file(cls, config_path: str) -> "ModelArgs":
        """Load model arguments from a config.json file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Create ModelArgs instance with values from config
        kwargs = {}
        field_names = {f.name for f in cls.__dataclass_fields__.values()}

        for key, value in config.items():
            if key in field_names:
                kwargs[key] = value

        return cls(**kwargs)

    def to_rust_config(self) -> RustConfig:
        """Convert to Rust Config object."""
        return RustConfig(
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            batch_size=self.batch_size,
        )


class Transformer:
    """
    High-level Python wrapper for the Rust Transformer implementation.
    """

    def __init__(self, model_args: ModelArgs):
        """Initialize Transformer with model arguments."""
        self.model_args = model_args
        self.config = model_args  # Alias for compatibility

        # 创建 Rust 后端
        rust_config = model_args.to_rust_config()
        self._rust_transformer = RustTransformer(rust_config)
        self._weights_loaded = False

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """Load model weights from state dictionary."""
        print(f"Loading {len(state_dict)} tensors into Rust Transformer...")

        if hasattr(self._rust_transformer, "load_state_dict"):
            self._rust_transformer.load_state_dict(state_dict)

        self._weights_loaded = True
        print("✓ Weights loaded successfully")

    def forward(
        self, sequences: List[List[int]], start_pos: int = 0
    ) -> List[List[float]]:
        """Forward pass through the transformer."""
        if not self._weights_loaded:
            raise RuntimeError("Model weights must be loaded before forward pass")

        return self._rust_transformer.forward(sequences, start_pos)

    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> List[List[int]]:
        """Generate text given prompt tokens."""
        if not self._weights_loaded:
            raise RuntimeError("Model weights must be loaded before generation")

        return self._rust_transformer.generate(
            prompt_tokens, max_gen_len, temperature, top_p
        )

    @property
    def device(self) -> str:
        """Get model device."""
        return "cpu"  # Rust 实现目前只支持 CPU

    def eval(self) -> "Transformer":
        """Set model to evaluation mode."""
        return self

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Transformer(\n"
            f"  hidden_size={self.model_args.hidden_size},\n"
            f"  num_layers={self.model_args.num_hidden_layers},\n"
            f"  num_heads={self.model_args.num_attention_heads},\n"
            f"  vocab_size={self.model_args.vocab_size},\n"
            f"  device={self.device}\n"
            f")"
        )


def create_transformer_from_config(config_path: str, **kwargs) -> Transformer:
    """
    Create a Transformer instance from a configuration file.

    Args:
        config_path: Path to the config.json file
        **kwargs: Additional arguments to override config values

    Returns:
        Initialized Transformer instance
    """
    model_args = ModelArgs.from_config_file(config_path)

    # Override any config values with provided kwargs
    for key, value in kwargs.items():
        if hasattr(model_args, key):
            setattr(model_args, key, value)
        else:
            print(f"Warning: Unknown config parameter '{key}' ignored")

    return Transformer(model_args)
