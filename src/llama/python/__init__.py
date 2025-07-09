# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

"""
eLLM - Efficient Language Model Implementation in Rust with Python API

This package provides Python bindings for a high-performance Rust-based
implementation of the Llama language model architecture.

Key Features:
- High-performance Rust backend with optimized kernels
- Memory-efficient f16 precision support
- SafeTensors model loading
- PyTorch-compatible Python API
- CPU-optimized SIMD operations

Main Components:
- ModelArgs: Configuration class (corresponds to Rust Config)
- Transformer: Main model class (corresponds to Rust Transformer)
- Llama: High-level generation interface
- Tokenizer: Text tokenization utilities

Example Usage:
    >>> from llama import ModelArgs, Transformer, Llama
    >>> 
    >>> # Create model configuration
    >>> model_args = ModelArgs.from_config_file("config.json")
    >>> 
    >>> # Create transformer model
    >>> model = Transformer(model_args)
    >>> 
    >>> # Or use high-level interface
    >>> llama = Llama.build(
    ...     ckpt_dir="models/llama-7b",
    ...     tokenizer_path="tokenizer.model",
    ...     max_seq_len=2048,
    ...     max_batch_size=1
    ... )
"""

from .model import (
    ModelArgs,
    Transformer, 
    create_transformer_from_config,
    RustFFI
)
from .generation import (
    Llama,
    CompletionPrediction,
    ChatPrediction,
    load_model_from_checkpoint,
    get_model_size_info
)
from .tokenizer import (
    Tokenizer,
    ChatFormat,
    Dialog,
    Message
)

__version__ = "0.1.0"
__author__ = "eLLM Team"
__email__ = "contact@ellm.ai"

# Public API
__all__ = [
    # Core model classes
    "ModelArgs",
    "Transformer", 
    "create_transformer_from_config",
    
    # High-level interfaces
    "Llama",
    "CompletionPrediction",
    "ChatPrediction",
    
    # Tokenization
    "Tokenizer",
    "ChatFormat",
    "Dialog", 
    "Message",
    
    # Utilities
    "load_model_from_checkpoint",
    "get_model_size_info",
    
    # Advanced
    "RustFFI",
]

# Package metadata
__package_info__ = {
    "name": "ellm",
    "version": __version__,
    "description": "Efficient Language Model Implementation in Rust with Python API",
    "author": __author__,
    "email": __email__,
    "url": "https://github.com/your-org/ellm",
    "license": "Llama 3 Community License",
    "python_requires": ">=3.8",
    "rust_backend": True,
    "supported_precisions": ["f16", "f32"],
    "supported_devices": ["cpu"],  # Future: ["cpu", "cuda"]
}

def get_version():
    """Get the current version of the eLLM package."""
    return __version__

def get_package_info():
    """Get detailed package information."""
    return __package_info__.copy()

def check_rust_backend():
    """Check if the Rust backend is available and working."""
    try:
        # In a full implementation, this would test the FFI connection
        model_args = ModelArgs()
        model = Transformer(model_args)
        print("✓ Rust backend is available")
        return True
    except Exception as e:
        print(f"✗ Rust backend check failed: {e}")
        return False

# Module-level configuration
class Config:
    """Global configuration for the eLLM package."""
    
    # Logging configuration
    DEBUG = False
    LOG_LEVEL = "INFO"
    
    # Performance settings
    DEFAULT_CPU_CORES = None  # Auto-detect
    ENABLE_PROFILING = False
    
    # Memory settings  
    DEFAULT_PRECISION = "f16"
    CACHE_SIZE_LIMIT = None  # No limit
    
    # Backend settings
    RUST_FFI_TIMEOUT = 30  # seconds
    FALLBACK_TO_PYTHON = False
    
    @classmethod
    def set_debug(cls, enabled: bool):
        """Enable or disable debug mode."""
        cls.DEBUG = enabled
        cls.LOG_LEVEL = "DEBUG" if enabled else "INFO"
    
    @classmethod
    def set_precision(cls, precision: str):
        """Set default precision (f16 or f32)."""
        if precision not in ["f16", "f32"]:
            raise ValueError("Precision must be 'f16' or 'f32'")
        cls.DEFAULT_PRECISION = precision
    
    @classmethod
    def get_config(cls) -> dict:
        """Get current configuration as dictionary."""
        return {
            "debug": cls.DEBUG,
            "log_level": cls.LOG_LEVEL,
            "default_cpu_cores": cls.DEFAULT_CPU_CORES,
            "enable_profiling": cls.ENABLE_PROFILING,
            "default_precision": cls.DEFAULT_PRECISION,
            "cache_size_limit": cls.CACHE_SIZE_LIMIT,
            "rust_ffi_timeout": cls.RUST_FFI_TIMEOUT,
            "fallback_to_python": cls.FALLBACK_TO_PYTHON,
        }

# Convenience functions
def set_debug_mode(enabled: bool = True):
    """Enable or disable debug mode globally."""
    Config.set_debug(enabled)

def set_default_precision(precision: str):
    """Set the default precision for computations."""
    Config.set_precision(precision)

def print_system_info():
    """Print system and package information."""
    import platform
    import sys
    
    print("eLLM System Information")
    print("=" * 50)
    print(f"Package Version: {__version__}")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU Count: {platform.processor()}")
    
    # Package configuration
    config = Config.get_config()
    print("\nPackage Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Check backend
    print("\nBackend Status:")
    rust_available = check_rust_backend()
    print(f"  Rust Backend: {'Available' if rust_available else 'Not Available'}")

# Initialize package
def _initialize_package():
    """Initialize the package with default settings."""
    # Set up logging if needed
    if Config.DEBUG:
        import logging
        logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
    
    # Perform any necessary startup checks
    # (In a full implementation, this might validate the Rust backend)

# Run initialization
_initialize_package()
