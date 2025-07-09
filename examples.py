#!/usr/bin/env python3
"""
eLLM 使用示例 - 展示如何使用 Rust 后端的 Python API
"""

import sys
from pathlib import Path

# 添加本地包路径
sys.path.insert(0, str(Path(__file__).parent / "python"))


def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")

    from ellm import ModelArgs, Transformer

    # 1. 创建模型配置
    config = ModelArgs(
        hidden_size=2048,  # 隐藏层大小
        num_hidden_layers=16,  # Transformer 层数
        num_attention_heads=16,  # 注意力头数
        vocab_size=32000,  # 词汇表大小
        max_seq_len=1024,  # 最大序列长度
    )

    print(f"模型配置:")
    print(f"  - 隐藏层大小: {config.hidden_size}")
    print(f"  - 层数: {config.num_hidden_layers}")
    print(f"  - 注意力头数: {config.num_attention_heads}")
    print(f"  - 词汇表大小: {config.vocab_size}")
    print(f"  - 注意力头大小: {config.attention_head_size}")

    # 2. 创建 Transformer 模型
    model = Transformer(config)
    print(f"\n模型创建成功: {model}")

    # 3. 模拟权重加载
    print("\n加载模拟权重...")
    mock_weights = {
        "model.embed_tokens.weight": "tensor_placeholder",
        "model.norm.weight": "tensor_placeholder",
        "lm_head.weight": "tensor_placeholder",
    }
    model.load_state_dict(mock_weights)

    # 4. 前向传播示例
    print("\n执行前向传播...")
    input_sequences = [
        [1, 50, 100, 200],  # 第一个序列
        [1, 75, 150, 300, 400],  # 第二个序列
    ]

    try:
        logits = model.forward(input_sequences)
        print(f"前向传播成功，输出形状: {len(logits)} x {len(logits[0])}")
        print(f"第一个序列的前5个logits: {logits[0][:5]}")
    except Exception as e:
        print(f"前向传播出错: {e}")


def example_generation():
    """文本生成示例"""
    print("\n=== 文本生成示例 ===")

    from ellm import ModelArgs, Transformer

    # 创建小模型用于快速测试
    config = ModelArgs(
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        vocab_size=1000,
        max_seq_len=256,
    )

    model = Transformer(config)

    # 加载模拟权重
    mock_weights = {"dummy": "weight"}
    model.load_state_dict(mock_weights)

    # 生成文本
    prompt_tokens = [
        [1, 10, 20, 30],  # BOS + 一些 tokens
    ]

    print(f"输入提示 tokens: {prompt_tokens[0]}")

    try:
        generated = model.generate(
            prompt_tokens, max_gen_len=20, temperature=0.7, top_p=0.9
        )

        print(f"生成的完整序列: {generated[0]}")
        print(f"新生成的 tokens: {generated[0][len(prompt_tokens[0]):]}")
    except Exception as e:
        print(f"生成出错: {e}")


def example_config_loading():
    """从配置文件加载示例"""
    print("\n=== 配置文件加载示例 ===")

    from ellm import ModelArgs, create_transformer_from_config
    import json
    import tempfile
    import os

    # 创建临时配置文件
    config_data = {
        "hidden_size": 4096,
        "vocab_size": 32000,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "intermediate_size": 11008,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-6,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "model_type": "llama",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f, indent=2)
        config_path = f.name

    try:
        # 从文件加载配置
        model_args = ModelArgs.from_config_file(config_path)
        print(f"从配置文件加载: {config_path}")
        print(f"配置: {model_args.hidden_size}x{model_args.num_hidden_layers}")

        # 创建模型
        model = create_transformer_from_config(
            config_path,
            max_seq_len=1024,  # 覆盖配置参数
        )
        print(f"模型创建成功: {model.model_args.max_seq_len}")

    finally:
        # 清理临时文件
        os.unlink(config_path)


def example_high_level_api():
    """高级 API 示例"""
    print("\n=== 高级 API 示例 ===")

    try:
        from ellm import Llama

        # 注意：这需要实际的模型文件，这里只是演示 API
        print("高级 Llama API 演示 (需要实际模型文件):")
        print("""
        # 初始化 Llama 模型
        llama = Llama(
            model_path="path/to/model",
            tokenizer_path="path/to/tokenizer",
            max_seq_len=2048
        )
        
        # 简单文本生成
        response = llama.generate(
            "解释一下什么是人工智能",
            max_gen_len=100,
            temperature=0.7
        )
        
        # 聊天对话
        messages = [
            {"role": "user", "content": "你好，请介绍一下自己"}
        ]
        response = llama.chat_completion(messages)
        """)

    except ImportError as e:
        print(f"高级 API 导入失败: {e}")


def main():
    """运行所有示例"""
    print("🚀 eLLM Python API 使用示例")
    print("=" * 50)

    try:
        example_basic_usage()
        example_generation()
        example_config_loading()
        example_high_level_api()

        print("\n" + "=" * 50)
        print("✅ 所有示例运行完成!")
        print("\n要构建和安装完整的 Rust 扩展，请运行:")
        print("  python build.py")

    except Exception as e:
        print(f"\n❌ 示例运行出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
