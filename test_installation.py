#!/usr/bin/env python3
"""
测试 eLLM 安装和基本功能
"""


def test_import():
    """测试模块导入"""
    print("Testing imports...")

    try:
        import ellm

        print(f"✓ ellm imported successfully, version: {ellm.__version__}")
    except ImportError as e:
        print(f"❌ Failed to import ellm: {e}")
        return False

    try:
        from ellm import ModelArgs, Transformer, Llama

        print("✓ Main classes imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import main classes: {e}")
        return False

    return True


def test_config():
    """测试配置功能"""
    print("\nTesting configuration...")

    try:
        from ellm import ModelArgs

        # 测试默认配置
        config = ModelArgs()
        print(
            f"✓ Default config created: {config.hidden_size}x{config.num_hidden_layers}"
        )

        # 测试自定义配置
        custom_config = ModelArgs(
            hidden_size=2048, num_hidden_layers=16, vocab_size=50000
        )
        print(
            f"✓ Custom config created: {custom_config.hidden_size}x{custom_config.num_hidden_layers}"
        )

        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_transformer():
    """测试 Transformer 功能"""
    print("\nTesting Transformer...")

    try:
        from ellm import ModelArgs, Transformer

        # 创建小模型用于测试
        config = ModelArgs(
            hidden_size=512, num_hidden_layers=4, num_attention_heads=8, vocab_size=1000
        )

        model = Transformer(config)
        print(f"✓ Transformer created: {model}")

        # 测试模拟的前向传播（在未加载权重时会失败，这是预期的）
        try:
            sequences = [[1, 2, 3, 4, 5]]
            logits = model.forward(sequences)
            print("❌ Forward pass should have failed without weights")
            return False
        except RuntimeError as e:
            if "weights must be loaded" in str(e):
                print("✓ Forward pass correctly failed without weights")
            else:
                print(f"❌ Unexpected error: {e}")
                return False

        return True
    except Exception as e:
        print(f"❌ Transformer test failed: {e}")
        return False


def test_rust_bindings():
    """测试 Rust 绑定（如果可用）"""
    print("\nTesting Rust bindings...")

    try:
        from ellm._lowlevel import Config, Transformer as RustTransformer

        # 测试 Rust Config
        rust_config = Config(
            hidden_size=512, vocab_size=1000, num_hidden_layers=4, num_attention_heads=8
        )
        print(f"✓ Rust Config created: {rust_config}")

        # 测试 Rust Transformer
        rust_model = RustTransformer(rust_config)
        print(f"✓ Rust Transformer created: {rust_model}")

        return True
    except ImportError:
        print("⚠️  Rust bindings not available (module not compiled)")
        return True  # 这不是错误，只是模块未编译
    except Exception as e:
        print(f"❌ Rust bindings test failed: {e}")
        return False


def main():
    """运行所有测试"""
    print("🧪 Testing eLLM Installation")
    print("=" * 40)

    tests = [test_import, test_config, test_transformer, test_rust_bindings]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
