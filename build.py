#!/usr/bin/env python3
"""
构建脚本 - 编译 Rust 扩展并安装 Python 包
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, cwd=None):
    """运行命令并检查结果"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(1)


def main():
    """主构建流程"""
    project_root = Path(__file__).parent

    print("🚀 Building eLLM Rust-Python Extension")
    print("=" * 50)

    # 1. 检查依赖
    print("📋 Checking dependencies...")

    # 检查 Rust
    try:
        subprocess.run(["rustc", "--version"], check=True, capture_output=True)
        print("✓ Rust is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Rust is not installed. Please install from https://rustup.rs/")
        sys.exit(1)

    # 检查 maturin
    try:
        subprocess.run(["maturin", "--version"], check=True, capture_output=True)
        print("✓ maturin is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("📦 Installing maturin...")
        run_command("pip install maturin[patchelf]")

    # 2. 编译 Rust 扩展
    print("\n🔨 Building Rust extension...")

    # 添加 python feature
    run_command("cargo build --features python", cwd=project_root)

    # 使用 maturin 构建 Python 轮子
    run_command("maturin develop --features python", cwd=project_root)

    # 3. 安装 Python 包
    print("\n📦 Installing Python package...")
    run_command("pip install -e .", cwd=project_root)

    # 4. 运行测试
    print("\n🧪 Running tests...")
    test_script = project_root / "test_installation.py"
    if test_script.exists():
        run_command(f"python {test_script}", cwd=project_root)
    else:
        print("⚠️  No test script found, skipping tests")

    print("\n✅ Build completed successfully!")
    print("\nYou can now use the eLLM package:")
    print("  python -c 'import ellm; print(ellm.__version__)'")


if __name__ == "__main__":
    main()
