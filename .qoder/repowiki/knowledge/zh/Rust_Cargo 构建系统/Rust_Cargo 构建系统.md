---
kind: build_system
name: Rust/Cargo 构建系统
category: build_system
scope:
    - '**'
source_files:
    - Cargo.toml
    - rust-toolchain.toml
    - .cargo/config.toml
---

本仓库采用纯 Rust + Cargo 的单一包构建体系，未引入 Makefile、Dockerfile 或 CI 流水线等外部构建工具。核心构建配置集中在以下位置：

- **Cargo.toml**：定义包名 `eLLM`、crate-type 同时输出 `cdylib` 与 `rlib`（便于 Python/FFI 调用），并通过多个 `[[bin]]` 条目将 `alignment/` 下的对齐测试直接暴露为独立二进制（如 `rope_alignment_test`、`silu_mul_alignment_test`、`qwen3_tokenizer_alignment` 等）。
- **rust-toolchain.toml**：锁定工具链为 `nightly`，启用 nightly-only 特性。
- **.cargo/config.toml**：全局关闭增量编译并设置 `-C target-cpu=native`，使每次构建都针对本机 CPU 指令集优化。
- **profile.release**：开启 `opt-level=3`、`lto=fat`、`codegen-units=1`、`panic="abort"`、`strip=symbols`、`incremental=false`，追求极致发布体积与运行性能；dev profile 则保留 `opt-level=1`、`incremental=true`、`codegen-units=16` 以加速迭代。
- **build-dependencies**：使用 `raw-cpuid` 在编译期探测 CPU 能力，配合 `src/kernel/x86_64/` 下按 `f16_512`、`f16_amx`、`f32_256` 子目录划分的 SIMD 内核，实现运行时/编译时自动选择最优路径。

依赖管理完全通过 Cargo 版本约束完成，无 vendoring、无 lock 文件提交约定说明。测试与基准分别通过 `cargo test` 与 `criterion`（dev-dependency）驱动，benchmark HTML 报告由 `features = ["html_reports"]` 生成。

未发现 Dockerfile、Makefile、GitHub Actions、CI 脚本或跨平台交叉编译配置，发布流程目前仅基于 `cargo build --release` 产物。