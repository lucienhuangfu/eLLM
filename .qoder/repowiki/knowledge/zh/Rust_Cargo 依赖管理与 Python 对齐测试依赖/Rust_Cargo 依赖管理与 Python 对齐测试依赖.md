---
kind: dependency_management
name: Rust/Cargo 依赖管理与 Python 对齐测试依赖
category: dependency_management
scope:
    - '**'
source_files:
    - Cargo.toml
    - .cargo/config.toml
    - alignment/requirements.txt
---

本仓库采用双语言依赖管理策略：Rust 侧使用 Cargo 单包模式，Python 侧通过 requirements.txt 声明对齐测试所需依赖。

**Rust 依赖（Cargo）**
- 单一 Cargo.toml 集中声明所有运行时、构建期与开发期依赖，未使用 workspace 多 crate 聚合。
- 核心运行时依赖包括异步运行时 tokio、HTTP 框架 axum、CLI 解析 clap、序列化 serde/serde_json/serde_yaml、张量 I/O safetensors、分词器 tiktoken-rs 等。
- 性能相关依赖如 num、itertools、memmap2、core_affinity、num_cpus 用于数值计算与 CPU 亲和性控制。
- 构建期依赖 raw-cpuid 用于在编译时探测 CPU 特性以选择最优内核路径。
- 开发期依赖包含基准测试 criterion、HTTP 客户端 hyper、tower 等。
- .cargo/config.toml 全局配置关闭增量编译并启用 -C target-cpu=native 优化标志。
- 未使用 Cargo.lock 锁定版本；依赖版本均显式指定主版本号或精确小版本，无通配符。
- 未配置私有 registry 或 git 子模块方式引入第三方库。

**Python 依赖（对齐测试）**
- alignment/requirements.txt 仅声明两个依赖：numpy>=1.21.0 和 torch>=1.13.0，用于生成参考输出并与 Rust 实现对比。
- 该文件位于 alignment/ 目录而非根目录，表明 Python 依赖仅服务于算子/模型对齐测试流程，不参与 eLLM 核心构建。

**设计决策**
- Rust 端保持最小化依赖面，避免引入大型 ML 框架，推理引擎完全自实现。
- Python 依赖隔离在对齐测试子工程中，不影响主二进制产物大小与安装复杂度。
- 未使用 vendoring 或 git submodule 管理第三方源码，全部通过 crates.io / PyPI 拉取。