# CLI Reference

eLLM 命令行接口提供了与 vLLM 兼容的配置选项，同时支持额外的 eLLM 特有功能。

## 命令概览

| 命令 | 说明 |
|------|------|
| `serve` | 启动 OpenAI 兼容的推理服务 |
| `chat` | 启动交互式聊天模式 |
| `complete` | 执行文本补全任务 |
| `run-batch` | 批量处理推理任务 |

## 通用参数

### `--config`

从配置文件读取 CLI 选项。必须是 YAML 格式的文件，支持 vLLM serve_args 格式。

```bash
ellm serve --config config.yaml
```

配置文件示例 (`config.yaml`)：
```yaml
model: /path/to/model
dtype: bf16
max_model_len: 8192
host: 0.0.0.0
port: 8080
```

### `--json-arg`

传递 JSON 格式的配置参数。支持两种格式：

**格式一：完整 JSON 对象**
```bash
ellm serve --json-arg '{"model": {"model": "/path/to/model"}, "serve": {"port": 8080}}'
```

**格式二：键值对**
```bash
ellm serve --json-arg model.model=/path/to/model --json-arg serve.port=8080
```

### 参数缩写

支持以下缩写参数：

| 缩写 | 完整参数 | 说明 |
|------|----------|------|
| `-p` | `--min-p` | 最小概率值 |
| `-m` | `--model` | 模型名称或路径 |
| `-d` | `--dtype` | 数据类型 |
| `-q` | `--quantization` | 量化方法 |
| `-s` | `--seed` | 随机种子 |
| `-H` | `--host` | 主机地址 |
| `-P` | `--port` | 端口号 |
| `-S` | `--stream` | 流式输出 |

### 配置优先级

配置项按以下优先级生效（从低到高）：

1. 配置文件 (`--config`)
2. CLI 参数
3. JSON 参数 (`--json-arg`)

## 使用示例

```bash
# 基本用法
ellm serve -m /models/qwen3-7b -d bf16 -H 0.0.0.0 -P 8080

# 使用配置文件
ellm serve --config config.yaml

# 混合使用（CLI 参数覆盖配置文件）
ellm serve --config config.yaml -P 9090

# 使用 JSON 参数（最高优先级）
ellm serve --config config.yaml --json-arg serve.port=9090
```