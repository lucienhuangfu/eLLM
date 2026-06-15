# `chat`

`ellm chat` 命令用于启动交互式聊天模式，与模型进行对话。

## 基本用法

```bash
# 基本聊天模式
ellm chat -m /models/qwen3-7b

# 使用流式输出
ellm chat -m /models/qwen3-7b -S

# 使用配置文件
ellm chat --config config.yaml
```

## 参数

### 通用参数

`chat` 命令支持所有通用参数，请参阅 [CLI Reference](index.md)。

### 聊天专用参数

#### `--system-prompt`

设置系统提示词，用于引导模型的行为和风格。

```bash
ellm chat -m /models/qwen3-7b --system-prompt "你是一个乐于助人的助手。"
```

#### `-S`, `--stream`

启用流式输出模式，实时显示模型的回复。

```bash
ellm chat -m /models/qwen3-7b -S
```

#### `--max-turns`

设置最大对话轮数。达到限制后自动结束对话。

```bash
ellm chat -m /models/qwen3-7b --max-turns 10
```

## 使用示例

```bash
# 启动交互式聊天
ellm chat -m /models/qwen3-7b -d bf16 -S

# 使用配置文件
ellm chat --config config.yaml

# 设置系统提示词和最大轮数
ellm chat -m /models/qwen3-7b --system-prompt "你是一个专业的技术顾问。" --max-turns 5
```

## 交互命令

在聊天模式下，支持以下命令：

| 命令 | 说明 |
|------|------|
| `/exit` | 退出聊天 |
| `/reset` | 重置对话历史 |
| `/help` | 显示帮助信息 |

## 配置文件示例

```yaml
model: /models/qwen3-7b
dtype: bf16
max_model_len: 8192

system_prompt: "你是一个乐于助人的助手。"
stream: true
max_turns: 20
```