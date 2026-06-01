# Runtime 模块总览

---

## 目录

1. [模块定位](#1-模块定位)
2. [架构层次](#2-架构层次)
3. [核心组件](#3-核心组件)
4. [数据流](#4-数据流)
5. [文件结构](#5-文件结构)

---

## 1. 模块定位

`src/runtime` 是 eLLM 推理执行层的核心运行时模块，负责将用户请求转换为可执行的计算任务，并协调多线程执行。

**核心职责**：
- **输入准备**：将聊天消息渲染为 prompt，编码为 token
- **批次调度**：按优先级规则生成本轮计算切片
- **线程执行**：管理线程池并行执行算子队列

---

## 2. 架构层次

```mermaid
flowchart TB
    subgraph Serving Layer
        A[chat_completions]
    end

    subgraph Runtime Layer
        B[输入准备]
        C[批次调度]
        D[线程执行]
    end

    subgraph Operators Layer
        E[Attention]
        F[MatMul]
        G[TopKSoftmax]
    end

    A --> B
    B --> C
    C --> D
    D --> E
    D --> F
    D --> G
```

| 层次 | 职责 | 关键组件 |
|------|------|----------|
| **输入准备** | Prompt 渲染与 Token 编码 | ChatTemplate, BatchSequence, TokenizerLoader |
| **批次调度** | 切片生成与任务分发 | BatchScheduler, TokenCounter |
| **线程执行** | 算子队列并行执行 | ServingRunner |

---

## 3. 核心组件

### 3.1 组件关系

```mermaid
classDiagram
    class BatchScheduler {
        -prefill_list: Vec~Vec~SequenceSlice~~
        -decode_list: DecodeList
        -batch_list: Arc~SharedMut~Vec~SequenceState~~~
        -prefill_scheduler: SliceScheduler
        +schedule_batch(): (usize, usize)
    }

    class TokenCounter {
        -current_tokens: AtomicUsize
        -threshold: usize
        -timeout: Duration
        -broadcast_sender: Sender~ScheduleTask~
        +increment(count)
        +run()
    }

    class ServingRunner {
        -operator_queue: Vec~Operator~T~~
        -batch_list: Arc~SharedMut~Vec~SequenceState~~~
        -task_sender: Sender~ScheduleTask~
        +start()
    }

    class SequenceState {
        +phase: Phase
        +sequence_index: usize
        +kv_index: usize
        +filling_length: usize
        +notify: Arc~Notify~
    }

    class SequenceSlice {
        +batch_index: usize
        +sequence_index: usize
        +token_start_index: usize
        +length: usize
        +last_token_flag: bool
    }

    class ScheduleTask {
        +prefill_size: usize
        +decode_size: usize
        +prefill_list: Vec~Vec~SequenceSlice~~
        +decode_list: DecodeList
        +timestamp: Instant
        +task_id: u64
    }

    BatchScheduler --> SequenceState
    BatchScheduler --> SequenceSlice
    BatchScheduler --> DecodeList
    TokenCounter --> BatchScheduler
    TokenCounter --> ScheduleTask
    ServingRunner --> ScheduleTask
    ServingRunner --> SequenceState
```

### 3.2 组件说明

| 组件 | 职责 | 文件位置 |
|------|------|----------|
| `BatchScheduler` | 生成本轮 prefill/decode 切片 | `scheduling/scheduler.rs` |
| `TokenCounter` | 统计 token 并触发调度 | `scheduling/token_counter.rs` |
| `ServingRunner` | 广播订阅式线程池执行器 | `runner.rs` |
| `SequenceState` | Batch 槽位状态 | `scheduling/state.rs` |
| `SequenceSlice` | 最小计算单元 | `scheduling/sequence_slice.rs` |
| `ScheduleTask` | 调度任务载体 | `scheduling/task.rs` |
| `BatchSequence` | Prompt 写入与结果解码 | `batch_sequence.rs` |
| `ChatTemplate` | 聊天模板渲染 | `chat_template.rs` |
| `TokenizerLoader` | Tokenizer 加载 | `tokenizer_loader.rs` |

---

## 4. 数据流

### 4.1 请求到执行流程

```mermaid
sequenceDiagram
    participant Client as 客户端
    participant Handler as chat_completions
    participant Template as ChatTemplate
    participant Tokenizer as TokenizerLoader
    participant BatchSeq as BatchSequence
    participant Counter as TokenCounter
    participant Scheduler as BatchScheduler
    participant Runner as ServingRunner
    participant Ops as Operators

    Client->>Handler: POST /chat/completions
    Handler->>Template: render(messages)
    Template-->>Handler: prompt
    Handler->>Tokenizer: encode(prompt)
    Tokenizer-->>Handler: tokens
    Handler->>BatchSeq: write_prompts(slot, tokens)
    Handler->>Counter: increment(write_len)
    Counter->>Scheduler: schedule_batch()
    Scheduler-->>Counter: prefill_list, decode_list
    Counter-->>Runner: broadcast ScheduleTask
    Runner->>Ops: 执行算子队列
    Ops->>Ops: 更新状态
    Ops-->>Handler: 通知完成
    Handler-->>Client: 返回响应
```

### 4.2 状态流转

```mermaid
stateDiagram-v2
    [*] --> Start
    Start --> Prefill: 写入 prompts
    Prefill --> Decode: filling_length == 0
    Decode --> Eos: 生成 eos token
    Eos --> Start: 释放槽位
```

---

## 5. 文件结构

```
src/runtime/
├── scheduling/
│   ├── scheduler.rs          # BatchScheduler 实现
│   ├── token_counter.rs      # TokenCounter 实现
│   ├── task.rs               # ScheduleTask 定义
│   ├── slice_scheduler.rs    # SliceScheduler 实现
│   ├── state.rs              # SequenceState, Phase 定义
│   └── sequence_slice.rs     # SequenceSlice, DecodeList 定义
├── batch_sequence.rs         # BatchSequence 实现
├── io/
│   ├── chat_template.rs      # ChatTemplate 实现
│   ├── tokenizer_loader.rs   # Tokenizer 加载
│   └── safetensors_loader.rs # 权重读取
├── runner.rs                 # ServingRunner 实现
└── mod.rs                    # 模块导出
```

---

**文档版本**: v2.0  
**最后更新**: 2026-06-01
