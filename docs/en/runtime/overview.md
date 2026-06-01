# Runtime Module Overview

---

## Table of Contents

1. [Module定位](#1-模块定位)
2. [Architecture Layers](#2-architecture-layers)
3. [Core Components](#3-core-components)
4. [Data Flow](#4-data-flow)
5. [File Structure](#5-file-structure)

---

## 1. Module定位

`src/runtime` is the core runtime module of the eLLM inference execution layer. It transforms user requests into executable computation tasks and coordinates multi-threaded execution.

**Core Responsibilities**:
- **Input Preparation**: Render chat messages to prompts, encode to tokens
- **Batch Scheduling**: Generate current-round computation slices by priority rules
- **Thread Execution**: Manage thread pool to execute operator queues in parallel

---

## 2. Architecture Layers

```mermaid
flowchart TB
    subgraph Serving Layer
        A[chat_completions]
    end

    subgraph Runtime Layer
        B[Input Preparation]
        C[Batch Scheduling]
        D[Thread Execution]
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

| Layer | Responsibility | Key Components |
|-------|---------------|----------------|
| **Input Preparation** | Prompt rendering & token encoding | ChatTemplate, BatchSequence, TokenizerLoader |
| **Batch Scheduling** | Slice generation & task distribution | BatchScheduler, SliceScheduler |
| **Thread Execution** | Operator queue parallel execution | ServingRunner |

---

## 3. Core Components

### 3.1 Component Relationships

```mermaid
classDiagram
    class BatchScheduler {
        -prefill_list: Vec~Vec~SequenceSlice~~
        -decode_list: DecodeList
        -batch_list: Arc~SharedMut~Vec~SequenceState~~~
        -prefill_scheduler: SliceScheduler
        +schedule_batch(): (usize, usize)
        +plan_next_round(): BatchPlan
    }

    class ServingRunner {
        -operator_queue: Vec~Operator~T~~
        -batch_scheduler: BatchScheduler
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

    class DecodeList {
        -slices: Vec~SequenceSlice~
        +push(slice)
        +clear()
        +total_token_count(): usize
    }

    BatchScheduler --> SequenceState
    BatchScheduler --> SequenceSlice
    BatchScheduler --> DecodeList
    ServingRunner --> BatchScheduler
```

### 3.2 Component Overview

| Component | Responsibility | File Location |
|-----------|---------------|---------------|
| `BatchScheduler` | Generate prefill/decode slices | `scheduling/scheduler.rs` |
| `SliceScheduler` | Static allocation for prefill slices | `scheduling/slice_scheduler.rs` |
| `ServingRunner` | Thread pool executor | `runner.rs` |
| `SequenceState` | Batch slot state | `scheduling/state.rs` |
| `SequenceSlice` | Minimal computation unit | `scheduling/sequence_slice.rs` |
| `BatchSequence` | Prompt writing & result decoding | `batch_sequence.rs` |
| `ChatTemplate` | Chat template rendering | `chat_template.rs` |
| `TokenizerLoader` | Tokenizer loading | `tokenizer_loader.rs` |

---

## 4. Data Flow

### 4.1 Request to Execution Flow

```mermaid
sequenceDiagram
    participant Client as Client
    participant Handler as chat_completions
    participant Template as ChatTemplate
    participant Tokenizer as TokenizerLoader
    participant BatchSeq as BatchSequence
    participant Scheduler as BatchScheduler
    participant Runner as ServingRunner
    participant Ops as Operators

    Client->>Handler: POST /chat/completions
    Handler->>Template: render(messages)
    Template-->>Handler: prompt
    Handler->>Tokenizer: encode(prompt)
    Tokenizer-->>Handler: tokens
    Handler->>BatchSeq: write_prompts(slot, tokens)
    BatchSeq->>Scheduler: Update SequenceState
    Handler->>Scheduler: Trigger scheduling
    Scheduler->>Scheduler: schedule_batch()
    Scheduler-->>Runner: prefill_list, decode_list
    Runner->>Ops: Execute operator queue
    Ops->>Ops: Update state
    Ops-->>Handler: Notify completion
    Handler-->>Client: Return response
```

### 4.2 State Transition

```mermaid
stateDiagram-v2
    [*] --> Start
    Start --> Prefill: Write prompts
    Prefill --> Decode: filling_length == 0
    Decode --> Eos: Generate eos token
    Eos --> Start: Release slot
```

---

## 5. File Structure

```
src/runtime/
├── scheduling/
│   ├── scheduler.rs          # BatchScheduler implementation
│   ├── slice_scheduler.rs    # SliceScheduler implementation
│   ├── state.rs              # SequenceState, Phase definitions
│   └── sequence_slice.rs     # SequenceSlice, DecodeList definitions
├── batch_sequence.rs         # BatchSequence implementation
├── chat_template.rs          # ChatTemplate implementation
├── tokenizer_loader.rs       # Tokenizer loading
├── operator.rs               # Operator trait
├── runner.rs                 # ServingRunner implementation
└── mod.rs                    # Module exports
```

---

**Document Version**: v3.0
**Last Updated**: 2026-06-01