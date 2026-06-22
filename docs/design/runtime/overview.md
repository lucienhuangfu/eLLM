# Runtime Module Overview

---

## Table of Contents

1. [Module Overview](#1-module-overview)
2. [Architecture Layers](#2-architecture-layers)
3. [Core Components](#3-core-components)
4. [Data Flow](#4-data-flow)
5. [File Structure](#5-file-structure)

---

## 1. Module Overview

`src/runtime` is the core runtime module of the eLLM inference execution layer. It transforms user requests into executable computation tasks and coordinates multi-threaded execution.

**Core Responsibilities**:
- **Input Preparation**: Render chat messages to prompts, encode to tokens
- **Batch Scheduling**: Generate current-round computation slices by priority rules
- **Thread Execution**: Manage thread pool to execute operator queues in parallel
- **Session Management**: Unified dialogue session management with reusable/non-reusable modes

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
        E[Session Management]
    end

    subgraph Operators Layer
        F[Attention]
        G[MatMul]
        H[TopKSoftmax]
    end

    A --> B
    B --> C
    B --> E
    C --> D
    D --> F
    D --> G
    D --> H
    E --> C
```

| Layer | Responsibility | Key Components |
|-------|---------------|----------------|
| **Input Preparation** | Prompt rendering & token encoding | ChatTemplate, BatchSequence, TokenizerLoader |
| **Batch Scheduling** | Slice generation & task distribution | Scheduler, SchedulerStrategy, SessionManager |
| **Thread Execution** | Operator queue parallel execution | ExecutorPool, SpinBarrier |
| **Session Management** | Unified session lifecycle management | SessionManager, SlotAllocator, DialogueSession |

---

## 3. Core Components

### 3.1 Component Relationships

```mermaid
classDiagram
    class Scheduler {
        -prefill_list: Vec~Vec~SequenceSlice~~
        -decode_list: DecodeList
        -batch_list: Arc~SharedMut~Vec~SequenceState~~~
        -strategy: SchedulerStrategy
        +schedule_batch(): (usize, usize)
        +plan_next_round(): BatchPlan
        +run()
    }

    class ExecutorPool {
        -operator_queue: Vec~Operator~T~~
        -shared_state: Arc~SharedState~
        +start()
        +execute_single_thread_batch()
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

    class SequenceStateMachine {
        +transition_to_prefill()
        +transition_to_decode()
        +transition_to_eos()
        +advance_sequence()
    }

    class SessionManager {
        +acquire_session(session_id, mode)
        +release_session(session_id, token_count)
        +calculate_delta(session_id, new_tokens)
        +get_cached_tokens(session_id)
    }

    class SlotAllocator {
        +allocate()
        +allocate_preferred(slot_index)
        +release(slot_index)
        +cancel_timer(slot_index)
    }

    class DialogueSession {
        +session_id: String
        +mode: SessionMode
        +slot_index: Option<usize>
        +token_count: usize
        +is_active: bool
    }

    Scheduler --> SequenceState
    Scheduler --> SequenceSlice
    Scheduler --> DecodeList
    Scheduler ..> SchedulerStrategy : uses
    ExecutorPool --> SharedState
    SequenceStateMachine ..> SequenceState : operates on
    SessionManager --> SlotAllocator
    SessionManager --> DialogueSession
```

### 3.2 Component Overview

| Component | Responsibility | File Location |
|-----------|---------------|---------------|
| `Scheduler` | Core scheduling logic, event-driven execution | `scheduling/scheduler.rs` |
| `SchedulerStrategy` | Scheduling strategy trait | `scheduling/strategy.rs` |
| `DefaultSchedulerStrategy` | Default scheduling implementation | `scheduling/strategy.rs` |
| `SliceScheduler` | Prefill token distribution across threads | `scheduling/strategy.rs` |
| `ExecutorPool` | Leader-follower thread pool executor | `executor/executor.rs` |
| `SequenceState` | Batch slot state | `scheduling/types.rs` |
| `SequenceStateMachine` | State transition logic | `scheduling/state_machine.rs` |
| `SequenceSlice` | Minimal computation unit | `scheduling/sequence_slice.rs` |
| `ScheduleTask` | Scheduling task carrier | `scheduling/types.rs` |
| `BatchSequence` | Prompt writing & result decoding | `scheduling/batch_sequence.rs` |
| `SessionManager` | Unified session lifecycle management | `scheduling/session.rs` |
| `SlotAllocator` | Simplified slot allocation | `scheduling/slot_allocator.rs` |
| `DialogueSession` | Session metadata structure | `scheduling/session.rs` |
| `ChatTemplate` | Chat template rendering | `io/chat_template.rs` |
| `TokenizerLoader` | Tokenizer loading | `io/tokenizer_loader.rs` |

---

## 4. Data Flow

### 4.1 Request to Execution Flow

```mermaid
sequenceDiagram
    participant Client as Client
    participant Handler as chat_completions
    participant Template as ChatTemplate
    participant Tokenizer as TokenizerLoader
    participant SessionMgr as SessionManager
    participant BatchSeq as BatchSequence
    participant Scheduler as Scheduler
    participant Runner as ExecutorPool
    participant Ops as Operators

    Client->>Handler: POST /chat/completions
    Handler->>Template: render(messages)
    Template-->>Handler: prompt
    Handler->>Tokenizer: encode(prompt)
    Tokenizer-->>Handler: tokens
    
    Handler->>SessionMgr: acquire_session(session_id, mode)
    alt Session can be reused
        SessionMgr-->>Handler: SessionHandle { is_reused: true }
        Handler->>SessionMgr: calculate_delta(session_id, tokens)
        Handler->>BatchSeq: write_tokens(delta_tokens)
    else New session
        SessionMgr-->>Handler: SessionHandle { is_reused: false }
        Handler->>BatchSeq: write_prompts(all_tokens)
    end
    
    BatchSeq->>Scheduler: Update SequenceState
    Handler->>Scheduler: notify_tokens(count)
    Scheduler->>Scheduler: trigger_schedule()
    Scheduler->>Scheduler: schedule_batch()
    Scheduler-->>Runner: ScheduleTask (prefill_list, decode_list)
    Runner->>Ops: Execute operator queue
    Ops->>Ops: Update state via SequenceStateMachine
    Ops-->>Handler: Notify completion
    Handler->>SessionMgr: release_session(session_id, token_count)
    Handler-->>Client: Return response
```

### 4.2 State Transition

```mermaid
stateDiagram-v2
    [*] --> Start
    Start --> Prefill: write_prompts / transition_to_prefill
    Prefill --> Prefill: advance_sequence (incomplete)
    Prefill --> Decode: advance_sequence (filling_length==0)
    Decode --> Decode: advance_sequence (incomplete)
    Decode --> Eos: transition_to_eos
    Decode --> Timeout: transition_to_timeout
    Prefill --> Eos: transition_to_eos
    Prefill --> Timeout: transition_to_timeout
    Eos --> Start: reset_to_start
    Timeout --> Start: reset_to_start
```

---

## 5. File Structure

```
src/runtime/
├── scheduling/
│   ├── mod.rs                # Scheduling submodule entry and re-exports
│   ├── scheduler.rs          # Scheduler implementation
│   ├── strategy.rs           # SchedulerStrategy, BatchPlan, DefaultSchedulerStrategy, SliceScheduler
│   ├── types.rs              # Phase, ScheduleTask, SequenceState definitions
│   ├── state_machine.rs      # SequenceStateMachine state transition logic
│   ├── sequence_slice.rs     # SequenceSlice, DecodeList definitions
│   ├── batch_sequence.rs     # BatchSequence implementation
│   ├── session.rs            # SessionManager, DialogueSession, SessionHandle, SessionMode
│   ├── slot_allocator.rs     # SlotAllocator implementation
│   └── initialization.rs     # build_batch_sequence, build_sequence_state helpers
├── executor/
│   ├── mod.rs                # Executor submodule entry
│   ├── executor.rs           # ExecutorPool implementation
│   ├── plan.rs               # BatchPlan, PlanBuilder
│   └── sync.rs               # SpinBarrier, BatchTracker synchronization primitives
├── io/
│   ├── mod.rs                # IO submodule entry
│   ├── chat_template.rs      # ChatTemplate implementation
│   ├── tokenizer_loader.rs   # Tokenizer loading (load_tiktoken)
│   ├── safetensors_loader.rs # Weight loading (SafeTensorsLoader)
│   └── from_safetensors.rs   # FromSafetensors trait (type conversion)
├── error.rs                  # Runtime error definitions
└── mod.rs                    # Module exports
```

---

**Document Version**: v4.1  
**Last Updated**: 2026-06-17  
**Major Changes**: Added slot delayed recycling mechanism and preferred slot reuse strategy