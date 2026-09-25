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
- **Session Management**: Unified dialogue session management with reusable/non-reusable modes and delayed slot recycling via SlotManager
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
| **Batch Scheduling** | Slice generation & task distribution | Scheduler, SchedulerStrategy, PlanBuilder |
| **Thread Execution** | Operator queue parallel execution | ExecutorPool, SpinBarrier |
| **Session Management** | Unified session lifecycle management | SlotManager, DialogueSession, SessionHandle |

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
| `Scheduler` | Core scheduling logic, event-driven execution with broadcast task distribution | `scheduler/core.rs` |
| `SchedulerStrategy` | Scheduling strategy trait | `scheduler/strategy.rs` |
| `DefaultSchedulerStrategy` | Default scheduling implementation delegating to PlanBuilder | `scheduler/strategy.rs` |
| `PlanBuilder` | Batch plan construction with slice distribution | `plan.rs` |
| `SliceScheduler` | Prefill token distribution across threads | `plan.rs` |
| `ExecutorPool` | Multi-thread executor with SpinBarrier synchronization | `executor/executor.rs` |
| `SpinBarrier` | Generation-based synchronization barrier for worker alignment | `executor/sync.rs` |
| `AdaptiveWait` | Adaptive backoff waiting helper | `executor/sync.rs` |
| `SlotState` | Slot state tracking with LRU pointers, phase, sequence, KV cache | `state/core.rs` |
| `SlotStateMachine` | State transition logic with validation | `state/machine.rs` |
| `SequenceSlice` | Minimal computation unit | `state/sequence.rs` |
| `DecodeList` | Decode slice collection with lookup and iteration utilities | `state/sequence.rs` |
| `DecodeLookupResult` | Result type for global index lookup | `state/sequence.rs` |
| `ScheduleTask` | Scheduling task carrier with timestamp | `scheduler/task.rs` |
| `BatchSequence` | Prompt writing & result decoding with tokenizer integration | `state/batch.rs` |
| `SharedState` | Shared state for scheduler-executor coordination | `state/shared.rs` |
| `SlotManager` | Unified slot and session management with LRU and delayed recycling | `session/slot_manager.rs` |
| `DialogueSession` | Session metadata structure | `session/types.rs` |
| `ChatTemplate` | Chat template rendering with MiniJinja | `io/chat_template.rs` |
| `TokenizerLoader` | Tokenizer loading from HuggingFace format | `io/tokenizer_loader.rs` |
| `SafeTensorsLoader` | Weight loading from SafeTensors format | `io/safetensors_loader.rs` |

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
├── scheduler/
│   ├── mod.rs                # Scheduler submodule entry
│   ├── core.rs               # Scheduler implementation
│   ├── strategy.rs           # SchedulerStrategy trait and DefaultSchedulerStrategy
│   └── task.rs               # ScheduleTask definition
├── session/
│   ├── mod.rs                # Session submodule entry
│   ├── slot_manager.rs       # SlotManager with LRU and session tracking
│   └── types.rs              # SessionMode, SessionHandle, DialogueSession
├── state/
│   ├── mod.rs                # State submodule entry
│   ├── core.rs               # SlotState definition with LRU pointers
│   ├── machine.rs            # SlotStateMachine state transitions
│   ├── types.rs              # Phase enum
│   ├── sequence.rs           # SequenceSlice, DecodeList, DecodeLookupResult
│   ├── batch.rs              # BatchSequence implementation
│   ├── shared.rs             # SharedState for cross-component sharing
│   └── state_init.rs         # build_batch_sequence, build_slot_state helpers
├── executor/
│   ├── mod.rs                # Executor submodule entry
│   ├── executor.rs           # ExecutorPool implementation
│   └── sync.rs               # SpinBarrier and AdaptiveWait primitives
├── io/
│   ├── mod.rs                # IO submodule entry
│   ├── chat_template.rs      # ChatTemplate implementation
│   ├── tokenizer_loader.rs   # Tokenizer loading (load_tiktoken)
│   ├── safetensors_loader.rs # Weight loading (SafeTensorsLoader)
│   └── from_safetensors.rs   # FromSafetensors trait (type conversion)
├── plan.rs                   # BatchPlan, PlanBuilder, SliceScheduler, PrefillCandidate
├── error.rs                  # Runtime error definitions
└── mod.rs                    # Module exports
```

---

**Document Version**: v5.2  
**Last Updated**: 2026-06-22  
**Major Changes**: Updated component list with SharedState, DecodeList, DecodeLookupResult; corrected session module file structure (removed slot_entry.rs); updated SlotState description with LRU pointers; added SafeTensorsLoader component; updated responsibility descriptions to match actual implementation
