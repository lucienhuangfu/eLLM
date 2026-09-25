# Executor Design Details

---

## Table of Contents

1. [Executor Overview](#1-executor-overview)
2. [Core Data Structures](#2-core-data-structures)
3. [Execution Flow](#3-execution-flow)
4. [Synchronization Model](#4-synchronization-model)
5. [Scheduling and Worker Division](#5-scheduling-and-worker-division)
6. [Concurrency Guarantees](#6-concurrency-guarantees)
7. [Implementation Notes](#7-implementation-notes)
8. [File Structure](#8-file-structure)

---

## 1. Executor Overview

`src/runtime/executor` is the runtime execution layer responsible for driving the operator queue on multiple worker threads.

Its role is narrower than the scheduler:

- The scheduler decides `what` to run in the current round.
- The executor coordinates `how` those operators are executed across threads.

The current implementation uses:

- one leader thread for batch scheduling
- `thread_num` blocking worker tasks spawned through Tokio
- a spin-based barrier to keep all workers aligned across operator boundaries
- a shared runtime state object to exchange the active task and batch state

In practice, the executor runs a tight loop:

1. leader thread selects the next batch plan
2. all workers synchronize at a barrier
3. each operator in the queue runs in lockstep across all workers
4. leader clears the work flag and the loop repeats

---

## 2. Core Data Structures

### 2.1 `ExecutorPool<T>`

`ExecutorPool` is the top-level execution controller.

```rust
pub struct ExecutorPool<T> {
    shared_state: Arc<SharedState>,
    operator_queue: Arc<[Operator<T>]>,
    thread_num: usize,
    handles: Vec<JoinHandle<()>>,
    strategy: Arc<dyn SchedulerStrategy>,
    timeout: Duration,
}
```

| Field | Purpose |
|-------|---------|
| `shared_state` | Shared runtime state for scheduler and workers |
| `operator_queue` | Ordered operator pipeline executed for each batch |
| `thread_num` | Number of worker threads participating in execution |
| `handles` | Tokio join handles for spawned blocking workers |
| `strategy` | Pluggable scheduling strategy |
| `timeout` | Idle sleep interval used when no batch is ready |

### 2.2 `SharedState`

`ExecutorPool` depends on `SharedState` to pass the current task and batch list to workers.

Important fields:

- `batch_list`: shared mutable slot state list
- `has_work`: atomic flag indicating whether a task is ready
- `current_task`: the active `ScheduleTask`

The executor treats `SharedState` as the single coordination point between scheduling and execution.

### 2.3 `ScheduleTask`

The task carried into each execution round comes from the scheduler and includes:

- `prefill_size`
- `decode_size`
- `prefill_list`
- `decode_list`
- `task_id`

The executor does not compute the plan itself; it only consumes it.

---

## 3. Execution Flow

### 3.1 Startup

`ExecutorPool::start()` creates a `SpinBarrier` with `thread_num` participants, then spawns one blocking task per worker.

Each spawned worker receives:

- a clone of `SharedState`
- the operator queue
- the barrier
- the worker id
- the shared scheduler strategy
- the idle timeout

### 3.2 Worker Loop

The worker logic is implemented in `run_worker()`.

```text
loop:
    if worker_id == 0:
        plan next round
        if no work:
            sleep(timeout)
            continue
        write task into SharedState
    else:
        spin until SharedState.has_work == true

    barrier.wait()
    read active ScheduleTask
    execute_batch()
    barrier.wait()

    if worker_id == 0:
        clear SharedState.has_work
```

### 3.3 Execution Phases

Each batch round is split into three phases:

1. **Scheduling phase**
   - only worker 0 runs the strategy
   - `BatchPlan` is converted into `ScheduleTask`

2. **Execution phase**
   - all workers enter `execute_batch()`
   - every operator runs once per round

3. **Completion phase**
   - workers synchronize again
   - worker 0 clears the work flag

This design keeps all workers aligned at operator boundaries and avoids one worker advancing into the next operator early.

---

## 4. Synchronization Model

### 4.1 `SpinBarrier`

The executor uses `SpinBarrier` from `sync.rs` to synchronize worker progress.

```rust
pub struct SpinBarrier {
    count: AtomicUsize,
    generation: AtomicU64,
    num_threads: usize,
}
```

The barrier implements generation-based synchronization:

- each round increments the barrier generation
- the last arriving thread resets the counter
- other threads spin until the generation changes

### 4.2 Barrier Usage in the Executor

The barrier is used at three levels:

- before the batch begins
- before each operator runs
- after each operator completes
- after the full operator queue finishes

This ensures all workers:

- start the same batch together
- observe the same operator ordering
- leave the batch together

### 4.3 `AdaptiveWait`

`AdaptiveWait` is a reusable wait helper with the same backoff pattern as the barrier:

- spin first
- then yield
- then sleep with exponential backoff

It is available from `sync.rs`, but the current executor path primarily relies on `SpinBarrier` and direct atomic polling.

---

## 5. Scheduling and Worker Division

### 5.1 Leader-Follower Model

The executor uses a fixed leader-follower split:

- **Worker 0** acts as the leader
- **Workers 1..N-1** wait for the `has_work` signal

Only the leader:

- calls the scheduler strategy
- writes the active task into shared state
- clears the work flag after the batch completes

### 5.2 Why This Split Exists

This reduces contention in the hot path:

- one thread performs planning
- all threads consume the same immutable task view
- no extra coordination is needed to decide who schedules next

### 5.3 Operator Queue Execution

`execute_batch()` runs the operator queue in order.

For each operator:

1. workers synchronize at the barrier
2. each worker obtains the mutable `batch_list`
3. the operator updates the batch state for its worker partition
4. workers synchronize again before the next operator

This makes the operator pipeline behave like a staged SIMD-style execution model across the batch.

---

## 6. Concurrency Guarantees

### 6.1 Shared State Access

The executor depends on the following invariants:

- the active task is written once by the leader before execution starts
- workers only read the task during a batch round
- `has_work` is the coarse-grained readiness flag
- `batch_list` is mutated by operators through the shared mutable wrapper

### 6.2 Safety Boundaries

The executor uses `unsafe` only at the point where it converts the shared batch list pointer into `&mut Vec<SlotState>`.

That is safe only because:

- workers are synchronized by the barrier
- each operator is expected to partition its work by `thread_id`
- the shared mutable wrapper ensures the runtime owns the aliasing discipline

### 6.3 Thread Count Handling

`thread_num` is clamped to at least `1` in the constructor and in `with_thread_count()`.

This avoids:

- zero-participant barriers
- zero-capacity join handle vectors
- divide-by-zero style errors in scheduling assumptions

---

## 7. Implementation Notes

### 7.1 Current Behavior

The current executor implementation is intentionally simple:

- it does not own request ingestion
- it does not own task queue buffering
- it does not implement worker stealing
- it does not perform load balancing between operators

Instead, it executes a single scheduler-produced batch plan at a time.

### 7.2 Idle Waiting

When worker 0 finds no available batch plan, it sleeps for `timeout` and retries.

This is a coarse fallback that avoids busy looping when the system is idle.

### 7.3 Logging

The executor prints simple lifecycle messages:

- worker startup
- task scheduling details

These logs are useful for debugging but are not a replacement for structured tracing.

### 7.4 Limitations

The current implementation has a few important constraints:

- `start()` only spawns workers; it does not return a shutdown handle
- worker loops are long-lived and currently do not include explicit stop signaling
- worker 0 is a coordination bottleneck by design
- the busy-spin wait on non-leader workers is efficient under load, but can burn CPU while idle

These are acceptable for a tight execution loop, but they are worth revisiting if the runtime needs graceful shutdown or lower idle power usage.

---

## 8. File Structure

```text
src/runtime/executor/
├── mod.rs       # Submodule entry and public re-exports
├── executor.rs  # ExecutorPool and batch execution loop
└── sync.rs      # SpinBarrier and adaptive wait primitives
```

### Related Modules

- `src/runtime/scheduler/strategy.rs` - scheduling strategy trait and default strategy
- `src/runtime/state/shared.rs` - shared execution state
- `src/runtime/plan.rs` - batch plan generation
- `src/runtime/state/core.rs` - slot state definitions used by operators

---

**Document Version**: v1.0  
**Last Updated**: 2026-07-13  
**Major Changes**: Added executor architecture, batch execution flow, synchronization model, and implementation notes aligned with `src/runtime/executor/executor.rs` and `src/runtime/executor/sync.rs`

