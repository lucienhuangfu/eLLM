# Fake Echo Operator

This document explains what `FakeEcho` is for, when it runs, and how the current implementation behaves.

Related code:

* `src/operators/fake_echo.rs`
* `src/runtime/operator.rs`
* `src/bin/fake_server.rs`

---

## 1. What It Does

`FakeEcho` is a test operator designed for integration debugging and end-to-end wiring checks.

Its goal is not to perform real model computation. Instead, it proves that the runtime path really executed an operator and that the operator produced a visible change to request state.

In the `fake_server` scenario, it can replace the complex forward pass of a real model so you can quickly verify the following:

1. Whether `ServingRunner` can schedule the operator queue correctly
2. Whether `batch_list` state advances from `Prefill` to `Eos`
3. Whether the waiting request's `Notify` gets woken up correctly

---

## 2. Execution Condition

`FakeEcho` runs only when `thread_id == 0`.

That is because thread 0 in the current `ServingRunner` takes responsibility for scheduling progress, and `FakeEcho` follows the same convention:

* Non-zero threads return immediately
* Only thread 0 iterates over `batch_list`

This has three benefits:

1. Deterministic behavior
2. Easier debugging
3. No extra cross-thread synchronization logic

---

## 3. Current Behavior

`FakeEcho` only handles records in `Phase::Prefill`.

If a slot is in `Prefill`, it does three things:

1. Read the slot's `filling_length`
2. Advance the slot to `Phase::Eos`
3. Wake the waiting request

### 3.1 State Changes

After processing, `FakeEcho` changes the slot state to:

* `sequence_index = 0`
* `kv_index = filling_length`
* `filling_length = 0`
* `phase = Phase::Eos`

Then it calls `notify.notify_one()` to wake the waiting request.

---

## 4. Relation to Real Serving

`FakeEcho` mainly serves `src/bin/fake_server.rs`.

The `fake_server` flow is:

1. Create `BatchSequence`
2. Create `BatchScheduler`
3. Put `FakeEcho` into the `ServingRunner` operator queue
4. Start the runtime thread and HTTP server

This means that after a request enters the system, it does not go through a real model forward pass, but it still experiences the full serving lifecycle:

1. Request is written into the token buffer
2. Slot enters `Prefill`
3. `ServingRunner` schedules `FakeEcho`
4. `FakeEcho` ends the request
5. The HTTP layer reads the generated result and returns it

---

## 5. Best Use Cases

`FakeEcho` is suitable for:

1. Checking whether serving and runtime are connected correctly
2. Verifying `Notify`, `Phase`, and slot reclamation logic
3. Debugging request lifecycle behavior
4. Performing minimal integration tests without real model weights

It is not suitable for numerical correctness validation, and it is not a substitute for tests of the real operators.

---

## 6. Code Entry

### 6.1 Operator Definition

`src/operators/fake_echo.rs` defines the `FakeEcho` implementation.

### 6.2 Runtime Dispatch

`src/runtime/operator.rs` dispatches to `FakeEcho::run()` via `Operator::FakeEcho`.

### 6.3 Fake Server

`src/bin/fake_server.rs` inserts `FakeEcho` into the operator queue so that it can complete `Prefill` requests directly in the runtime.

---

## 7. Design Notes

The current implementation is a test-friendly fake operator, not a general business operator.

Keep in mind:

1. It only cares about `Prefill` and does not implement real decode
2. It does not write the token buffer directly, so it is better suited to request-lifecycle integration tests
3. Its behavior is intentionally simplified; the goal is "make it obvious that it ran," not "generate text like a real model"
