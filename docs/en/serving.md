# Serving Module Overview

---

`src/serving` provides a thin OpenAI-compatible HTTP service wrapper. It does not perform model computation itself; instead, it is responsible for:

* Receiving `/v1/chat/completions` requests
* Allocating batch slots for requests
* Waiting for the external inference runner to finish
* Returning results in streaming or non-streaming format

---

## 1. Module Entry

`serving::run()` is the service entry point:

1. Start the `token_counter.run()` background timer task
2. Build `ApiState` via `build_api_state()`
3. Register routes
4. Bind to `0.0.0.0:8000`
5. Start the Axum HTTP server

The currently exposed endpoints are:

* `POST /v1/chat/completions`
* `GET /status`

---

## 2. State Structure

`ApiState` maintains the following shared objects:

* `batch_sequences`: `Arc<SharedMut<BatchSequence<f16>>>` — holds the tokenizer and token sequence buffer
* `batch_states`: `Arc<SharedMut<Vec<SequenceState>>>` — inference state for each batch slot
* `token_counter`: `Arc<TokenCounter>` — scheduling trigger
* `free_slots`: `Arc<Mutex<VecDeque<usize>>>` — queue of free slot indices
* `available_slots`: `Arc<Semaphore>` — concurrency control semaphore

At startup, `batch_states` is scanned and slots in `Phase::Start` are placed into the free queue, with the semaphore initialized to the same count.

The semaphore and queue are used for slot occupancy management, preventing more than `batch_size` concurrent requests from writing simultaneously.

---

## 3. Request Flow

The `chat_completions()` handler proceeds as follows:

1. Generate a request ID (nanosecond timestamp format: `chatcmpl-{nanos}`)
2. Read the `stream` parameter to decide between streaming and non-streaming responses
3. Call `assign_slot_with_messages()` to acquire a free slot
4. `notifier.notified().await` — async wait for inference completion
5. Read the generated result from `batch_states`, decode via `decode_generated_text()`
6. Call `reclaim_slot()` to release the slot
7. Return an OpenAI-style response

If prompt writing fails (tokenization error), the handler returns `500` immediately.

---

## 4. Slot Allocation Logic

Slot allocation is handled by `assign_slot_with_messages()`:

1. Acquire a permit from the `Semaphore` (backpressure — blocks when all `batch_size` slots are occupied)
2. Pop a free slot index from the `free_slots` queue
3. Call `batch_sequences.write_prompts()` to render the chat template, tokenize, and write into the sequence buffer
4. Set the slot state to `Phase::Prefill`; initialize `sequence_index` and `kv_index` to `0`, set `filling_length`
5. Call `permit.forget()` to detach the permit from RAII — the permit is released manually later by `reclaim_slot()`
6. Call `token_counter.increment(write_len)` to trigger scheduling
7. Return `(slot_index, notifier)`

Slot reclamation is handled by `reclaim_slot(state, slot_index, release_permit)`:

* Reset state back to `Phase::Start`
* Clear `sequence_index` and `kv_index` to sentinel values (`usize::MAX`)
* Reset `filling_length` to 0
* Push the slot back into the `free_slots` queue
* If `release_permit` is `true`, call `available_slots.add_permits(1)` to restore one semaphore permit

---

## 5. Response Format

### Non-streaming

Returns a `ChatCompletionResponse`, structurally compatible with the OpenAI chat completion API:

* `id`: `chatcmpl-{nanos}`
* `object`: `"chat.completion"`
* `created`: Unix timestamp (seconds)
* `model`: the model field from the request
* `choices[0].message`: `{role: "assistant", content: <generated text>}`
* `choices[0].finish_reason`: `"stop"`

### Streaming

Returns an SSE response whose event body is `StreamResponse` (`object: "chat.completion.chunk"`).

The full generated text is split using `split_inclusive(' ')` — each chunk retains its trailing space — and emitted word by word. The last chunk carries `finish_reason: "stop"`. The current streaming path post-processes the fully generated text rather than emitting tokens incrementally during inference.

---

## 6. `/status` Endpoint

`GET /status` returns a simple JSON object for health checks:

```json
{
  "status": "running",
  "mode": "single_threaded_background_processing",
  "info": "Inference and HTTP server run on a single OS thread using current_thread runtime"
}
```

---

## 7. Service Initialization

`initialize_serving_resources(config)` in `resources.rs` initializes all components in order:

1. Load model config (`config.json`) and generation config (`generation_config.json`)
2. Load weights via `SafeTensorsLoader`, write into the global memory pool
3. Extract sampling parameters (top_k, top_k_simd, top_p, min_p, do_sample, eos_token_id_list)
4. Determine thread config (worker_threads = max(total - async_threads, 1), async_threads = 2)
5. Build `BatchSequence` (tokenizer + sequence buffer)
6. Build `batch_states` (one `SequenceState` per slot)
7. Create scheduling components (`BatchScheduler` + `TokenCounter` + broadcast channel)
8. Create RoPE position embeddings, initialize the model, run one forward pass (populates the operator queue)
9. Create `ServingRunner` (takes the operator list from the global queue)

Returns `ServingResources` containing all runtime components.

---

## 8. Current Implementation Traits

* The HTTP layer follows an OpenAI-compatible style
* Inference is driven by an external `ServingRunner`; the serving layer itself does not run model computation
* Slot management uses `Semaphore + VecDeque` for backpressure control
* `temperature` is supported in the request body and written into `batch_temperature` for sampling
* `max_tokens` and `top_p` are preserved in the request body but are not part of the current scheduling logic

---

## 9. Code Entry References

* `src/serving/mod.rs` — HTTP server entry, routes, API data structures
* `src/serving/config.rs` — `ServingConfig` (environment variable reading)
* `src/serving/resources.rs` — `ServingResources` integrated initialization
* `src/serving/model_setup.rs` — model loading, parameter extraction, thread config
* `src/serving/model.rs` — model initialization and forward pass wrapper
* `src/serving/scheduler.rs` — scheduling component creation
* `src/serving/chat_handlers.rs` — `chat_completions` HTTP handler

---

## 10. Streaming: Comparison with vLLM

### vLLM's streaming granularity

vLLM streams **one SSE chunk per generated token** (1 token/chunk). The decode loop pushes each token to the client immediately, resulting in very low time-to-first-token (TTFT). This is true incremental streaming.

### Current eLLM streaming implementation

eLLM reuses the existing `notify: Arc<Notify>` field in `SequenceState` — no new fields are introduced.

**Inference side (`TopKSoftmax::run`)**

After writing each decoded token (including the EOS token), `record.notify.notify_one()` is called immediately. For EOS, `phase` is set to `Phase::Eos` before notifying.

**Serving side (`chat_handlers::build_stream_response`)**

The streaming path no longer waits for a single completion notification. Instead it loops:

1. `notifier.notified().await` — wait for the next token
2. Read `record.sequence_index` (the position just written by `TopKSoftmax`) and `record.phase`
3. Call `batch_sequences.decode_single_token(slot_index, token_index)` to decode the single token
4. Push one SSE chunk immediately
5. If `phase == Eos`, emit the final chunk with `finish_reason: "stop"` and break
6. After the loop, call `reclaim_slot()` to release the slot

The non-streaming path is unchanged: it still waits for the single EOS notification and decodes all generated tokens at once.

### Comparison with vLLM

| | vLLM | eLLM (current) |
|---|---|---|
| Streaming granularity | 1 token/chunk | 1 token/chunk |
| New fields added | — | none (reuses `Notify`) |
| TTFT | very low | very low |
| Non-streaming path | unchanged | unchanged |
