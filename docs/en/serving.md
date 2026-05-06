# Serving Module Overview

---

`src/serving` provides a thin OpenAI-compatible HTTP service wrapper. It does not perform model computation itself; instead, it is responsible for:

* Receiving `/v1/chat/completions` requests
* Allocating batch slots for requests
* Waiting for the external inference runner to finish
* Returning results in streaming or non-streaming format

The current implementation is explicitly single-thread driven and does not do multithreaded concurrent writes. The design intent is not just to stay "simple"; rather, the single-thread model is already enough for the current workload, so HTTP only needs to serially receive, write, and return data without introducing extra synchronization overhead.

---

## 1. Module Entry

`serving::run()` is the service entry point:

1. Initialize `AppState`
2. Register routes
3. Bind to `0.0.0.0:8000`
4. Start the Axum HTTP server

The serving side runs under a single-thread background processing model, so request handling and slot writes both proceed serially.

The currently exposed endpoints are:

* `POST /v1/chat/completions`
* `GET /status`

---

## 2. State Structure

`AppState` maintains two core shared objects:

* `batch_sequences`
* `batch_list`

It also maintains slot management structures:

* `free_slots`: queue of free slots
* `available_slots`: semaphore used to control how many slots are available

At startup, it scans `batch_list` and places slots in `Phase::Start` into the free queue.

The semaphore and queue are mainly used for slot occupancy management, not to parallelize the write path; actual writes are still completed on a single thread because that is sufficient for the current service scenario.

---

## 3. Request Flow

The `chat_completions()` handler can be summarized as:

1. Generate a request ID
2. Read the `stream` parameter to decide between streaming and non-streaming responses
3. Acquire a free slot
4. Write the request messages into the batch position
5. Switch the slot state to `Phase::Prefill`
6. Wait for the external inference completion signal
7. Read the generated text and reclaim the slot
8. Return an OpenAI-style response

If prompt writing fails, the handler returns `500` immediately.

---

## 4. Slot Allocation Logic

Slot allocation is handled by `assign_slot_with_messages()`:

* First acquire a permit from the semaphore
* Then pop an index from the `free_slots` queue
* Then write messages and update `SequenceState`

On success, it returns a `Notify` handle to the handler so the handler can wait for inference completion on that slot.

Slot reclamation is handled by `reclaim_slot()`:

* Reset the state back to `Phase::Start`
* Clear `sequence_index` and `kv_index` to sentinel values
* Reset `filling_length`
* Put the slot back into the free queue
* Return the semaphore permit when needed

---

## 5. Response Format

### Non-streaming

Returns a `ChatCompletionResponse`, structurally similar to an OpenAI chat completion response:

* `id`
* `object = "chat.completion"`
* `created`
* `model`
* `choices[0].message`

### Streaming

Returns an SSE response whose event body is `StreamResponse`.

In the current implementation, the full generated text is split by spaces and emitted chunk by chunk, so the streaming path is closer to "sending the final text in pieces" than to incremental token generation during inference.

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

## 7. Current Implementation Traits

* The HTTP layer follows an OpenAI-compatible style
* Request handling and batch writes are both executed serially on one thread
* The server itself does not run model inference
* Inference is driven by an external runner
* The current mode is single-thread background processing
* `temperature`, `max_tokens`, and `top_p` are preserved in the request body but are not part of the current scheduling logic

---

## 8. Code Entry References

* `src/serving/mod.rs`
* `src/serving/bootstrap.rs`
* `src/serving/handlers.rs`
* `src/serving/types.rs`
