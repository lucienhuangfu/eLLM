//! Fake echo operator for integration tests and the standalone fake server.
//!
//! # Overview
//!
//! `FakeEcho` is a minimal, self-contained operator designed to validate the runtime
//! execution pipeline **without** requiring a real model or learned weights. It acts as
//! a simple "echo" — it reads the last token of each sequence and copies it forward,
//! effectively repeating the final token until a maximum length is reached.
//!
//! # Behaviour
//!
//! - All threads participate in the work (no idle threads).
//! - It only reacts to `Phase::Prefill` sequences in the `decode_list`.
//! - For each sequence slice it:
//!   1. Reads the **last token** from the sequence buffer.
//!   2. Copies that token to the next position in the sequence.
//!   3. Advances `sequence_index` and `kv_index`.
//!   4. Transitions the request from `Phase::Prefill` → `Phase::Decode`.
//! - When the write position reaches 99 (i.e. the sequence has 99 tokens):
//!   - Writes `eos_id` instead of echoing the token.
//!   - Transitions the request to `Phase::Eos`.
//!   - Calls `notify_one()` to wake the waiting slot so the caller can collect results.
//!
//! # Design Goal
//!
//! Prove that the runtime correctly schedules and executes an operator end-to-end
//! without needing a full model forward pass or any operator-owned state.
//!
//! # Memory Layout
//!
//! The `sequences_ptr` points to a flat, row-major buffer where each batch occupies
//! `sequence_stride` consecutive `usize` elements:
//!
//! ```text
//! batch 0: [tok0, tok1, tok2, ..., 0, 0, ...]   ← sequence_stride elements
//! batch 1: [tok0, tok1, tok2, ..., 0, 0, ...]   ← sequence_stride elements
//! ...
//! ```
//!
//! The operator reads from and writes into this buffer using unsafe pointer arithmetic.

use crate::operators::assign::assign;
use crate::runtime::SequenceSlice;
use crate::runtime::{Phase, SlotState};

/// A lightweight "echo" operator that repeats the last token of each sequence
/// until a maximum length is reached, then writes an EOS token.
///
/// This is **not** a real neural-network operator. It exists solely for:
/// - Integration tests that need to verify the runtime scheduling pipeline.
/// - The standalone fake server (`fake_server.rs`) that demonstrates serving
///   without loading actual model weights.
///
/// # Fields
///
/// * `sequences_ptr`   – Raw pointer to the flat token buffer shared with the runtime.
///                        Each batch row has `sequence_stride` elements.
///                        The operator reads the last token and writes the next token
///                        through this pointer using unsafe arithmetic.
///
/// * `sequence_stride` – The number of `usize` elements allocated per batch in the
///                        flat buffer. Acts as the row stride for computing the
///                        address: `batch_index * sequence_stride + offset`.
///
/// * `eos_id`          – The token ID that signals end-of-sequence. When the sequence
///                        reaches length 99, this value is written and the slot is
///                        transitioned to `Phase::Eos`.
#[derive(Clone)]
pub struct FakeEcho {
    sequences_ptr: *mut usize,
    sequence_stride: usize,
    eos_id: usize,
}

impl FakeEcho {
    /// Create a new `FakeEcho` operator.
    ///
    /// # Arguments
    ///
    /// * `sequences_ptr`   – Pointer to the shared flat token buffer (mutable).
    /// * `sequence_stride` – Number of `usize` slots per batch row.
    /// * `eos_id`          – Token ID used to mark end-of-sequence.
    pub fn new(sequences_ptr: *mut usize, sequence_stride: usize, eos_id: usize) -> Self {
        Self {
            sequences_ptr,
            sequence_stride,
            eos_id,
        }
    }

    /// Execute the fake-echo logic for the slice of `decode_list` assigned to this thread.
    ///
    /// For each sequence slice in the assigned range:
    /// 1. Read the **last token** currently stored in the sequence buffer.
    /// 2. If the next write position is **< 99**:
    ///    - Copy (echo) the last token to the next position.
    ///    - Advance `sequence_index` and `kv_index` by 1.
    ///    - If the slot is still in `Phase::Prefill`, transition it to `Phase::Decode`
    ///      and reset `filling_length` to 0.
    /// 3. If the next write position is **>= 99**:
    ///    - Write `eos_id` at the current position.
    ///    - Advance `sequence_index` and `kv_index` by 1.
    ///    - Transition the slot to `Phase::Eos` and wake the waiting consumer via
    ///      `notify.notify_one()`.
    ///
    /// # Arguments
    ///
    /// * `_prefill_size`  – (unused) Number of prefill tokens. Present for API compatibility
    ///                       with real operators.
    /// * `_decode_size`   – (unused) Number of decode tokens. Present for API compatibility.
    /// * `thread_num`     – Total number of worker threads participating in this operator.
    /// * `thread_id`      – 0-based index of the current thread.
    /// * `_prefill_list`  – (unused) Prefill sequence slices. `FakeEcho` only processes decode slices.
    /// * `decode_list`    – Slice of `SequenceSlice` describing the sequences to process.
    /// * `batch_list`     – Mutable reference to the slot state table. Each entry tracks
    ///                       the phase, indices, and notification primitive for one request.
    pub fn run(
        &self,
        _prefill_size: usize,
        _decode_size: usize,
        thread_num: usize,
        thread_id: usize,
        _prefill_list: &[Vec<SequenceSlice>],
        decode_list: &[SequenceSlice],
        batch_list: &mut Vec<SlotState>,
    ) {
        // ---- Step 1: Determine this thread's work range ----
        // `assign` splits `decode_list.len()` evenly across `thread_num` threads.
        // Returns `None` if this thread has nothing to do (e.g. more threads than sequences).
        let Some((begin, end)) = assign(decode_list.len(), thread_num, thread_id) else {
            return;
        };

        // ---- Step 2: Process each sequence slice in [begin, end) ----
        for slice in decode_list.iter().take(end).skip(begin) {
            // `batch_index` identifies which slot (row in `batch_list`) this slice belongs to.
            let batch_index = slice.batch_index;

            // Look up the mutable slot state for this batch. Skip if out of bounds.
            let record = match batch_list.get_mut(batch_index) {
                Some(r) => r,
                None => continue,
            };

            // ---- Step 3: Read the last token from the sequence buffer ----
            // The last token sits at offset (sequence_index + length - 1) within the
            // batch's row in the flat buffer.
            let last_token_index = slice.sequence_index + slice.length - 1;
            let last_token = unsafe {
                *self
                    .sequences_ptr
                    .add(batch_index * self.sequence_stride + last_token_index)
            };

            // ---- Step 4: Compute the write position ----
            // The next token should be written right after the current sequence content.
            let write_start = slice.sequence_index + slice.length;

            if write_start >= 99 {
                // ---- Case A: Sequence is long enough → emit EOS ----
                // Write the EOS token at the current position to signal end-of-sequence.
                let eos_write_index = batch_index * self.sequence_stride + write_start;
                unsafe {
                    *self.sequences_ptr.add(eos_write_index) = self.eos_id;
                }
                // Update the slot metadata to reflect the new sequence length.
                record.sequence_index = write_start + 1;
                record.kv_index = record.kv_index.saturating_add(1);
                record.filling_length = 0;
                // Transition to Eos phase so the scheduler knows this request is done.
                record.phase = Phase::Eos;
                // Wake up the async task waiting for this request's result.
                record.notify.notify_one();
            } else {
                // ---- Case B: Normal echo — copy the last token forward ----
                let write_index = batch_index * self.sequence_stride + write_start;
                unsafe {
                    *self.sequences_ptr.add(write_index) = last_token;
                }
                // Advance the slot's position counters.
                record.sequence_index = write_start + 1;
                record.kv_index = record.kv_index.saturating_add(1);
                // If this is the first run after prefill, transition Prefill → Decode.
                if matches!(record.phase, Phase::Prefill) {
                    record.filling_length = 0;
                    record.phase = Phase::Decode;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FakeEcho;
    use crate::runtime::SequenceSlice;
    use crate::runtime::{Phase, SlotState};

    fn decode_state(sequence_index: usize, kv_index: usize) -> SlotState {
        let mut s = SlotState::idle();
        s.start_decode(sequence_index, kv_index);
        s
    }

    fn prefill_state(sequence_index: usize, filling_length: usize) -> SlotState {
        let mut s = SlotState::idle();
        s.start_prefill(sequence_index, filling_length);
        s
    }

    /// Number of pre-written tokens in each sequence (none of them are eos_id).
    const PRE_TOKEN_COUNT: usize = 10;
    /// The eos_id used across all tests.
    const EOS_ID: usize = 100;

    /// Helper: fill `count` tokens starting at `offset` in the flat buffer for a given batch.
    /// Token values are deterministic: (offset + i) % 50 + 1, guaranteed != EOS_ID (100).
    fn fill_sequence(
        sequences: &mut [usize],
        batch_index: usize,
        stride: usize,
        offset: usize,
        count: usize,
    ) {
        for i in 0..count {
            sequences[batch_index * stride + offset + i] = (offset + i) % 50 + 1;
        }
    }

    #[test]
    fn fake_echo_copies_single_token_each_run() {
        let sequence_stride = 256;
        let mut sequences = vec![0usize; 2 * sequence_stride];
        // Pre-write 10 tokens into batch 0: values are 1,2,3,...,10 (none == eos_id)
        fill_sequence(&mut sequences, 0, sequence_stride, 0, PRE_TOKEN_COUNT);

        let echo = FakeEcho::new(sequences.as_mut_ptr(), sequence_stride, EOS_ID);
        let mut batch_list = vec![prefill_state(0, 0)];
        let prefill_list = vec![];
        let decode_list = vec![SequenceSlice {
            batch_index: 0,
            sequence_index: 0,
            token_start_index: 0,
            length: PRE_TOKEN_COUNT,
            last_token_flag: true,
        }];

        echo.run(0, 1, 1, 0, &prefill_list, &decode_list, &mut batch_list);

        assert_eq!(batch_list[0].phase, Phase::Decode);
        assert_eq!(batch_list[0].sequence_index, PRE_TOKEN_COUNT + 1);
        assert_eq!(batch_list[0].kv_index, 1);
        assert_eq!(batch_list[0].filling_length, 0);
        // The echoed token should be the last pre-written token: (9) % 50 + 1 = 10
        assert_eq!(sequences[PRE_TOKEN_COUNT], 10);
    }

    #[test]
    fn fake_echo_runs_on_all_threads() {
        let sequence_stride = 256;
        let mut sequences = vec![0usize; 2 * sequence_stride];
        // Pre-write 10 tokens into batch 0
        fill_sequence(&mut sequences, 0, sequence_stride, 0, PRE_TOKEN_COUNT);

        let echo = FakeEcho::new(sequences.as_mut_ptr(), sequence_stride, EOS_ID);
        let mut batch_list = vec![prefill_state(PRE_TOKEN_COUNT, 0)];
        let prefill_list = vec![];
        let decode_list = vec![SequenceSlice {
            batch_index: 0,
            sequence_index: 0,
            token_start_index: 0,
            length: PRE_TOKEN_COUNT,
            last_token_flag: true,
        }];

        echo.run(0, 1, 2, 0, &prefill_list, &decode_list, &mut batch_list);

        assert_eq!(batch_list[0].phase, Phase::Decode);
        assert_eq!(batch_list[0].sequence_index, PRE_TOKEN_COUNT + 1);
        assert_eq!(batch_list[0].kv_index, PRE_TOKEN_COUNT + 1);
        assert_eq!(batch_list[0].filling_length, 0);
        // Echoed token: last of the 10 pre-written = (9) % 50 + 1 = 10
        assert_eq!(sequences[PRE_TOKEN_COUNT], 10);
    }

    #[test]
    fn fake_echo_thread_assignment() {
        let sequence_stride = 256;
        let mut sequences = vec![0usize; 3 * sequence_stride];
        // Pre-write 10 tokens into each of the 3 batches
        for batch in 0..3 {
            fill_sequence(&mut sequences, batch, sequence_stride, 0, PRE_TOKEN_COUNT);
        }

        let echo = FakeEcho::new(sequences.as_mut_ptr(), sequence_stride, EOS_ID);
        let mut batch_list = vec![
            prefill_state(PRE_TOKEN_COUNT, 0),
            prefill_state(PRE_TOKEN_COUNT, 0),
            prefill_state(PRE_TOKEN_COUNT, 0),
        ];
        let prefill_list = vec![];
        let decode_list = vec![
            SequenceSlice {
                batch_index: 0,
                sequence_index: 0,
                token_start_index: 0,
                length: PRE_TOKEN_COUNT,
                last_token_flag: true,
            },
            SequenceSlice {
                batch_index: 1,
                sequence_index: 0,
                token_start_index: 0,
                length: PRE_TOKEN_COUNT,
                last_token_flag: true,
            },
            SequenceSlice {
                batch_index: 2,
                sequence_index: 0,
                token_start_index: 0,
                length: PRE_TOKEN_COUNT,
                last_token_flag: true,
            },
        ];

        echo.run(0, 3, 2, 0, &prefill_list, &decode_list, &mut batch_list);
        echo.run(0, 3, 2, 1, &prefill_list, &decode_list, &mut batch_list);

        assert_eq!(batch_list[0].phase, Phase::Decode);
        assert_eq!(batch_list[1].phase, Phase::Decode);
        assert_eq!(batch_list[2].phase, Phase::Decode);

        // Each batch's echoed token should be the last pre-written token: 10
        assert_eq!(sequences[PRE_TOKEN_COUNT], 10);
        assert_eq!(sequences[sequence_stride + PRE_TOKEN_COUNT], 10);
        assert_eq!(sequences[sequence_stride * 2 + PRE_TOKEN_COUNT], 10);
    }

    #[test]
    fn fake_echo_stops_at_length_100_with_eos() {
        let sequence_stride = 256;
        let mut sequences = vec![0usize; 2 * sequence_stride];
        // Pre-write 99 tokens into batch 0 (positions 0..98), none are eos_id.
        // The first 10 are from fill_sequence; the rest (10..99) are also non-eos.
        fill_sequence(&mut sequences, 0, sequence_stride, 0, 99);
        // Verify none of the 99 tokens are eos_id
        for i in 0..99 {
            assert_ne!(
                sequences[i], EOS_ID,
                "pre-written token at {i} must not be eos_id"
            );
        }

        let echo = FakeEcho::new(sequences.as_mut_ptr(), sequence_stride, EOS_ID);
        // sequence_index=89, length=10 → write_start = 89 + 10 = 99 → triggers EOS
        let mut batch_list = vec![decode_state(99, 99)];
        let prefill_list = vec![];
        let decode_list = vec![SequenceSlice {
            batch_index: 0,
            sequence_index: 89,
            token_start_index: 89,
            length: 10,
            last_token_flag: true,
        }];

        echo.run(0, 1, 1, 0, &prefill_list, &decode_list, &mut batch_list);

        assert_eq!(batch_list[0].phase, Phase::Eos);
        assert_eq!(batch_list[0].sequence_index, 100);
        assert_eq!(batch_list[0].kv_index, 100);
        assert_eq!(sequences[99], EOS_ID);
    }

    #[test]
    fn fake_echo_handles_decode_phase() {
        let sequence_stride = 256;
        let mut sequences = vec![0usize; 2 * sequence_stride];
        // Pre-write 10 tokens into batch 0
        fill_sequence(&mut sequences, 0, sequence_stride, 0, PRE_TOKEN_COUNT);

        let echo = FakeEcho::new(sequences.as_mut_ptr(), sequence_stride, EOS_ID);
        let mut batch_list = vec![decode_state(PRE_TOKEN_COUNT, PRE_TOKEN_COUNT)];
        let prefill_list = vec![];
        let decode_list = vec![SequenceSlice {
            batch_index: 0,
            sequence_index: 0,
            token_start_index: 0,
            length: PRE_TOKEN_COUNT,
            last_token_flag: true,
        }];

        echo.run(0, 1, 1, 0, &prefill_list, &decode_list, &mut batch_list);

        assert_eq!(batch_list[0].phase, Phase::Decode);
        // Echoed token: last of the 10 pre-written = 10
        assert_eq!(sequences[PRE_TOKEN_COUNT], 10);
    }
}
