use std::sync::Arc;
use std::thread;
use std::time::Duration;

use super::scheduler_plan::{PrefillCandidate, SliceScheduler};
use crate::common::send_sync_ptr::SharedMut;
use crate::common::sequence_slice::{DecodeList, SequenceSlice};
use crate::common::state::{Phase, SequenceState};

pub struct BatchScheduler {
    pub prefill_list: Vec<Vec<SequenceSlice>>,
    pub decode_list: DecodeList,
    pub batch_list: Arc<SharedMut<Vec<SequenceState>>>,
    prefill_scheduler: SliceScheduler,
    max_prefill_size: usize,
    max_decode_size: usize,
    thread_num: usize,
}

enum BatchPlan {
    Decode(Vec<(usize, usize)>),
    Prefill {
        candidates: Vec<PrefillCandidate>,
        total_tokens: usize,
    },
    Idle,
}

impl BatchScheduler {
    fn build(sequence_length: usize, batch_size: usize, thread_num: usize) -> Self {
        let build_prefill_list = || {
            (0..thread_num)
                .map(|_| Vec::with_capacity(batch_size))
                .collect::<Vec<_>>()
        };

        Self {
            max_decode_size: batch_size,
            max_prefill_size: sequence_length * batch_size,
            batch_list: Arc::new(SharedMut::new(Vec::with_capacity(batch_size))),
            thread_num,
            prefill_scheduler: SliceScheduler::new(batch_size * thread_num),
            prefill_list: build_prefill_list(),
            decode_list: DecodeList::with_capacity(batch_size),
        }
    }

    fn clear_round_outputs(&mut self) {
        for task in self.prefill_list.iter_mut() {
            task.clear();
        }
        self.decode_list.clear();
    }

    fn plan_next_round(&self) -> BatchPlan {
        let max_decode_size = self.max_decode_size;
        self.batch_list.with(|batch_list| {
            let mut decode_candidates = Vec::with_capacity(max_decode_size);
            let mut total_tokens = 0usize;
            let mut candidates = Vec::with_capacity(batch_list.len());
            let mut has_decode = false;

            for (batch_index, record) in batch_list.iter().enumerate() {
                match record.phase {
                    Phase::Decode => {
                        has_decode = true;
                        if decode_candidates.len() < max_decode_size {
                            decode_candidates.push((batch_index, record.kv_index));
                        }
                    }
                    Phase::Prefill => {
                        let remaining = record.filling_length;
                        total_tokens += remaining;
                        candidates.push(PrefillCandidate {
                            batch_index,
                            sequence_index: record.sequence_index,
                            remaining,
                        });
                    }
                    _ => {}
                }
            }

            if has_decode {
                BatchPlan::Decode(decode_candidates)
            } else if candidates.is_empty() {
                BatchPlan::Idle
            } else {
                BatchPlan::Prefill {
                    candidates,
                    total_tokens: total_tokens.min(self.max_prefill_size),
                }
            }
        })
    }

    // Test method (simplest example):
    // 1) Build a small batch list with one Prefill record.
    // 2) Call schedule_batch and verify prefill/decode counts and slices.
    // Example:
    // let mut scheduler = BatchScheduler::new(8, 1, 1);
    // scheduler.batch_list.push(SequenceState {
    //     phase: Phase::Prefill,
    //     sequence_index: 4,
    //     kv_index: 0,
    //     notify: std::sync::Arc::new(tokio::sync::Notify::new()),
    // });
    // let (prefill, decode) = scheduler.schedule_batch();
    // assert!(prefill > 0);
    // assert_eq!(decode, 0);
    pub fn new(sequence_length: usize, batch_size: usize, thread_num: usize) -> Self {
        Self::build(sequence_length, batch_size, thread_num)
    }

    fn schedule_decode_round(&mut self, decode_candidates: Vec<(usize, usize)>) -> usize {
        self.clear_round_outputs();
        let mut decode_count = 0usize;
        for (batch_index, sequence_index) in decode_candidates {
            let token_start_index = decode_count;

            self.decode_list.push(SequenceSlice {
                batch_index,
                sequence_index,
                token_start_index,
                length: 1,
                last_token_flag: true,
            });
            decode_count += 1;
        }

        decode_count
    }

    fn schedule_prefill_round(
        &mut self,
        prefill_candidates: Vec<PrefillCandidate>,
        total_tokens: usize,
    ) -> usize {
        self.clear_round_outputs();
        let mut prefill_count = 0usize;
        self.prefill_scheduler.init(total_tokens);

        let prefill_scheduler = &mut self.prefill_scheduler;
        let prefill_list = &mut self.prefill_list;
        let decode_list = &mut self.decode_list;
        self.batch_list.with_mut(|_| {
            for candidate in prefill_candidates.iter().copied() {
                if prefill_scheduler.is_done() {
                    break;
                }

                let scheduled_before = prefill_count;
                let attention_length = candidate
                    .remaining
                    .min(prefill_scheduler.remaining_tokens());
                if attention_length > 0 {
                    decode_list.push(SequenceSlice {
                        batch_index: candidate.batch_index,
                        sequence_index: candidate.sequence_index,
                        token_start_index: scheduled_before,
                        length: attention_length,
                        last_token_flag: attention_length == candidate.remaining,
                    });
                }

                prefill_scheduler.schedule_for_sequence(
                    candidate.batch_index,
                    candidate.sequence_index,
                    candidate.remaining,
                    0,
                    prefill_list,
                    &mut prefill_count,
                );
            }
        });

        prefill_count
    }

    pub fn schedule_batch(&mut self) -> (usize, usize) {
        let prefill_task_count = self.thread_num.min(self.prefill_list.len());

        self.prefill_scheduler.set_task_count(prefill_task_count);

        loop {
            match self.plan_next_round() {
                BatchPlan::Decode(decode_candidates) => {
                    let decode_count = self.schedule_decode_round(decode_candidates);
                    return (0, decode_count);
                }
                BatchPlan::Prefill {
                    candidates,
                    total_tokens,
                } => {
                    let prefill_count = self.schedule_prefill_round(candidates, total_tokens);
                    return (prefill_count, self.decode_list.len());
                }
                BatchPlan::Idle => {
                    thread::sleep(Duration::from_millis(1));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::scheduler_allocator::FairTaskAllocator;
    use super::*;
    use tokio::sync::Notify;

    // Build a decode-phase record.
    // Decode records represent already-prefilled sequences that should emit
    // exactly one token this round, so `filling_length` is always 0 here.
    fn decode_state(sequence_index: usize, kv_index: usize) -> SequenceState {
        SequenceState {
            filling_length: 0,
            phase: Phase::Decode,
            sequence_index,
            kv_index,
            notify: std::sync::Arc::new(Notify::new()),
        }
    }

    // Build a prefill-phase record.
    // The `kv_index` is derived from `sequence_index + filling_length` so the
    // state matches the scheduler's expectation that prefill advances both
    // cursors together when work is assigned.
    fn prefill_state(sequence_index: usize, filling_length: usize) -> SequenceState {
        SequenceState {
            filling_length,
            phase: Phase::Prefill,
            sequence_index,
            kv_index: sequence_index + filling_length,
            notify: std::sync::Arc::new(Notify::new()),
        }
    }

    #[test]
    // Empty batch means there is no work to schedule, so the scheduler must
    // stay idle instead of inventing slices or spinning on bogus state.
    fn plan_next_round_returns_idle_for_empty_batch() {
        let scheduler = BatchScheduler::new(16, 4, 3);

        match scheduler.plan_next_round() {
            BatchPlan::Idle => {}
            _ => panic!("expected idle plan for an empty batch"),
        }
    }

    #[test]
    // Decode has hard priority over prefill.
    // This test mixes both phases in a longer batch and verifies that the
    // scheduler selects only decode candidates, truncates to `batch_size`, and
    // leaves every prefill task untouched for this round.
    fn schedule_decode_round_limits_to_batch_size_and_prefill_is_ignored() {
        let mut scheduler = BatchScheduler::new(16, 4, 3);
        scheduler.batch_list.with_mut(|batch_list| {
            // Interleave decode and prefill entries to prove that prefill
            // records are ignored whenever at least one decode record exists.
            batch_list.push(decode_state(100, 128));
            batch_list.push(prefill_state(0, 14));
            batch_list.push(decode_state(200, 256));
            batch_list.push(prefill_state(32, 12));
            batch_list.push(decode_state(300, 384));
            batch_list.push(decode_state(400, 512));
            batch_list.push(prefill_state(64, 10));
        });

        let (prefill, decode) = scheduler.schedule_batch();

        assert_eq!(prefill, 0);
        assert_eq!(decode, 4);

        // The prefill side stays empty because the round is decode-only.
        assert!(scheduler.prefill_list.iter().all(Vec::is_empty));
        assert_eq!(scheduler.decode_list.len(), 4);

        // Decode slices are emitted in batch order, one token per slice.
        let expected = [(0, 128), (2, 256), (4, 384), (5, 512)];
        for (slice, &(batch_index, sequence_index)) in scheduler.decode_list.iter().zip(expected.iter())
        {
            assert_eq!(slice.batch_index, batch_index);
            assert_eq!(slice.sequence_index, sequence_index);
            assert_eq!(slice.length, 1);
            assert!(slice.last_token_flag);
        }
        for (index, slice) in scheduler.decode_list.iter().enumerate() {
            assert_eq!(slice.token_start_index, index);
        }
    }

    #[test]
    // Prefill rounds are the main exercise for the static slice planner.
    // This test uses long sequences so we can observe how the allocator splits
    // work across threads while preserving per-sequence continuity.
    fn schedule_prefill_round_splits_long_sequences_across_threads() {
        let mut scheduler = BatchScheduler::new(8, 4, 3);
        scheduler.batch_list.with_mut(|batch_list| {
            // Three long prefill entries:
            // - the first fits almost perfectly into the first task quota,
            // - the second spans multiple thread buckets,
            // - the third keeps the round non-trivial after the first two.
            batch_list.push(prefill_state(0, 10));
            batch_list.push(prefill_state(32, 9));
            batch_list.push(prefill_state(64, 4));
        });

        let (prefill, decode) = scheduler.schedule_batch();

        // All 23 tokens are within the prefill window, so nothing is truncated.
        assert_eq!(prefill, 23);
        assert_eq!(decode, 3);
        assert_eq!(scheduler.decode_list.len(), 3);

        // Each entry in `decode_list` is the attention view for one sequence.
        // The first token_start_index of each slice tracks the global offset.
        let decode_lengths: Vec<usize> = scheduler.decode_list.iter().map(|slice| slice.length).collect();
        let decode_flags: Vec<bool> = scheduler
            .decode_list
            .iter()
            .map(|slice| slice.last_token_flag)
            .collect();
        let decode_starts: Vec<usize> = scheduler
            .decode_list
            .iter()
            .map(|slice| slice.token_start_index)
            .collect();

        assert_eq!(decode_lengths, vec![10, 9, 4]);
        assert_eq!(decode_flags, vec![true, true, true]);
        assert_eq!(decode_starts, vec![0, 10, 19]);

        // The prefill side is split by thread quota, not by sequence count.
        // Each thread gets a continuous chunk of tokens, and a sequence may be
        // split across multiple thread-local slices.
        assert_eq!(scheduler.prefill_list.len(), 3);
        assert_eq!(scheduler.prefill_list[0].len(), 1);
        assert_eq!(scheduler.prefill_list[1].len(), 2);
        assert_eq!(scheduler.prefill_list[2].len(), 2);

        // Thread 0 gets the first long chunk from sequence 0.
        let t0 = &scheduler.prefill_list[0][0];
        assert_eq!(t0.batch_index, 0);
        assert_eq!(t0.sequence_index, 0);
        assert_eq!(t0.token_start_index, 0);
        assert_eq!(t0.length, 8);

        // Thread 1 receives the tail of sequence 0 and the first chunk of sequence 1.
        let t1_first = &scheduler.prefill_list[1][0];
        assert_eq!(t1_first.batch_index, 0);
        assert_eq!(t1_first.sequence_index, 8);
        assert_eq!(t1_first.token_start_index, 8);
        assert_eq!(t1_first.length, 2);

        let t1_second = &scheduler.prefill_list[1][1];
        assert_eq!(t1_second.batch_index, 1);
        assert_eq!(t1_second.sequence_index, 32);
        assert_eq!(t1_second.token_start_index, 10);
        assert_eq!(t1_second.length, 6);

        // Thread 2 finishes the remaining part of sequence 1 and then consumes
        // sequence 2, demonstrating that the scheduler keeps advancing the
        // sequence cursor across slice boundaries.
        let t2_first = &scheduler.prefill_list[2][0];
        assert_eq!(t2_first.batch_index, 1);
        assert_eq!(t2_first.sequence_index, 38);
        assert_eq!(t2_first.token_start_index, 16);
        assert_eq!(t2_first.length, 3);

        let t2_second = &scheduler.prefill_list[2][1];
        assert_eq!(t2_second.batch_index, 2);
        assert_eq!(t2_second.sequence_index, 64);
        assert_eq!(t2_second.token_start_index, 19);
        assert_eq!(t2_second.length, 4);
    }

    #[test]
    // Prefill must respect the round window.
    // Here the total demand is larger than the allowed prefill capacity, so
    // the last sequence is only partially scheduled and its `last_token_flag`
    // should stay false.
    fn schedule_prefill_round_truncates_to_window_and_marks_partial_last_slice_false() {
        let mut scheduler = BatchScheduler::new(5, 2, 2);
        scheduler.batch_list.with_mut(|batch_list| {
            // Long enough to force truncation after the first two sequences.
            batch_list.push(prefill_state(0, 4));
            batch_list.push(prefill_state(16, 5));
            batch_list.push(prefill_state(32, 6));
        });

        let (prefill, decode) = scheduler.schedule_batch();

        // `sequence_length * batch_size = 10`, so only 10 tokens can be
        // scheduled even though the batch asks for 15.
        assert_eq!(prefill, 10);
        assert_eq!(decode, 3);
        assert_eq!(scheduler.decode_list.len(), 3);

        // The first two slices are fully scheduled; the last one is truncated.
        let lengths: Vec<usize> = scheduler.decode_list.iter().map(|slice| slice.length).collect();
        let starts: Vec<usize> = scheduler
            .decode_list
            .iter()
            .map(|slice| slice.token_start_index)
            .collect();
        let flags: Vec<bool> = scheduler
            .decode_list
            .iter()
            .map(|slice| slice.last_token_flag)
            .collect();
        let sequence_indices: Vec<usize> =
            scheduler.decode_list.iter().map(|slice| slice.sequence_index).collect();

        assert_eq!(lengths, vec![4, 5, 1]);
        assert_eq!(starts, vec![0, 4, 9]);
        assert_eq!(flags, vec![true, true, false]);
        assert_eq!(sequence_indices, vec![0, 16, 32]);

        // Thread-local prefill slices preserve contiguous token ranges.
        assert_eq!(scheduler.prefill_list[0].len(), 2);
        assert_eq!(scheduler.prefill_list[1].len(), 2);

        let thread0 = &scheduler.prefill_list[0];
        assert_eq!(thread0[0].batch_index, 0);
        assert_eq!(thread0[0].sequence_index, 0);
        assert_eq!(thread0[0].token_start_index, 0);
        assert_eq!(thread0[0].length, 4);
        assert_eq!(thread0[1].batch_index, 1);
        assert_eq!(thread0[1].sequence_index, 16);
        assert_eq!(thread0[1].token_start_index, 4);
        assert_eq!(thread0[1].length, 1);

        let thread1 = &scheduler.prefill_list[1];
        assert_eq!(thread1[0].batch_index, 1);
        assert_eq!(thread1[0].sequence_index, 17);
        assert_eq!(thread1[0].token_start_index, 5);
        assert_eq!(thread1[0].length, 4);
        assert_eq!(thread1[1].batch_index, 2);
        assert_eq!(thread1[1].sequence_index, 32);
        assert_eq!(thread1[1].token_start_index, 9);
        assert_eq!(thread1[1].length, 1);
    }

    #[test]
    // Two consecutive rounds should not leak stale slice data.
    // The first round schedules prefill, the second round switches to decode,
    // and the old prefill buffers must be cleared before new output is written.
    fn schedule_batch_clears_stale_outputs_between_rounds() {
        let mut scheduler = BatchScheduler::new(8, 3, 2);
        scheduler.batch_list.with_mut(|batch_list| {
            // First round: long enough to populate both thread-local prefill
            // queues and the flat decode list.
            batch_list.push(prefill_state(0, 7));
            batch_list.push(prefill_state(24, 6));
            batch_list.push(prefill_state(48, 5));
        });

        let (prefill, decode) = scheduler.schedule_batch();
        assert_eq!(prefill, 18);
        assert_eq!(decode, 3);
        assert!(scheduler.prefill_list.iter().any(|task| !task.is_empty()));
        assert_eq!(scheduler.decode_list.len(), 3);

        // Replace the batch with decode-only work and verify the old prefill
        // slices disappear instead of being appended to.
        scheduler.batch_list.with_mut(|batch_list| {
            batch_list.clear();
            batch_list.push(decode_state(128, 140));
            batch_list.push(decode_state(256, 260));
            batch_list.push(decode_state(384, 390));
            batch_list.push(decode_state(512, 520));
        });

        let (prefill, decode) = scheduler.schedule_batch();
        assert_eq!(prefill, 0);
        assert_eq!(decode, 3);
        assert!(scheduler.prefill_list.iter().all(Vec::is_empty));
        assert_eq!(scheduler.decode_list.len(), 3);
        assert_eq!(
            scheduler
                .decode_list
                .iter()
                .map(|slice| slice.sequence_index)
                .collect::<Vec<_>>(),
            vec![140, 260, 390]
        );
    }

    #[test]
    // Decode should still win even if prefill entries are larger and more
    // numerous. This verifies the phase decision itself, not the slice size.
    fn schedule_decode_priority_overrides_prefill_even_with_long_backlog() {
        let mut scheduler = BatchScheduler::new(12, 5, 4);
        scheduler.batch_list.with_mut(|batch_list| {
            // A busy batch with both phases interleaved.
            batch_list.push(prefill_state(0, 11));
            batch_list.push(prefill_state(32, 13));
            batch_list.push(decode_state(100, 144));
            batch_list.push(prefill_state(64, 7));
            batch_list.push(decode_state(200, 288));
            batch_list.push(decode_state(300, 320));
            batch_list.push(prefill_state(96, 5));
            batch_list.push(decode_state(400, 448));
        });

        let (prefill, decode) = scheduler.schedule_batch();

        assert_eq!(prefill, 0);
        assert_eq!(decode, 4);
        assert!(scheduler.prefill_list.iter().all(Vec::is_empty));

        // Only decode records are kept, in batch order, up to `batch_size`.
        assert_eq!(
            scheduler
                .decode_list
                .iter()
                .map(|slice| (slice.batch_index, slice.sequence_index, slice.length))
                .collect::<Vec<_>>(),
            vec![(2, 144, 1), (4, 288, 1), (5, 320, 1), (7, 448, 1)]
        );
    }

    #[test]
    // FairTaskAllocator should divide a long token run as evenly as possible.
    // With 47 tokens and 6 tasks, the first five tasks get 8 tokens and the
    // final task gets 7.
    fn fair_task_allocator_balances_long_token_runs() {
        let mut allocator = FairTaskAllocator::new(6);
        allocator.init(47);

        let mut per_task = [0usize; 6];
        while let Some(task_index) = allocator.current_task_index() {
            let taken = allocator.take(usize::MAX);
            if taken == 0 {
                break;
            }
            per_task[task_index] += taken;
        }

        assert_eq!(per_task, [8, 8, 8, 8, 8, 7]);
        assert!(allocator.is_done());
        assert_eq!(allocator.scheduled_tokens(), 47);
    }

    #[test]
    // When there are more tasks than tokens, only the first `total_tokens`
    // tasks should become active and each should receive exactly one token.
    fn fair_task_allocator_handles_more_tasks_than_tokens() {
        let mut allocator = FairTaskAllocator::new(8);
        allocator.init(5);

        let mut trace = Vec::new();
        while let Some(task_index) = allocator.current_task_index() {
            let taken = allocator.take(usize::MAX);
            if taken == 0 {
                break;
            }
            trace.push((task_index, taken));
        }

        assert_eq!(trace, vec![(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]);
        assert!(allocator.is_done());
        assert_eq!(allocator.scheduled_tokens(), 5);
    }
}
