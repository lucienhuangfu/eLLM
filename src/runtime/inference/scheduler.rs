use std::sync::Arc;
use std::thread;
use std::time::Duration;

use super::scheduler_plan::{PrefillCandidate, SliceScheduler};
use super::state::{Phase, SequenceState};
use crate::common::send_sync_ptr::SharedMut;
use crate::common::sequence_slice::{RoundTokenSlices, SequenceSlice};

pub struct BatchScheduler {
    pub prefill_list: Vec<Vec<SequenceSlice>>,
    pub round_token_slices: RoundTokenSlices,
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
    fn clear_round_outputs(&mut self) {
        for task in self.prefill_list.iter_mut() {
            task.clear();
        }
        self.round_token_slices.clear();
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
            round_token_slices: RoundTokenSlices::with_capacity(batch_size),
        }
    }

    fn schedule_decode_round(&mut self, decode_candidates: Vec<(usize, usize)>) -> usize {
        self.clear_round_outputs();
        let mut decode_count = 0usize;
        for (batch_index, sequence_index) in decode_candidates {
            let token_start_index = decode_count;

            self.round_token_slices.push(SequenceSlice {
                batch_index,
                sequence_index,
                token_start_index,
                length: 1,
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
        let round_token_slices = &mut self.round_token_slices;
        self.batch_list.with_mut(|batch_list| {
            for candidate in prefill_candidates.iter().copied() {
                if prefill_scheduler.is_done() {
                    break;
                }

                let scheduled_before = prefill_count;
                let attention_length = candidate
                    .remaining
                    .min(prefill_scheduler.remaining_tokens());
                if attention_length > 0 {
                    round_token_slices.push(SequenceSlice {
                        batch_index: candidate.batch_index,
                        sequence_index: candidate.sequence_index,
                        token_start_index: scheduled_before,
                        length: attention_length,
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

                let scheduled_for_record = prefill_count.saturating_sub(scheduled_before);
                if let Some(record) = batch_list.get_mut(candidate.batch_index) {
                    if scheduled_for_record > 0 {
                        record.sequence_index =
                            record.sequence_index.saturating_add(scheduled_for_record);
                        record.filling_length =
                            record.filling_length.saturating_sub(scheduled_for_record);
                    }

                    if record.filling_length == 0 {
                        record.phase = Phase::Decode;
                    }
                }
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
                    return (prefill_count, self.round_token_slices.len());
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

    fn state(phase: Phase, sequence_index: usize, kv_index: usize) -> SequenceState {
        SequenceState {
            filling_length: kv_index.saturating_sub(sequence_index),
            phase,
            sequence_index,
            kv_index,
            notify: std::sync::Arc::new(Notify::new()),
        }
    }

    #[test]
    fn schedule_prefill_only() {
        // Input:
        // - Eight SequenceState in Phase::Prefill
        // - sequence_index=0, length=6 => 6 tokens remaining per record
        // - sequence_length=8, batch_size=32, thread_num=8
        // Output (expected):
        // - prefill == 48 (all 8 * 6 tokens scheduled)
        // - decode == 0 (no decode tokens scheduled in prefill-only case)
        // - prefill_list[0..8] each has one slice starting at sequence position 0
        // - round_token_slices records one full attention slice per record
        let batch_size = 32;
        let mut scheduler = BatchScheduler::new(8, batch_size, 8);
        scheduler.batch_list.with_mut(|batch_list| {
            for _ in 0..8 {
                batch_list.push(state(Phase::Prefill, 0, 6));
            }
        });

        let (prefill, decode) = scheduler.schedule_batch();

        assert_eq!(prefill, 48);
        assert_eq!(decode, 8);

        assert_eq!(scheduler.prefill_list.len(), 8);
        assert_eq!(scheduler.round_token_slices.len(), 8);

        scheduler.batch_list.with(|batch_list| {
            for record in batch_list.iter().take(8) {
                assert_eq!(record.phase, Phase::Decode);
                assert_eq!(record.sequence_index, 6);
                assert_eq!(record.kv_index, 6);
                assert_eq!(record.filling_length, 0);
            }
        });

        for task_index in 0..8 {
            assert_eq!(scheduler.prefill_list[task_index].len(), 1);
            let slice = &scheduler.prefill_list[task_index][0];
            assert_eq!(slice.batch_index, task_index);
            assert_eq!(slice.sequence_index, 0);
            assert_eq!(slice.token_start_index, task_index * 6);
            assert_eq!(slice.length, 6);
        }

        for sequence_offset in 0..8 {
            let attention_slice = &scheduler.round_token_slices[sequence_offset];
            assert_eq!(attention_slice.batch_index, sequence_offset);
            assert_eq!(attention_slice.sequence_index, 0);
            assert_eq!(attention_slice.token_start_index, sequence_offset * 6);
            assert_eq!(attention_slice.length, 6);
        }
    }

    #[test]
    fn schedule_decode_only_prioritizes_decode_tasks() {
        // Input:
        // - batch_size=2 limits decode scheduling to 2 records
        // - Mixed phases: Decode, Prefill, Decode, Decode
        // Output (expected):
        // - decode == 2, prefill == 0
        // - Only first two decode records are scheduled
        // - decode slices are evenly assigned to two tasks
        // - prefill_list stays empty because decode has priority
        let mut scheduler = BatchScheduler::new(8, 2, 2);
        scheduler.batch_list.with_mut(|batch_list| {
            batch_list.push(state(Phase::Decode, 10, 11));
            batch_list.push(state(Phase::Prefill, 0, 6));
            batch_list.push(state(Phase::Decode, 11, 12));
            batch_list.push(state(Phase::Decode, 12, 13));
        });

        let (prefill, decode) = scheduler.schedule_batch();

        assert_eq!(prefill, 0);
        assert_eq!(decode, 2);

        for task_index in 0..2 {
            assert!(scheduler.prefill_list[task_index].is_empty());
        }
        assert_eq!(scheduler.round_token_slices.len(), 2);
        assert_eq!(scheduler.round_token_slices[0].length, 1);
        assert_eq!(scheduler.round_token_slices[1].length, 1);

        let first = &scheduler.round_token_slices[0];
        assert_eq!(first.batch_index, 0);
        assert_eq!(first.sequence_index, 11);
        assert_eq!(first.token_start_index, 0);

        let second = &scheduler.round_token_slices[1];
        assert_eq!(second.batch_index, 2);
        assert_eq!(second.sequence_index, 12);
        assert_eq!(second.token_start_index, 1);
    }

    #[test]
    fn schedule_decode_round_clears_stale_prefill_slices() {
        let mut scheduler = BatchScheduler::new(8, 2, 2);
        scheduler.batch_list.with_mut(|batch_list| {
            batch_list.push(state(Phase::Prefill, 0, 4));
        });

        let (prefill, decode) = scheduler.schedule_batch();
        assert_eq!(prefill, 4);
        assert_eq!(decode, 1);
        assert!(scheduler.prefill_list.iter().any(|task| !task.is_empty()));

        scheduler.batch_list.with_mut(|batch_list| {
            batch_list.clear();
            batch_list.push(state(Phase::Decode, 8, 9));
        });

        let (prefill, decode) = scheduler.schedule_batch();
        assert_eq!(prefill, 0);
        assert_eq!(decode, 1);
        assert!(scheduler.prefill_list.iter().all(Vec::is_empty));
        assert_eq!(scheduler.round_token_slices.len(), 1);
        assert_eq!(scheduler.round_token_slices[0].batch_index, 0);
    }

    #[test]
    fn fair_task_allocator_balances_tokens() {
        // Input:
        // - task_count=3
        // - total_tokens=11
        // Output (expected):
        // - token allocation per task = [4, 4, 3]
        // - scheduled_tokens == 11
        let mut allocator = FairTaskAllocator::new(3);
        allocator.init(11);

        let mut per_task = [0usize; 3];
        while let Some(task_index) = allocator.current_task_index() {
            let taken = allocator.take(10);
            if taken == 0 {
                break;
            }
            per_task[task_index] += taken;
        }

        assert_eq!(per_task, [4, 4, 3]);
        assert!(allocator.is_done());
        assert_eq!(allocator.scheduled_tokens(), 11);
    }

    #[test]
    fn fair_task_allocator_zero_tokens() {
        // Input:
        // - task_count=4
        // - total_tokens=0
        // Output (expected):
        // - allocator is done immediately
        // - current_task_index == None
        // - scheduled_tokens == 0
        let mut allocator = FairTaskAllocator::new(4);
        allocator.init(0);

        assert!(allocator.is_done());
        assert_eq!(allocator.current_task_index(), None);
        assert_eq!(allocator.scheduled_tokens(), 0);
    }

    #[test]
    fn schedule_prefill_partial_keeps_unscheduled_records_in_prefill() {
        // Input:
        // - sequence_length=4, batch_size=2 => max_prefill_size = 8
        // - Two Prefill records each require 6 tokens (total 12 > 8)
        // Output (expected):
        // - first record fully scheduled -> switches to Decode
        // - second record partially scheduled -> remains Prefill with advanced sequence_index
        let mut scheduler = BatchScheduler::new(4, 2, 2);
        scheduler.batch_list.with_mut(|batch_list| {
            batch_list.push(state(Phase::Prefill, 0, 6));
            batch_list.push(state(Phase::Prefill, 0, 6));
        });

        let (prefill, decode) = scheduler.schedule_batch();

        assert_eq!(prefill, 8);
        assert_eq!(decode, 2);

        scheduler.batch_list.with(|batch_list| {
            assert_eq!(batch_list[0].phase, Phase::Decode);
            assert_eq!(batch_list[0].sequence_index, 6);
            assert_eq!(batch_list[0].kv_index, 6);
            assert_eq!(batch_list[0].filling_length, 0);
            assert_eq!(batch_list[1].phase, Phase::Prefill);
            assert_eq!(batch_list[1].sequence_index, 2);
            assert_eq!(batch_list[1].kv_index, 6);
            assert_eq!(batch_list[1].filling_length, 4);
        });

        assert_eq!(scheduler.round_token_slices.len(), 2);

        let first_attention = &scheduler.round_token_slices[0];
        assert_eq!(first_attention.batch_index, 0);
        assert_eq!(first_attention.sequence_index, 0);
        assert_eq!(first_attention.token_start_index, 0);
        assert_eq!(first_attention.length, 6);

        let second_attention = &scheduler.round_token_slices[1];
        assert_eq!(second_attention.batch_index, 1);
        assert_eq!(second_attention.sequence_index, 0);
        assert_eq!(second_attention.token_start_index, 6);
        assert_eq!(second_attention.length, 2);

        let first_lift = &scheduler.round_token_slices[0];
        assert_eq!(first_lift.batch_index, 0);
        assert_eq!(first_lift.sequence_index, 0);
        assert_eq!(first_lift.token_start_index, 0);
        assert_eq!(first_lift.length, 6);

        let second_lift = &scheduler.round_token_slices[1];
        assert_eq!(second_lift.batch_index, 1);
        assert_eq!(second_lift.sequence_index, 0);
        assert_eq!(second_lift.token_start_index, 6);
        assert_eq!(second_lift.length, 2);
    }

    #[test]
    fn schedule_prefill_partial_resumes_from_updated_sequence_index() {
        let mut scheduler = BatchScheduler::new(4, 2, 2);
        scheduler.batch_list.with_mut(|batch_list| {
            batch_list.push(state(Phase::Prefill, 0, 6));
            batch_list.push(state(Phase::Prefill, 0, 6));
        });

        let (prefill, decode) = scheduler.schedule_batch();

        assert_eq!(prefill, 8);
        assert_eq!(decode, 2);

        scheduler.batch_list.with(|batch_list| {
            assert_eq!(batch_list[1].phase, Phase::Prefill);
            assert_eq!(batch_list[1].sequence_index, 2);
            assert_eq!(batch_list[1].kv_index, 6);
            assert_eq!(batch_list[1].filling_length, 4);
        });

        scheduler.batch_list.with_mut(|batch_list| {
            batch_list[0].phase = Phase::Eos;
        });

        let (prefill, decode) = scheduler.schedule_batch();

        assert_eq!(prefill, 4);
        assert_eq!(decode, 1);

        scheduler.batch_list.with(|batch_list| {
            assert_eq!(batch_list[0].phase, Phase::Eos);
            assert_eq!(batch_list[0].sequence_index, 6);
            assert_eq!(batch_list[1].phase, Phase::Decode);
            assert_eq!(batch_list[1].sequence_index, 6);
            assert_eq!(batch_list[1].filling_length, 0);
        });

        assert_eq!(scheduler.round_token_slices.len(), 1);
        let resumed_attention = &scheduler.round_token_slices[0];
        assert_eq!(resumed_attention.batch_index, 1);
        assert_eq!(resumed_attention.sequence_index, 2);
        assert_eq!(resumed_attention.token_start_index, 0);
        assert_eq!(resumed_attention.length, 4);

        let resumed_lift = &scheduler.round_token_slices[0];
        assert_eq!(resumed_lift.batch_index, 1);
        assert_eq!(resumed_lift.sequence_index, 2);
        assert_eq!(resumed_lift.token_start_index, 0);
        assert_eq!(resumed_lift.length, 4);

        assert_eq!(scheduler.prefill_list[0].len(), 1);
        let resumed_slice = &scheduler.prefill_list[0][0];
        assert_eq!(resumed_slice.batch_index, 1);
        assert_eq!(resumed_slice.sequence_index, 2);
        assert_eq!(resumed_slice.token_start_index, 0);
        assert_eq!(resumed_slice.length, 2);

        assert_eq!(scheduler.prefill_list[1].len(), 1);
        let resumed_slice = &scheduler.prefill_list[1][0];
        assert_eq!(resumed_slice.batch_index, 1);
        assert_eq!(resumed_slice.sequence_index, 4);
        assert_eq!(resumed_slice.token_start_index, 2);
        assert_eq!(resumed_slice.length, 2);
    }

    #[test]
    fn schedule_prefill_splits_by_record_length() {
        let mut scheduler = BatchScheduler::new(16, 1, 2);
        scheduler.batch_list.with_mut(|batch_list| {
            batch_list.push(SequenceState {
                sequence_index: 0,
                kv_index: 32,
                filling_length: 6,
                phase: Phase::Prefill,
                notify: std::sync::Arc::new(Notify::new()),
            });
        });

        let (prefill, decode) = scheduler.schedule_batch();

        assert_eq!(prefill, 6);
        assert_eq!(decode, 1);
        assert_eq!(scheduler.round_token_slices.len(), 1);
        assert_eq!(scheduler.round_token_slices[0].length, 6);
        assert_eq!(scheduler.round_token_slices[0].token_start_index, 0);

        scheduler.batch_list.with(|batch_list| {
            assert_eq!(batch_list[0].sequence_index, 6);
            assert_eq!(batch_list[0].kv_index, 32);
            assert_eq!(batch_list[0].filling_length, 0);
            assert_eq!(batch_list[0].phase, Phase::Decode);
        });
    }
}
