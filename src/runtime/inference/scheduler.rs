use std::sync::Arc;
use std::thread;
use std::time::Duration;

use super::scheduler_plan::{BatchWork, PrefillCandidate, SliceScheduler};
use super::state::{Phase, SequenceState};
use crate::common::send_sync_ptr::SharedMut;
use crate::common::sequence_slice::SequenceSlice;

pub struct BatchScheduler {
    pub prefill_list: Vec<Vec<SequenceSlice>>,
    pub decode_list: Vec<Vec<SequenceSlice>>,
    pub attention_list: Vec<SequenceSlice>,
    pub batch_list: Arc<SharedMut<Vec<SequenceState>>>,
    decode_scheduler: SliceScheduler,
    prefill_scheduler: SliceScheduler,
    max_prefill_size: usize,
    max_decode_size: usize,
    thread_num: usize,
}

impl BatchScheduler {
    fn push_prefill_lift_slice(
        decode_list: &mut [Vec<SequenceSlice>],
        batch_index: usize,
        sequence_index: usize,
        token_start_index: usize,
    ) {
        if decode_list.is_empty() {
            return;
        }

        let task_index = batch_index % decode_list.len();
        decode_list[task_index].push(SequenceSlice {
            batch_index,
            sequence_index,
            token_start_index,
            lift_index: batch_index,
            length: 1,
        });
    }

    fn clear_decode_round(&mut self) {
        for task in self.decode_list.iter_mut() {
            task.clear();
        }
        self.attention_list.clear();
    }

    fn clear_prefill_round(&mut self) {
        for task in self.prefill_list.iter_mut() {
            task.clear();
        }
        for task in self.decode_list.iter_mut() {
            task.clear();
        }
        self.attention_list.clear();
    }

    fn collect_decode_candidates(&self) -> Vec<(usize, usize)> {
        let max_decode_size = self.max_decode_size;
        self.batch_list.with(|batch_list| {
            batch_list
                .iter()
                .enumerate()
                .filter(|(_, record)| record.phase == Phase::Decode)
                .take(max_decode_size)
                .map(|(batch_index, record)| (batch_index, record.kv_index))
                .collect::<Vec<_>>()
        })
    }

    fn collect_prefill_candidates(&self) -> (Vec<PrefillCandidate>, usize) {
        self.batch_list.with(|batch_list| {
            let mut total_tokens = 0usize;
            let mut candidates = Vec::with_capacity(batch_list.len());

            for (batch_index, record) in batch_list.iter().enumerate() {
                if record.phase == Phase::Prefill {
                    let remaining = record.filling_length;
                    total_tokens += remaining;
                    candidates.push(PrefillCandidate {
                        batch_index,
                        sequence_index: record.sequence_index,
                        remaining,
                    });
                }
            }

            (candidates, total_tokens)
        })
    }

    fn next_batch_work(&self) -> BatchWork {
        self.batch_list.with(|batch_list| {
            let mut has_prefill = false;

            for record in batch_list.iter() {
                match record.phase {
                    Phase::Decode => return BatchWork::Decode,
                    Phase::Prefill => has_prefill = true,
                    _ => {}
                }
            }

            if has_prefill {
                BatchWork::Prefill
            } else {
                BatchWork::Idle
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
        let build_list = || {
            (0..thread_num)
                .map(|_| Vec::with_capacity(batch_size))
                .collect::<Vec<_>>()
        };

        Self {
            max_decode_size: batch_size,
            max_prefill_size: sequence_length * batch_size,
            batch_list: Arc::new(SharedMut::new(Vec::with_capacity(batch_size))),
            thread_num,
            decode_scheduler: SliceScheduler::new(batch_size),
            prefill_scheduler: SliceScheduler::new(batch_size * thread_num),
            prefill_list: build_list(),
            decode_list: build_list(),
            attention_list: Vec::with_capacity(batch_size),
        }
    }

    fn schedule_decode_only(&mut self, decode_count: &mut usize) {
        self.clear_decode_round();
        let decode_candidates = self.collect_decode_candidates();

        self.decode_scheduler.init(decode_candidates.len());

        for (batch_index, sequence_index) in decode_candidates {
            if self.decode_scheduler.is_done() {
                break;
            }

            let scheduled_before = *decode_count;
            let attention_length = 1usize.min(self.decode_scheduler.remaining_tokens());
            if attention_length == 0 {
                break;
            }

            self.attention_list.push(SequenceSlice {
                batch_index,
                sequence_index,
                token_start_index: scheduled_before,
                lift_index: 0,
                length: attention_length,
            });

            self.decode_scheduler.schedule_for_sequence(
                batch_index,
                sequence_index,
                1,
                0,
                &mut self.decode_list,
                decode_count,
            );
        }
    }

    fn schedule_prefill(&mut self, prefill_count: &mut usize) {
        self.clear_prefill_round();
        let (prefill_candidates, prefill_total_tokens) = self.collect_prefill_candidates();

        let total_tokens = prefill_total_tokens.min(self.max_prefill_size);
        self.prefill_scheduler.init(total_tokens);

        // decode_list is cleared above, so no decode tokens are used yet.
        self.decode_scheduler.init(0);

        let prefill_scheduler = &mut self.prefill_scheduler;
        let prefill_list = &mut self.prefill_list;
        let attention_list = &mut self.attention_list;
        self.batch_list.with_mut(|batch_list| {
            for candidate in prefill_candidates.iter().copied() {
                if prefill_scheduler.is_done() {
                    break;
                }

                let scheduled_before = *prefill_count;
                let attention_length = candidate
                    .remaining
                    .min(prefill_scheduler.remaining_tokens());
                if attention_length > 0 {
                    attention_list.push(SequenceSlice {
                        batch_index: candidate.batch_index,
                        sequence_index: candidate.sequence_index,
                        token_start_index: scheduled_before,
                        lift_index: 0,
                        length: attention_length,
                    });
                }

                prefill_scheduler.schedule_for_sequence(
                    candidate.batch_index,
                    candidate.sequence_index,
                    candidate.remaining,
                    0,
                    prefill_list,
                    prefill_count,
                );

                let scheduled_for_record = prefill_count.saturating_sub(scheduled_before);
                if let Some(record) = batch_list.get_mut(candidate.batch_index) {
                    if scheduled_for_record > 0 {
                        let last_token_index = scheduled_before + scheduled_for_record - 1;
                        let last_sequence_index =
                            candidate.sequence_index + scheduled_for_record - 1;
                        Self::push_prefill_lift_slice(
                            self.decode_list.as_mut_slice(),
                            candidate.batch_index,
                            last_sequence_index,
                            last_token_index,
                        );

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
    }

    pub fn schedule_batch(&mut self) -> (usize, usize) {
        let decode_task_count = self.thread_num.min(self.decode_list.len());
        let prefill_task_count = self.thread_num.min(self.prefill_list.len());

        self.decode_scheduler.set_task_count(decode_task_count);
        self.prefill_scheduler.set_task_count(prefill_task_count);

        let mut prefill_count = 0usize;
        let mut decode_count = 0usize;

        loop {
            match self.next_batch_work() {
                BatchWork::Decode => {
                    self.schedule_decode_only(&mut decode_count);
                    return (prefill_count, decode_count);
                }
                BatchWork::Prefill => {
                    self.schedule_prefill(&mut prefill_count);
                    return (prefill_count, decode_count);
                }
                BatchWork::Idle => {
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
        // - decode_list is empty across all tasks
        let batch_size = 32;
        let mut scheduler = BatchScheduler::new(8, batch_size, 8);
        scheduler.batch_list.with_mut(|batch_list| {
            for _ in 0..8 {
                batch_list.push(state(Phase::Prefill, 0, 6));
            }
        });

        let (prefill, decode) = scheduler.schedule_batch();

        assert_eq!(prefill, 48);
        assert_eq!(decode, 0);

        assert_eq!(scheduler.prefill_list.len(), 8);
        assert_eq!(scheduler.decode_list.len(), 8);
        assert_eq!(scheduler.attention_list.len(), 8);

        scheduler.batch_list.with(|batch_list| {
            for record in batch_list.iter().take(8) {
                assert_eq!(record.phase, Phase::Decode);
                assert_eq!(record.sequence_index, 6);
                assert_eq!(record.kv_index, 6);
                assert_eq!(record.filling_length, 0);
            }
        });

        for task_index in 0..8 {
            assert_eq!(scheduler.decode_list[task_index].len(), 1);
            let lift_slice = &scheduler.decode_list[task_index][0];
            assert_eq!(lift_slice.batch_index, task_index);
            assert_eq!(lift_slice.sequence_index, 5);
            assert_eq!(lift_slice.token_start_index, task_index * 6 + 5);
            assert_eq!(lift_slice.lift_index, task_index);
            assert_eq!(lift_slice.length, 1);
        }

        for task_index in 0..8 {
            assert_eq!(scheduler.prefill_list[task_index].len(), 1);
            let slice = &scheduler.prefill_list[task_index][0];
            assert_eq!(slice.batch_index, task_index);
            assert_eq!(slice.sequence_index, 0);
            assert_eq!(slice.token_start_index, task_index * 6);
            assert_eq!(slice.length, 6);
        }

        for sequence_offset in 0..8 {
            let attention_slice = &scheduler.attention_list[sequence_offset];
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
            assert_eq!(scheduler.decode_list[task_index].len(), 1);
            assert_eq!(scheduler.decode_list[task_index][0].length, 1);
        }
        assert_eq!(scheduler.attention_list.len(), 2);

        let first = &scheduler.decode_list[0][0];
        assert_eq!(first.batch_index, 0);
        assert_eq!(first.sequence_index, 11);
        assert_eq!(first.token_start_index, 0);

        let second = &scheduler.decode_list[1][0];
        assert_eq!(second.batch_index, 2);
        assert_eq!(second.sequence_index, 12);
        assert_eq!(second.token_start_index, 1);

        let first_attention = &scheduler.attention_list[0];
        assert_eq!(first_attention.batch_index, first.batch_index);
        assert_eq!(first_attention.sequence_index, first.sequence_index);
        assert_eq!(first_attention.token_start_index, first.token_start_index);
        assert_eq!(first_attention.length, 1);

        let second_attention = &scheduler.attention_list[1];
        assert_eq!(second_attention.batch_index, second.batch_index);
        assert_eq!(second_attention.sequence_index, second.sequence_index);
        assert_eq!(second_attention.token_start_index, second.token_start_index);
        assert_eq!(second_attention.length, 1);
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
        assert_eq!(decode, 0);

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

        assert_eq!(scheduler.attention_list.len(), 2);

        let first_attention = &scheduler.attention_list[0];
        assert_eq!(first_attention.batch_index, 0);
        assert_eq!(first_attention.sequence_index, 0);
        assert_eq!(first_attention.token_start_index, 0);
        assert_eq!(first_attention.length, 6);

        let second_attention = &scheduler.attention_list[1];
        assert_eq!(second_attention.batch_index, 1);
        assert_eq!(second_attention.sequence_index, 0);
        assert_eq!(second_attention.token_start_index, 6);
        assert_eq!(second_attention.length, 2);

        assert_eq!(scheduler.decode_list[0].len(), 1);
        let first_lift = &scheduler.decode_list[0][0];
        assert_eq!(first_lift.batch_index, 0);
        assert_eq!(first_lift.sequence_index, 5);
        assert_eq!(first_lift.token_start_index, 5);
        assert_eq!(first_lift.lift_index, 0);
        assert_eq!(first_lift.length, 1);

        assert_eq!(scheduler.decode_list[1].len(), 1);
        let second_lift = &scheduler.decode_list[1][0];
        assert_eq!(second_lift.batch_index, 1);
        assert_eq!(second_lift.sequence_index, 1);
        assert_eq!(second_lift.token_start_index, 7);
        assert_eq!(second_lift.lift_index, 1);
        assert_eq!(second_lift.length, 1);
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
        assert_eq!(decode, 0);

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
        assert_eq!(decode, 0);

        scheduler.batch_list.with(|batch_list| {
            assert_eq!(batch_list[0].phase, Phase::Eos);
            assert_eq!(batch_list[0].sequence_index, 6);
            assert_eq!(batch_list[1].phase, Phase::Decode);
            assert_eq!(batch_list[1].sequence_index, 6);
            assert_eq!(batch_list[1].filling_length, 0);
        });

        assert_eq!(scheduler.attention_list.len(), 1);
        let resumed_attention = &scheduler.attention_list[0];
        assert_eq!(resumed_attention.batch_index, 1);
        assert_eq!(resumed_attention.sequence_index, 2);
        assert_eq!(resumed_attention.token_start_index, 0);
        assert_eq!(resumed_attention.length, 4);

        assert_eq!(scheduler.decode_list[0].len(), 0);
        assert_eq!(scheduler.decode_list[1].len(), 1);
        let resumed_lift = &scheduler.decode_list[1][0];
        assert_eq!(resumed_lift.batch_index, 1);
        assert_eq!(resumed_lift.sequence_index, 5);
        assert_eq!(resumed_lift.token_start_index, 3);
        assert_eq!(resumed_lift.lift_index, 1);
        assert_eq!(resumed_lift.length, 1);

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
        assert_eq!(decode, 0);
        assert_eq!(scheduler.attention_list.len(), 1);
        assert_eq!(scheduler.attention_list[0].length, 6);
        assert_eq!(scheduler.decode_list[0].len(), 1);
        assert_eq!(scheduler.decode_list[0][0].token_start_index, 5);
        assert_eq!(scheduler.decode_list[0][0].lift_index, 0);

        scheduler.batch_list.with(|batch_list| {
            assert_eq!(batch_list[0].sequence_index, 6);
            assert_eq!(batch_list[0].kv_index, 32);
            assert_eq!(batch_list[0].filling_length, 0);
            assert_eq!(batch_list[0].phase, Phase::Decode);
        });
    }
}
