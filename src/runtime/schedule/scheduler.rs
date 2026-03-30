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

    fn decode_state(sequence_index: usize, kv_index: usize) -> SequenceState {
        SequenceState {
            filling_length: 0,
            phase: Phase::Decode,
            sequence_index,
            kv_index,
            notify: std::sync::Arc::new(Notify::new()),
        }
    }

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
    fn plan_next_round_returns_idle_for_empty_batch() {
        let scheduler = BatchScheduler::new(16, 4, 3);

        match scheduler.plan_next_round() {
            BatchPlan::Idle => {}
            _ => panic!("expected idle plan for an empty batch"),
        }
    }

    #[test]
    fn schedule_decode_round_limits_to_batch_size_and_prefill_is_ignored() {
        let mut scheduler = BatchScheduler::new(16, 4, 3);
        scheduler.batch_list.with_mut(|batch_list| {
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

        assert!(scheduler.prefill_list.iter().all(Vec::is_empty));
        assert_eq!(scheduler.decode_list.len(), 4);

        let expected = [(0, 128), (2, 256), (4, 384), (5, 512)];
        for (slice, &(batch_index, sequence_index)) in
            scheduler.decode_list.iter().zip(expected.iter())
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
    fn schedule_prefill_round_splits_long_sequences_across_threads() {
        let mut scheduler = BatchScheduler::new(8, 4, 3);
        scheduler.batch_list.with_mut(|batch_list| {
            batch_list.push(prefill_state(0, 10));
            batch_list.push(prefill_state(32, 9));
            batch_list.push(prefill_state(64, 4));
        });

        let (prefill, decode) = scheduler.schedule_batch();

        assert_eq!(prefill, 23);
        assert_eq!(decode, 3);
        assert_eq!(scheduler.decode_list.len(), 3);

        let decode_lengths: Vec<usize> = scheduler
            .decode_list
            .iter()
            .map(|slice| slice.length)
            .collect();
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

        assert_eq!(scheduler.prefill_list.len(), 3);
        assert_eq!(scheduler.prefill_list[0].len(), 1);
        assert_eq!(scheduler.prefill_list[1].len(), 2);
        assert_eq!(scheduler.prefill_list[2].len(), 2);

        let t0 = &scheduler.prefill_list[0][0];
        assert_eq!(t0.batch_index, 0);
        assert_eq!(t0.sequence_index, 0);
        assert_eq!(t0.token_start_index, 0);
        assert_eq!(t0.length, 8);

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
    fn schedule_prefill_round_truncates_to_window_and_marks_partial_last_slice_false() {
        let mut scheduler = BatchScheduler::new(5, 2, 2);
        scheduler.batch_list.with_mut(|batch_list| {
            batch_list.push(prefill_state(0, 4));
            batch_list.push(prefill_state(16, 5));
            batch_list.push(prefill_state(32, 6));
        });

        let (prefill, decode) = scheduler.schedule_batch();

        assert_eq!(prefill, 10);
        assert_eq!(decode, 3);
        assert_eq!(scheduler.decode_list.len(), 3);
    }
}
