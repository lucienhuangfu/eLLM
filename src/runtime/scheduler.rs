use std::sync::Arc;
use std::thread;
use std::time::Duration;

use super::slice_scheduler::{PrefillCandidate, SliceScheduler};
use crate::common::send_sync_ptr::SharedMut;
use crate::common::sequence_slice::{DecodeList, SequenceSlice};
use crate::common::state::{Phase, SequenceState};

pub struct BatchScheduler {
    pub prefill_list: Vec<Vec<SequenceSlice>>,
    /// Attention/decode slices for the current round.
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
        // The prefill buffers are built once and then cleared/reused every round.
        // Each thread gets a preallocated slice buffer, so the total reserved
        // capacity is thread_num * batch_size.
        let build_prefill_list = || {
            let mut prefill_list = Vec::with_capacity(thread_num);
            for _ in 0..thread_num {
                prefill_list.push(Vec::with_capacity(batch_size));
            }
            prefill_list
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
        // Reuse the already allocated output buffers for the next scheduling round.
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
                            decode_candidates.push((batch_index, record.sequence_index));
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
    fn schedule_decode_round_uses_one_decode_sequence() {
        let mut scheduler = BatchScheduler::new(16, 4, 3);
        scheduler.batch_list.with_mut(|batch_list| {
            batch_list.push(decode_state(100, 128));
        });

        let (prefill, decode_tokens) = scheduler.schedule_batch();

        assert_eq!(prefill, 0);
        assert_eq!(decode_tokens, 1);

        assert!(scheduler.prefill_list.iter().all(Vec::is_empty));
        assert_eq!(scheduler.decode_list.len(), 1);

        let slice = &scheduler.decode_list[0];
        assert_eq!(slice.batch_index, 0);
        assert_eq!(slice.sequence_index, 100);
        assert_eq!(slice.token_start_index, 0);
        assert_eq!(slice.length, 1);
        assert!(slice.last_token_flag);
    }

    #[test]
    fn schedule_prefill_round_limits_one_sequence_to_max_prefill_size() {
        let mut scheduler = BatchScheduler::new(8, 4, 3);
        scheduler.batch_list.with_mut(|batch_list| {
            batch_list.push(prefill_state(0, 23));
        });

        let (prefill_tokens, decode_slices) = scheduler.schedule_batch();

        assert_eq!(prefill_tokens, 23.min(8 * 4));
        assert_eq!(decode_slices, 1);
        // Prefill rounds still populate decode_list: the last token path is
        // consumed by the attention/decode side of the pipeline.
        assert_eq!(scheduler.decode_list.len(), 1);

        let attention_slice = &scheduler.decode_list[0];
        assert_eq!(attention_slice.batch_index, 0);
        assert_eq!(attention_slice.sequence_index, 0);
        assert_eq!(attention_slice.token_start_index, 0);
        assert_eq!(attention_slice.length, 23);
        assert!(attention_slice.last_token_flag);

        assert_eq!(scheduler.prefill_list.len(), 3);
        assert_eq!(scheduler.prefill_list[0].len(), 1);
        assert_eq!(scheduler.prefill_list[1].len(), 1);
        assert_eq!(scheduler.prefill_list[2].len(), 1);

        let t0 = &scheduler.prefill_list[0][0];
        assert_eq!(t0.batch_index, 0);
        assert_eq!(t0.sequence_index, 0);
        assert_eq!(t0.token_start_index, 0);
        assert_eq!(t0.length, 8);

        let t1 = &scheduler.prefill_list[1][0];
        assert_eq!(t1.batch_index, 0);
        assert_eq!(t1.sequence_index, 8);
        assert_eq!(t1.token_start_index, 8);
        assert_eq!(t1.length, 8);

        let t2 = &scheduler.prefill_list[2][0];
        assert_eq!(t2.batch_index, 0);
        assert_eq!(t2.sequence_index, 16);
        assert_eq!(t2.token_start_index, 16);
        assert_eq!(t2.length, 7);
    }

    #[test]
    fn schedule_prefill_round_truncates_to_max_prefill_size() {
        let mut scheduler = BatchScheduler::new(5, 2, 2);
        scheduler.batch_list.with_mut(|batch_list| {
            batch_list.push(prefill_state(0, 13));
        });

        let (prefill_tokens, decode_slices) = scheduler.schedule_batch();

        assert_eq!(prefill_tokens, 10);
        assert_eq!(decode_slices, 1);
        // Even when prefill is truncated, the returned decode_list carries the
        // slice for the last-token path of that prefill request.
        assert_eq!(scheduler.decode_list.len(), 1);

        let attention_slice = &scheduler.decode_list[0];
        assert_eq!(attention_slice.batch_index, 0);
        assert_eq!(attention_slice.sequence_index, 0);
        assert_eq!(attention_slice.token_start_index, 0);
        assert_eq!(attention_slice.length, 10);
        assert!(!attention_slice.last_token_flag);

        assert_eq!(scheduler.prefill_list.len(), 2);
        assert_eq!(scheduler.prefill_list[0].len(), 1);
        assert_eq!(scheduler.prefill_list[1].len(), 1);

        let first = &scheduler.prefill_list[0][0];
        assert_eq!(first.batch_index, 0);
        assert_eq!(first.sequence_index, 0);
        assert_eq!(first.token_start_index, 0);
        assert_eq!(first.length, 5);
        assert!(!first.last_token_flag);

        let second = &scheduler.prefill_list[1][0];
        assert_eq!(second.batch_index, 0);
        assert_eq!(second.sequence_index, 5);
        assert_eq!(second.token_start_index, 5);
        assert_eq!(second.length, 5);
        assert!(!second.last_token_flag);
    }

    #[test]
    fn schedule_batch_prefers_decode_when_both_phases_exist() {
        let mut scheduler = BatchScheduler::new(16, 4, 3);
        scheduler.batch_list.with_mut(|batch_list| {
            batch_list.push(prefill_state(0, 6));
            batch_list.push(decode_state(100, 128));
            batch_list.push(prefill_state(32, 3));
        });

        let (prefill_tokens, decode_tokens) = scheduler.schedule_batch();

        assert_eq!(prefill_tokens, 0);
        assert_eq!(decode_tokens, 1);
        assert!(scheduler.prefill_list.iter().all(Vec::is_empty));
        assert_eq!(scheduler.decode_list.len(), 1);

        let slice = &scheduler.decode_list[0];
        assert_eq!(slice.batch_index, 1);
        assert_eq!(slice.sequence_index, 100);
        assert_eq!(slice.token_start_index, 0);
        assert_eq!(slice.length, 1);
        assert!(slice.last_token_flag);
    }
}
