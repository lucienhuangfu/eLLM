use std::sync::atomic::{AtomicU64, Ordering};

use crate::runtime::state::core::SlotState;
use crate::runtime::state::sequence::{DecodeList, SequenceSlice};
use crate::runtime::state::types::Phase;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchMode {
    Decode,
    Prefill,
    Mixed,
}

#[derive(Debug)]
pub struct BatchPlan {
    pub mode: BatchMode,
    pub prefill_size: usize,
    pub decode_size: usize,
    pub prefill_list: Vec<Vec<SequenceSlice>>,
    pub decode_list: DecodeList,
    pub task_id: u64,
}

impl BatchPlan {
    #[inline]
    pub fn new(task_id: u64) -> Self {
        Self {
            mode: BatchMode::Decode,
            prefill_size: 0,
            decode_size: 0,
            prefill_list: Vec::new(),
            decode_list: DecodeList::with_capacity(0),
            task_id,
        }
    }

    #[inline]
    pub fn sequence_count(&self) -> usize {
        self.decode_size
            + (if self.mode == BatchMode::Prefill || self.mode == BatchMode::Mixed {
                1
            } else {
                0
            })
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.prefill_size == 0 && self.decode_size == 0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PrefillCandidate {
    pub batch_index: usize,
    pub sequence_index: usize,
    pub remaining: usize,
}

pub struct PlanBuilder {
    max_decode_size: usize,
    max_prefill_size: usize,
    thread_num: usize,
    next_task_id: AtomicU64,
}

impl PlanBuilder {
    #[inline]
    pub fn new(max_decode_size: usize, max_prefill_size: usize, thread_num: usize) -> Self {
        Self {
            max_decode_size,
            max_prefill_size,
            thread_num,
            next_task_id: AtomicU64::new(1),
        }
    }

    pub fn build_plan(&self, batch_list: &[SlotState]) -> BatchPlan {
        let mut plan = BatchPlan::new(self.next_task_id.fetch_add(1, Ordering::Relaxed));

        let mut decode_candidates = Vec::with_capacity(self.max_decode_size.min(batch_list.len()));
        let mut prefill_candidates = Vec::new();
        let mut has_decode = false;
        let mut has_prefill = false;

        for (batch_index, record) in batch_list.iter().enumerate() {
            match record.phase {
                Phase::Decode if decode_candidates.len() < self.max_decode_size => {
                    has_decode = true;
                    decode_candidates.push((batch_index, record.sequence_index));
                }
                Phase::Prefill => {
                    has_prefill = true;
                    prefill_candidates.push(PrefillCandidate {
                        batch_index,
                        sequence_index: record.sequence_index,
                        remaining: record.filling_length,
                    });
                }
                Phase::Decode => has_decode = true,
                _ => {}
            }
        }

        plan.mode = match (has_prefill, has_decode) {
            (true, true) => BatchMode::Mixed,
            (true, false) => BatchMode::Prefill,
            (false, true) => BatchMode::Decode,
            (false, false) => return plan,
        };

        if has_decode {
            self.build_decode(&mut plan, &decode_candidates);
        }

        if has_prefill {
            self.build_prefill(&mut plan, &prefill_candidates);
        }

        plan
    }

    fn build_decode(&self, plan: &mut BatchPlan, candidates: &[(usize, usize)]) {
        plan.decode_list.clear();

        for (idx, &(batch_index, sequence_index)) in candidates.iter().enumerate() {
            plan.decode_list.push(SequenceSlice {
                batch_index,
                sequence_index,
                token_start_index: idx,
                length: 1,
                last_token_flag: true,
            });
        }

        plan.decode_size = candidates.len();
    }

    fn build_prefill(&self, plan: &mut BatchPlan, candidates: &[PrefillCandidate]) {
        let total_tokens: usize = candidates.iter().map(|c| c.remaining).sum();
        let total_tokens = total_tokens.min(self.max_prefill_size);

        if total_tokens == 0 {
            return;
        }

        plan.prefill_list
            .resize_with(self.thread_num, || Vec::new());

        let avg_tokens_per_thread = total_tokens / self.thread_num;
        for list in plan.prefill_list.iter_mut() {
            list.reserve(avg_tokens_per_thread.saturating_add(1));
        }

        let mut scheduler = SliceScheduler::new(self.thread_num, total_tokens);
        let mut prefill_count = 0usize;

        for &candidate in candidates {
            if scheduler.is_done() {
                break;
            }

            let attention_length = candidate.remaining.min(scheduler.remaining_tokens());
            if attention_length > 0 {
                plan.decode_list.push(SequenceSlice {
                    batch_index: candidate.batch_index,
                    sequence_index: candidate.sequence_index,
                    token_start_index: prefill_count,
                    length: attention_length,
                    last_token_flag: attention_length == candidate.remaining,
                });
            }

            scheduler.schedule_sequence(
                candidate.batch_index,
                candidate.sequence_index,
                candidate.remaining,
                &mut plan.prefill_list,
                &mut prefill_count,
            );
        }

        plan.prefill_size = prefill_count;
    }
}

struct SliceScheduler {
    thread_num: usize,
    total_tokens: usize,
    scheduled_tokens: usize,
    quotas: Vec<usize>,
    current_thread: usize,
}

impl SliceScheduler {
    fn new(thread_num: usize, total_tokens: usize) -> Self {
        let base_quota = total_tokens / thread_num;
        let extra_quota = total_tokens % thread_num;

        let quotas: Vec<usize> = (0..thread_num)
            .map(|i| base_quota + if i < extra_quota { 1 } else { 0 })
            .collect();

        Self {
            thread_num,
            total_tokens,
            scheduled_tokens: 0,
            quotas,
            current_thread: 0,
        }
    }

    #[inline]
    fn is_done(&self) -> bool {
        self.scheduled_tokens >= self.total_tokens
    }

    #[inline]
    fn remaining_tokens(&self) -> usize {
        self.total_tokens - self.scheduled_tokens
    }

    fn schedule_sequence(
        &mut self,
        batch_index: usize,
        sequence_index: usize,
        mut remaining: usize,
        prefill_list: &mut [Vec<SequenceSlice>],
        prefill_count: &mut usize,
    ) {
        let mut sequence_cursor = sequence_index;

        while remaining > 0 && !self.is_done() {
            while self.current_thread < self.thread_num && self.quotas[self.current_thread] == 0 {
                self.current_thread += 1;
            }

            if self.current_thread >= self.thread_num {
                break;
            }

            let available = self.quotas[self.current_thread]
                .min(remaining)
                .min(self.remaining_tokens());
            if available == 0 {
                break;
            }

            prefill_list[self.current_thread].push(SequenceSlice {
                batch_index,
                sequence_index: sequence_cursor,
                token_start_index: *prefill_count,
                length: available,
                last_token_flag: false,
            });

            *prefill_count += available;
            self.quotas[self.current_thread] -= available;
            self.scheduled_tokens += available;
            remaining -= available;
            sequence_cursor += available;
        }
    }
}
