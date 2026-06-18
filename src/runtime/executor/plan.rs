use std::sync::Arc;

use crate::runtime::state::sequence::{DecodeList, SequenceSlice};
use crate::runtime::state::types::{Phase, SequenceState};

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

    pub fn sequence_count(&self) -> usize {
        self.decode_size + (if self.mode == BatchMode::Prefill || self.mode == BatchMode::Mixed { 1 } else { 0 })
    }

    pub fn is_empty(&self) -> bool {
        self.prefill_size == 0 && self.decode_size == 0
    }
}

pub struct PlanBuilder {
    max_decode_size: usize,
    max_prefill_size: usize,
    thread_num: usize,
    next_task_id: u64,
}

impl PlanBuilder {
    pub fn new(max_decode_size: usize, max_prefill_size: usize, thread_num: usize) -> Self {
        Self {
            max_decode_size,
            max_prefill_size,
            thread_num,
            next_task_id: 1,
        }
    }

    pub fn build_plan(&mut self, batch_list: &[SequenceState]) -> BatchPlan {
        let mut plan = BatchPlan::new(self.next_task_id);
        self.next_task_id += 1;

        let mut decode_candidates: Vec<(usize, usize)> = Vec::with_capacity(self.max_decode_size);
        let mut prefill_candidates: Vec<(usize, usize, usize)> = Vec::new();
        let mut has_decode = false;

        for (batch_index, record) in batch_list.iter().enumerate() {
            match record.phase {
                Phase::Decode => {
                    has_decode = true;
                    if decode_candidates.len() < self.max_decode_size {
                        decode_candidates.push((batch_index, record.sequence_index));
                    }
                }
                Phase::Prefill => {
                    prefill_candidates.push((batch_index, record.sequence_index, record.filling_length));
                }
                _ => {}
            }
        }

        let has_prefill = !prefill_candidates.is_empty();

        if has_prefill && has_decode {
            plan.mode = BatchMode::Mixed;
        } else if has_prefill {
            plan.mode = BatchMode::Prefill;
        } else if has_decode {
            plan.mode = BatchMode::Decode;
        } else {
            return plan;
        }

        if has_prefill {
            self.build_prefill(&mut plan, &prefill_candidates);
        }

        if has_decode {
            self.build_decode(&mut plan, &decode_candidates);
        }

        plan
    }

    fn build_decode(&self, plan: &mut BatchPlan, candidates: &[(usize, usize)]) {
        plan.decode_list.clear();

        for (idx, (batch_index, sequence_index)) in candidates.iter().enumerate() {
            plan.decode_list.push(SequenceSlice {
                batch_index: *batch_index,
                sequence_index: *sequence_index,
                token_start_index: idx,
                length: 1,
                last_token_flag: true,
            });
        }

        plan.decode_size = candidates.len();
    }

    fn build_prefill(&self, plan: &mut BatchPlan, candidates: &[(usize, usize, usize)]) {
        let total_tokens: usize = candidates.iter().map(|c| c.2).sum();
        let total_tokens = total_tokens.min(self.max_prefill_size);

        plan.prefill_list.resize_with(self.thread_num, || Vec::new());

        let mut prefill_count = 0usize;
        let task_count = self.thread_num.min(plan.prefill_list.len());

        let mut scheduler = SliceScheduler::new(task_count);
        scheduler.init(total_tokens);

        for &(batch_index, sequence_index, remaining) in candidates {
            if scheduler.is_done() {
                break;
            }

            let attention_length = remaining.min(scheduler.remaining_tokens());
            if attention_length > 0 {
                plan.decode_list.push(SequenceSlice {
                    batch_index,
                    sequence_index,
                    token_start_index: prefill_count,
                    length: attention_length,
                    last_token_flag: attention_length == remaining,
                });
            }

            scheduler.schedule_for_sequence(
                batch_index,
                sequence_index,
                remaining,
                0,
                &mut plan.prefill_list,
                &mut prefill_count,
            );
        }

        plan.prefill_size = prefill_count;
    }
}

struct SliceScheduler {
    task_count: usize,
    total_tokens: usize,
    scheduled_tokens: usize,
    current_task: usize,
    current_task_remaining: usize,
}

impl SliceScheduler {
    fn new(task_count: usize) -> Self {
        Self {
            task_count,
            total_tokens: 0,
            scheduled_tokens: 0,
            current_task: 0,
            current_task_remaining: 0,
        }
    }

    fn init(&mut self, total_tokens: usize) {
        self.total_tokens = total_tokens;
        self.scheduled_tokens = 0;
        self.current_task = 0;
        self.current_task_remaining = if total_tokens > 0 {
            self.quota_for(0)
        } else {
            0
        };
    }

    fn is_done(&self) -> bool {
        self.scheduled_tokens >= self.total_tokens
    }

    fn remaining_tokens(&self) -> usize {
        self.total_tokens.saturating_sub(self.scheduled_tokens)
    }

    fn active_task_count(&self) -> usize {
        let base_quota = self.total_tokens / self.task_count;
        if base_quota == 0 {
            self.total_tokens.min(self.task_count)
        } else {
            self.task_count
        }
    }

    fn quota_for(&self, task_index: usize) -> usize {
        let base_quota = self.total_tokens / self.task_count;
        let extra_quota = self.total_tokens % self.task_count;
        if task_index < extra_quota {
            base_quota + 1
        } else {
            base_quota
        }
    }

    fn advance_to_next_task(&mut self) {
        while self.current_task < self.active_task_count() && self.current_task_remaining == 0 {
            self.current_task += 1;
            if self.current_task < self.active_task_count() {
                self.current_task_remaining = self.quota_for(self.current_task);
            }
        }
    }

    fn current_task_index(&mut self) -> Option<usize> {
        if self.is_done() {
            return None;
        }

        self.advance_to_next_task();

        if self.current_task >= self.active_task_count() {
            None
        } else {
            Some(self.current_task)
        }
    }

    fn take(&mut self, max_take: usize) -> usize {
        if self.is_done() || max_take == 0 {
            return 0;
        }

        let available = self.total_tokens - self.scheduled_tokens;
        let take = max_take.min(available).min(self.current_task_remaining);
        if take == 0 {
            return 0;
        }

        self.scheduled_tokens += take;
        self.current_task_remaining -= take;
        take
    }

    fn schedule_for_sequence(
        &mut self,
        batch_index: usize,
        sequence_index: usize,
        mut remaining: usize,
        token_offset: usize,
        slice_list: &mut Vec<Vec<SequenceSlice>>,
        token_count: &mut usize,
    ) {
        if self.is_done() {
            return;
        }

        let mut sequence_cursor = sequence_index;

        while remaining > 0 && !self.is_done() {
            let Some(task_index) = self.current_task_index() else {
                break;
            };

            let token_start_index = token_offset + self.scheduled_tokens;
            let take = self.take(remaining);
            if take == 0 {
                break;
            }

            slice_list[task_index].push(SequenceSlice {
                batch_index,
                sequence_index: sequence_cursor,
                token_start_index,
                length: take,
                last_token_flag: false,
            });

            *token_count += take;
            remaining -= take;
            sequence_cursor += take;
        }
    }
}