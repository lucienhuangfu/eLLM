use super::sequence_slice::{DecodeList, SequenceSlice};
use super::types::{Phase, SequenceState};

pub trait SchedulerStrategy: Send + Sync + 'static {
    fn plan_next_round(
        &self,
        batch_list: &[SequenceState],
        max_decode_size: usize,
        max_prefill_size: usize,
    ) -> BatchPlan;

    fn schedule_decode_round(
        &self,
        decode_candidates: Vec<(usize, usize)>,
        decode_list: &mut DecodeList,
    ) -> usize;

    fn schedule_prefill_round(
        &self,
        candidates: Vec<PrefillCandidate>,
        total_tokens: usize,
        prefill_list: &mut Vec<Vec<SequenceSlice>>,
        decode_list: &mut DecodeList,
        thread_num: usize,
    ) -> usize;
}

#[derive(Debug)]
pub enum BatchPlan {
    Decode(Vec<(usize, usize)>),
    Prefill {
        candidates: Vec<PrefillCandidate>,
        total_tokens: usize,
    },
    Idle,
}

#[derive(Debug, Clone, Copy)]
pub struct PrefillCandidate {
    pub batch_index: usize,
    pub sequence_index: usize,
    pub remaining: usize,
}

#[derive(Debug, Clone)]
pub struct DefaultSchedulerStrategy {
    max_decode_size: usize,
    max_prefill_size: usize,
}

impl DefaultSchedulerStrategy {
    pub fn new(max_decode_size: usize, max_prefill_size: usize) -> Self {
        Self {
            max_decode_size,
            max_prefill_size,
        }
    }
}

impl SchedulerStrategy for DefaultSchedulerStrategy {
    fn plan_next_round(
        &self,
        batch_list: &[SequenceState],
        max_decode_size: usize,
        max_prefill_size: usize,
    ) -> BatchPlan {
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
                    total_tokens += record.filling_length;
                    candidates.push(PrefillCandidate {
                        batch_index,
                        sequence_index: record.sequence_index,
                        remaining: record.filling_length,
                    });
                }
                _ => {}
            }
        }

        if !candidates.is_empty() {
            BatchPlan::Prefill {
                candidates,
                total_tokens: total_tokens.min(max_prefill_size),
            }
        } else if has_decode {
            BatchPlan::Decode(decode_candidates)
        } else {
            BatchPlan::Idle
        }
    }

    fn schedule_decode_round(
        &self,
        decode_candidates: Vec<(usize, usize)>,
        decode_list: &mut DecodeList,
    ) -> usize {
        let decode_count = decode_candidates.len();

        for (idx, (batch_index, sequence_index)) in decode_candidates.into_iter().enumerate() {
            decode_list.push(SequenceSlice {
                batch_index,
                sequence_index,
                token_start_index: idx,
                length: 1,
                last_token_flag: true,
            });
        }

        decode_count
    }

    fn schedule_prefill_round(
        &self,
        candidates: Vec<PrefillCandidate>,
        total_tokens: usize,
        prefill_list: &mut Vec<Vec<SequenceSlice>>,
        decode_list: &mut DecodeList,
        thread_num: usize,
    ) -> usize {
        let mut prefill_count = 0usize;
        let task_count = thread_num.min(prefill_list.len());

        let mut scheduler = SliceScheduler::new(task_count);
        scheduler.init(total_tokens);

        for candidate in candidates {
            if scheduler.is_done() {
                break;
            }

            let attention_length = candidate
                .remaining
                .min(scheduler.remaining_tokens());
            if attention_length > 0 {
                decode_list.push(SequenceSlice {
                    batch_index: candidate.batch_index,
                    sequence_index: candidate.sequence_index,
                    token_start_index: prefill_count,
                    length: attention_length,
                    last_token_flag: attention_length == candidate.remaining,
                });
            }

            scheduler.schedule_for_sequence(
                candidate.batch_index,
                candidate.sequence_index,
                candidate.remaining,
                0,
                prefill_list,
                &mut prefill_count,
            );
        }

        prefill_count
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

    fn set_task_count(&mut self, task_count: usize) {
        self.task_count = task_count;
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