use super::task::{BatchMode, ScheduleTask};
use crate::runtime::state::core::SlotState;
use crate::runtime::state::sequence::{DecodeList, SequenceSlice};
use crate::runtime::state::types::Phase;

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
}

impl PlanBuilder {
    #[inline]
    pub fn new(max_decode_size: usize, max_prefill_size: usize, thread_num: usize) -> Self {
        Self {
            max_decode_size,
            max_prefill_size,
            thread_num,
        }
    }

    pub fn build_task(&self, batch_list: &[SlotState], task: &mut ScheduleTask, task_id: u64) {
        task.reset(task_id);

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

        task.mode = match (has_prefill, has_decode) {
            (true, true) => BatchMode::Mixed,
            (true, false) => BatchMode::Prefill,
            (false, true) => BatchMode::Decode,
            (false, false) => return,
        };

        if has_decode {
            self.build_decode(task, &decode_candidates);
        }

        if has_prefill {
            self.build_prefill(task, &prefill_candidates);
        }
    }

    fn build_decode(&self, task: &mut ScheduleTask, candidates: &[(usize, usize)]) {
        task.decode_list.clear();

        for (idx, &(batch_index, sequence_index)) in candidates.iter().enumerate() {
            task.decode_list.push(SequenceSlice {
                batch_index,
                sequence_index,
                token_start_index: idx,
                length: 1,
                last_token_flag: true,
            });
        }

        task.decode_size = candidates.len();
    }

    fn build_prefill(&self, task: &mut ScheduleTask, candidates: &[PrefillCandidate]) {
        let total_tokens: usize = candidates.iter().map(|c| c.remaining).sum();
        let total_tokens = total_tokens.min(self.max_prefill_size);

        if total_tokens == 0 {
            return;
        }

        task.resize_prefill_list(self.thread_num);

        let avg_tokens_per_thread = total_tokens / self.thread_num;
        for list in task.prefill_list.iter_mut() {
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
                task.decode_list.push(SequenceSlice {
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
                &mut task.prefill_list,
                &mut prefill_count,
            );
        }

        task.prefill_size = prefill_count;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::state::types::Phase;

    #[test]
    fn test_plan_builder_new() {
        let builder = PlanBuilder::new(32, 1024, 4);
        let mut task = ScheduleTask::new(0);
        builder.build_task(&[], &mut task, 0);
        assert!(task.is_empty());
    }

    #[test]
    fn test_plan_builder_empty_batch() {
        let builder = PlanBuilder::new(32, 1024, 4);
        let mut task = ScheduleTask::new(0);
        builder.build_task(&[], &mut task, 0);
        assert!(task.is_empty());
        assert_eq!(task.mode, BatchMode::Decode);
    }

    #[test]
    fn test_plan_builder_only_start_states() {
        let builder = PlanBuilder::new(32, 1024, 4);
        let batch_list = vec![SlotState::new_start_state(), SlotState::new_start_state()];
        let mut task = ScheduleTask::new(0);
        builder.build_task(&batch_list, &mut task, 0);
        assert!(task.is_empty());
    }

    #[test]
    fn test_plan_builder_decode_only() {
        let builder = PlanBuilder::new(32, 1024, 4);

        let mut batch_list = Vec::new();
        for i in 0..5 {
            let mut state = SlotState::new_decode_state(i * 10, i * 10);
            state.phase = Phase::Decode;
            batch_list.push(state);
        }

        let mut task = ScheduleTask::new(0);
        builder.build_task(&batch_list, &mut task, 1);

        assert_eq!(task.mode, BatchMode::Decode);
        assert_eq!(task.decode_size, 5);
        assert_eq!(task.prefill_size, 0);
        assert_eq!(task.sequence_count(), 5);
        assert_eq!(task.task_id, 1);
    }

    #[test]
    fn test_plan_builder_decode_exceeds_limit() {
        let builder = PlanBuilder::new(3, 1024, 4);

        let mut batch_list = Vec::new();
        for i in 0..10 {
            let mut state = SlotState::new_decode_state(i, i);
            state.phase = Phase::Decode;
            batch_list.push(state);
        }

        let mut task = ScheduleTask::new(0);
        builder.build_task(&batch_list, &mut task, 1);

        assert_eq!(task.mode, BatchMode::Decode);
        assert_eq!(task.decode_size, 3);
    }

    #[test]
    fn test_plan_builder_prefill_only() {
        let builder = PlanBuilder::new(32, 1024, 4);

        let mut batch_list = Vec::new();
        for i in 0..3 {
            let state = SlotState::new_prefill_state(i * 10, 100);
            batch_list.push(state);
        }

        let mut task = ScheduleTask::new(0);
        builder.build_task(&batch_list, &mut task, 1);

        assert_eq!(task.mode, BatchMode::Prefill);
        assert!(task.prefill_size > 0);
        assert_eq!(task.decode_size, 0);
    }

    #[test]
    fn test_plan_builder_mixed_mode() {
        let builder = PlanBuilder::new(32, 1024, 4);

        let mut batch_list = Vec::new();
        for i in 0..2 {
            let mut state = SlotState::new_decode_state(i, i);
            state.phase = Phase::Decode;
            batch_list.push(state);
        }
        for i in 0..2 {
            let state = SlotState::new_prefill_state(i * 10 + 100, 50);
            batch_list.push(state);
        }

        let mut task = ScheduleTask::new(0);
        builder.build_task(&batch_list, &mut task, 1);

        assert_eq!(task.mode, BatchMode::Mixed);
        assert_eq!(task.decode_size, 2);
        assert!(task.prefill_size > 0);
    }

    #[test]
    fn test_plan_builder_mixed_with_inactive_states() {
        let builder = PlanBuilder::new(32, 1024, 4);

        let mut batch_list = vec![SlotState::new_start_state(), SlotState::new_start_state()];

        let mut decode_state = SlotState::new_decode_state(0, 0);
        decode_state.phase = Phase::Decode;
        batch_list.push(decode_state);

        let mut eos_state = SlotState::new_start_state();
        eos_state.phase = Phase::Eos;
        batch_list.push(eos_state);

        let mut task = ScheduleTask::new(0);
        builder.build_task(&batch_list, &mut task, 1);

        assert_eq!(task.mode, BatchMode::Decode);
        assert_eq!(task.decode_size, 1);
    }

    #[test]
    fn test_plan_builder_task_id() {
        let builder = PlanBuilder::new(32, 1024, 4);
        let mut task = ScheduleTask::new(0);
        builder.build_task(&[], &mut task, 42);
        assert_eq!(task.task_id, 42);
    }

    #[test]
    fn test_plan_builder_prefill_multi_thread() {
        let builder = PlanBuilder::new(32, 100, 4);

        let batch_list = vec![
            SlotState::new_prefill_state(0, 50),
            SlotState::new_prefill_state(50, 30),
        ];

        let mut task = ScheduleTask::new(0);
        builder.build_task(&batch_list, &mut task, 1);

        assert_eq!(task.mode, BatchMode::Prefill);
        assert!(task.prefill_size <= 100);
        assert_eq!(task.prefill_list.len(), 4);
    }

    #[test]
    fn test_plan_builder_prefill_exceeds_limit() {
        let builder = PlanBuilder::new(32, 50, 4);

        let batch_list = vec![
            SlotState::new_prefill_state(0, 100),
            SlotState::new_prefill_state(100, 100),
        ];

        let mut task = ScheduleTask::new(0);
        builder.build_task(&batch_list, &mut task, 1);

        assert!(task.prefill_size <= 50);
    }

    #[test]
    fn test_plan_builder_decode_list_content() {
        let builder = PlanBuilder::new(32, 1024, 4);

        let mut batch_list = Vec::new();
        for i in 0..3 {
            let mut state = SlotState::new_decode_state(i * 10, i * 10);
            state.phase = Phase::Decode;
            batch_list.push(state);
        }

        let mut task = ScheduleTask::new(0);
        builder.build_task(&batch_list, &mut task, 1);

        assert_eq!(task.decode_list.len(), 3);

        for (idx, slice) in task.decode_list.as_slice().iter().enumerate() {
            assert_eq!(slice.token_start_index, idx);
            assert_eq!(slice.length, 1);
            assert!(slice.last_token_flag);
        }
    }

    #[test]
    fn test_plan_builder_single_thread() {
        let builder = PlanBuilder::new(32, 100, 1);

        let batch_list = vec![SlotState::new_prefill_state(0, 50)];

        let mut task = ScheduleTask::new(0);
        builder.build_task(&batch_list, &mut task, 1);

        assert_eq!(task.prefill_list.len(), 1);
    }

    #[test]
    fn test_plan_builder_empty_prefill_candidates() {
        let builder = PlanBuilder::new(32, 100, 4);

        let mut batch_list = Vec::new();
        let mut state = SlotState::new_decode_state(0, 0);
        state.phase = Phase::Decode;
        batch_list.push(state);

        let mut task = ScheduleTask::new(0);
        builder.build_task(&batch_list, &mut task, 1);

        assert_eq!(task.mode, BatchMode::Decode);
        assert_eq!(task.prefill_size, 0);
    }

    #[test]
    fn test_plan_builder_prefill_zero_filling_length() {
        let builder = PlanBuilder::new(32, 100, 4);

        let batch_list = vec![SlotState::new_prefill_state(0, 0)];

        let mut task = ScheduleTask::new(0);
        builder.build_task(&batch_list, &mut task, 1);

        assert_eq!(task.prefill_size, 0);
    }

    #[test]
    fn test_plan_builder_multiple_calls_independent() {
        let builder = PlanBuilder::new(32, 1024, 4);

        let batch_list1 = vec![SlotState::new_prefill_state(0, 100)];
        let batch_list2 = vec![SlotState::new_decode_state(0, 0)];

        let mut task1 = ScheduleTask::new(0);
        builder.build_task(&batch_list1, &mut task1, 1);

        let mut task2 = ScheduleTask::new(0);
        builder.build_task(&batch_list2, &mut task2, 2);

        assert_eq!(task1.mode, BatchMode::Prefill);
        assert_eq!(task2.mode, BatchMode::Decode);
    }

    #[test]
    fn test_plan_builder_many_decode_candidates() {
        let builder = PlanBuilder::new(10, 1024, 4);

        let mut batch_list = Vec::new();
        for i in 0..100 {
            let mut state = SlotState::new_decode_state(i, i);
            state.phase = Phase::Decode;
            batch_list.push(state);
        }

        let mut task = ScheduleTask::new(0);
        builder.build_task(&batch_list, &mut task, 1);

        assert_eq!(task.decode_size, 10);
    }

    #[test]
    fn test_plan_builder_many_prefill_candidates() {
        let builder = PlanBuilder::new(32, 200, 4);

        let mut batch_list = Vec::new();
        for i in 0..10 {
            batch_list.push(SlotState::new_prefill_state(i * 100, 50));
        }

        let mut task = ScheduleTask::new(0);
        builder.build_task(&batch_list, &mut task, 1);

        assert!(task.prefill_size <= 200);
    }

    #[test]
    fn test_plan_builder_task_reuse() {
        let builder = PlanBuilder::new(32, 1024, 4);

        let batch_list1 = vec![SlotState::new_decode_state(0, 0)];
        let batch_list2 = vec![SlotState::new_prefill_state(0, 50)];

        let mut task = ScheduleTask::new(0);
        builder.build_task(&batch_list1, &mut task, 1);

        assert_eq!(task.mode, BatchMode::Decode);
        assert_eq!(task.decode_size, 1);

        builder.build_task(&batch_list2, &mut task, 2);

        assert_eq!(task.mode, BatchMode::Prefill);
        assert!(task.prefill_size > 0);
        assert_eq!(task.decode_size, 0);
        assert_eq!(task.task_id, 2);
    }
}
