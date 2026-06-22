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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// 测试 BatchMode 枚举的相等性比较
    #[test]
    fn test_batch_mode_equality() {
        assert_eq!(BatchMode::Decode, BatchMode::Decode);
        assert_eq!(BatchMode::Prefill, BatchMode::Prefill);
        assert_eq!(BatchMode::Mixed, BatchMode::Mixed);
        assert_ne!(BatchMode::Decode, BatchMode::Prefill);
        assert_ne!(BatchMode::Prefill, BatchMode::Mixed);
        assert_ne!(BatchMode::Mixed, BatchMode::Decode);
    }

    /// 测试 BatchMode 枚举的 Debug 实现
    #[test]
    fn test_batch_mode_debug() {
        assert!(format!("{:?}", BatchMode::Decode).contains("Decode"));
        assert!(format!("{:?}", BatchMode::Prefill).contains("Prefill"));
        assert!(format!("{:?}", BatchMode::Mixed).contains("Mixed"));
    }

    /// 测试 BatchMode 枚举的 Copy 特性
    #[test]
    fn test_batch_mode_copy() {
        let mode = BatchMode::Decode;
        let copied = mode;
        assert_eq!(mode, copied);
    }

    /// 测试 BatchPlan::new 创建空计划
    #[test]
    fn test_batch_plan_new() {
        let plan = BatchPlan::new(42);
        assert_eq!(plan.mode, BatchMode::Decode);
        assert_eq!(plan.prefill_size, 0);
        assert_eq!(plan.decode_size, 0);
        assert_eq!(plan.task_id, 42);
        assert!(plan.prefill_list.is_empty());
        assert!(plan.is_empty());
    }

    /// 测试 BatchPlan::sequence_count 空计划
    #[test]
    fn test_batch_plan_sequence_count_empty() {
        let plan = BatchPlan::new(1);
        assert_eq!(plan.sequence_count(), 0);
    }

    /// 测试 BatchPlan::sequence_count 仅 Decode 模式
    #[test]
    fn test_batch_plan_sequence_count_decode_only() {
        let mut plan = BatchPlan::new(1);
        plan.mode = BatchMode::Decode;
        plan.decode_size = 5;
        assert_eq!(plan.sequence_count(), 5);
    }

    /// 测试 BatchPlan::sequence_count 仅 Prefill 模式
    #[test]
    fn test_batch_plan_sequence_count_prefill_only() {
        let mut plan = BatchPlan::new(1);
        plan.mode = BatchMode::Prefill;
        plan.prefill_size = 10;
        assert_eq!(plan.sequence_count(), 1); // Prefill 模式计数为 1
    }

    /// 测试 BatchPlan::sequence_count Mixed 模式
    #[test]
    fn test_batch_plan_sequence_count_mixed() {
        let mut plan = BatchPlan::new(1);
        plan.mode = BatchMode::Mixed;
        plan.prefill_size = 10;
        plan.decode_size = 3;
        assert_eq!(plan.sequence_count(), 4); // decode_size + 1
    }

    /// 测试 BatchPlan::is_empty
    #[test]
    fn test_batch_plan_is_empty() {
        let mut plan = BatchPlan::new(1);
        assert!(plan.is_empty());

        plan.decode_size = 1;
        assert!(!plan.is_empty());

        plan.decode_size = 0;
        plan.prefill_size = 1;
        assert!(!plan.is_empty());
    }

    /// 测试 BatchPlan::is_empty 边界情况
    #[test]
    fn test_batch_plan_is_empty_boundary() {
        let mut plan = BatchPlan::new(1);
        plan.mode = BatchMode::Mixed;
        assert!(plan.is_empty());

        plan.prefill_size = 0;
        plan.decode_size = 0;
        assert!(plan.is_empty());
    }

    /// 测试 PrefillCandidate 创建和字段访问
    #[test]
    fn test_prefill_candidate() {
        let candidate = PrefillCandidate {
            batch_index: 2,
            sequence_index: 100,
            remaining: 50,
        };
        assert_eq!(candidate.batch_index, 2);
        assert_eq!(candidate.sequence_index, 100);
        assert_eq!(candidate.remaining, 50);
    }

    /// 测试 PrefillCandidate Copy 特性
    #[test]
    fn test_prefill_candidate_copy() {
        let candidate = PrefillCandidate {
            batch_index: 1,
            sequence_index: 10,
            remaining: 20,
        };
        let copied = candidate;
        assert_eq!(candidate.batch_index, copied.batch_index);
        assert_eq!(candidate.sequence_index, copied.sequence_index);
        assert_eq!(candidate.remaining, copied.remaining);
    }

    /// 测试 PlanBuilder::new
    #[test]
    fn test_plan_builder_new() {
        let builder = PlanBuilder::new(32, 1024, 4);
        // 验证 builder 创建成功（内部字段为私有）
        let plan = builder.build_plan(&[]);
        assert!(plan.is_empty());
    }

    /// 测试 PlanBuilder::build_plan 空批次列表
    #[test]
    fn test_plan_builder_empty_batch() {
        let builder = PlanBuilder::new(32, 1024, 4);
        let plan = builder.build_plan(&[]);
        assert!(plan.is_empty());
        assert_eq!(plan.mode, BatchMode::Decode); // 默认模式
    }

    /// 测试 PlanBuilder::build_plan 仅包含 Start 状态
    #[test]
    fn test_plan_builder_only_start_states() {
        let builder = PlanBuilder::new(32, 1024, 4);
        let batch_list = vec![
            SlotState::new_start_state(),
            SlotState::new_start_state(),
        ];
        let plan = builder.build_plan(&batch_list);
        assert!(plan.is_empty());
    }

    /// 测试 PlanBuilder::build_plan 仅 Decode 状态
    #[test]
    fn test_plan_builder_decode_only() {
        let builder = PlanBuilder::new(32, 1024, 4);

        let mut batch_list = Vec::new();
        for i in 0..5 {
            let mut state = SlotState::new_decode_state(i * 10, i * 10);
            state.phase = Phase::Decode;
            batch_list.push(state);
        }

        let plan = builder.build_plan(&batch_list);
        assert_eq!(plan.mode, BatchMode::Decode);
        assert_eq!(plan.decode_size, 5);
        assert_eq!(plan.prefill_size, 0);
        assert_eq!(plan.sequence_count(), 5);
    }

    /// 测试 PlanBuilder::build_plan Decode 状态超出限制
    #[test]
    fn test_plan_builder_decode_exceeds_limit() {
        let builder = PlanBuilder::new(3, 1024, 4); // max_decode_size = 3

        let mut batch_list = Vec::new();
        for i in 0..10 {
            let mut state = SlotState::new_decode_state(i, i);
            state.phase = Phase::Decode;
            batch_list.push(state);
        }

        let plan = builder.build_plan(&batch_list);
        assert_eq!(plan.mode, BatchMode::Decode);
        assert_eq!(plan.decode_size, 3); // 应该被限制为 max_decode_size
    }

    /// 测试 PlanBuilder::build_plan 仅 Prefill 状态
    #[test]
    fn test_plan_builder_prefill_only() {
        let builder = PlanBuilder::new(32, 1024, 4);

        let mut batch_list = Vec::new();
        for i in 0..3 {
            let state = SlotState::new_prefill_state(i * 10, 100);
            batch_list.push(state);
        }

        let plan = builder.build_plan(&batch_list);
        assert_eq!(plan.mode, BatchMode::Prefill);
        assert!(plan.prefill_size > 0);
        assert_eq!(plan.decode_size, 0);
    }

    /// 测试 PlanBuilder::build_plan Mixed 模式
    #[test]
    fn test_plan_builder_mixed_mode() {
        let builder = PlanBuilder::new(32, 1024, 4);

        let mut batch_list = Vec::new();
        // 添加 Decode 状态
        for i in 0..2 {
            let mut state = SlotState::new_decode_state(i, i);
            state.phase = Phase::Decode;
            batch_list.push(state);
        }
        // 添加 Prefill 状态
        for i in 0..2 {
            let state = SlotState::new_prefill_state(i * 10 + 100, 50);
            batch_list.push(state);
        }

        let plan = builder.build_plan(&batch_list);
        assert_eq!(plan.mode, BatchMode::Mixed);
        assert_eq!(plan.decode_size, 2);
        assert!(plan.prefill_size > 0);
    }

    /// 测试 PlanBuilder::build_plan 混合状态包含 Start/Eos
    #[test]
    fn test_plan_builder_mixed_with_inactive_states() {
        let builder = PlanBuilder::new(32, 1024, 4);

        let mut batch_list = vec![
            SlotState::new_start_state(),
            SlotState::new_start_state(),
        ];

        // 添加 Decode 状态
        let mut decode_state = SlotState::new_decode_state(0, 0);
        decode_state.phase = Phase::Decode;
        batch_list.push(decode_state);

        // 添加 Eos 状态（应该被忽略）
        let mut eos_state = SlotState::new_start_state();
        eos_state.phase = Phase::Eos;
        batch_list.push(eos_state);

        let plan = builder.build_plan(&batch_list);
        assert_eq!(plan.mode, BatchMode::Decode);
        assert_eq!(plan.decode_size, 1);
    }

    /// 测试 PlanBuilder task_id 递增
    #[test]
    fn test_plan_builder_task_id_increment() {
        let builder = PlanBuilder::new(32, 1024, 4);

        let plan1 = builder.build_plan(&[]);
        let plan2 = builder.build_plan(&[]);
        let plan3 = builder.build_plan(&[]);

        assert!(plan2.task_id > plan1.task_id);
        assert!(plan3.task_id > plan2.task_id);
    }

    /// 测试 PlanBuilder 多线程预填充分配
    #[test]
    fn test_plan_builder_prefill_multi_thread() {
        let builder = PlanBuilder::new(32, 100, 4); // 4 threads, max 100 prefill tokens

        let batch_list = vec![
            SlotState::new_prefill_state(0, 50),
            SlotState::new_prefill_state(50, 30),
        ];

        let plan = builder.build_plan(&batch_list);
        assert_eq!(plan.mode, BatchMode::Prefill);
        assert!(plan.prefill_size <= 100);
        assert_eq!(plan.prefill_list.len(), 4); // 4 threads
    }

    /// 测试 PlanBuilder 预填充超出限制
    #[test]
    fn test_plan_builder_prefill_exceeds_limit() {
        let builder = PlanBuilder::new(32, 50, 4); // max_prefill_size = 50

        let batch_list = vec![
            SlotState::new_prefill_state(0, 100),
            SlotState::new_prefill_state(100, 100),
        ];

        let plan = builder.build_plan(&batch_list);
        assert!(plan.prefill_size <= 50);
    }

    /// 测试 SliceScheduler 创建和配额分配
    #[test]
    fn test_slice_scheduler_quotas() {
        // 通过 PlanBuilder 间接测试 SliceScheduler
        let builder = PlanBuilder::new(32, 100, 4); // 4 threads, 100 tokens

        let batch_list = vec![SlotState::new_prefill_state(0, 100)];

        let plan = builder.build_plan(&batch_list);
        // 验证总预填充量不超过限制
        assert!(plan.prefill_size <= 100);
    }

    /// 测试 PlanBuilder decode_list 内容正确性
    #[test]
    fn test_plan_builder_decode_list_content() {
        let builder = PlanBuilder::new(32, 1024, 4);

        let mut batch_list = Vec::new();
        for i in 0..3 {
            let mut state = SlotState::new_decode_state(i * 10, i * 10);
            state.phase = Phase::Decode;
            batch_list.push(state);
        }

        let plan = builder.build_plan(&batch_list);
        assert_eq!(plan.decode_list.len(), 3);

        for (idx, slice) in plan.decode_list.as_slice().iter().enumerate() {
            assert_eq!(slice.token_start_index, idx);
            assert_eq!(slice.length, 1);
            assert!(slice.last_token_flag);
        }
    }

    /// 测试 PlanBuilder 线程数为 1 的边界情况
    #[test]
    fn test_plan_builder_single_thread() {
        let builder = PlanBuilder::new(32, 100, 1);

        let batch_list = vec![SlotState::new_prefill_state(0, 50)];

        let plan = builder.build_plan(&batch_list);
        assert_eq!(plan.prefill_list.len(), 1);
    }

    /// 测试 PlanBuilder 空预填充候选
    #[test]
    fn test_plan_builder_empty_prefill_candidates() {
        let builder = PlanBuilder::new(32, 100, 4);

        // 只有 Decode 状态，没有 Prefill
        let mut batch_list = Vec::new();
        let mut state = SlotState::new_decode_state(0, 0);
        state.phase = Phase::Decode;
        batch_list.push(state);

        let plan = builder.build_plan(&batch_list);
        assert_eq!(plan.mode, BatchMode::Decode);
        assert_eq!(plan.prefill_size, 0);
    }

    /// 测试 PlanBuilder 预填充 filling_length 为 0
    #[test]
    fn test_plan_builder_prefill_zero_filling_length() {
        let builder = PlanBuilder::new(32, 100, 4);

        let batch_list = vec![SlotState::new_prefill_state(0, 0)];

        let plan = builder.build_plan(&batch_list);
        // filling_length 为 0 时应该被跳过
        assert_eq!(plan.prefill_size, 0);
    }

    /// 测试 BatchPlan Debug 实现
    #[test]
    fn test_batch_plan_debug() {
        let plan = BatchPlan::new(1);
        let debug_str = format!("{:?}", plan);
        assert!(debug_str.contains("BatchPlan"));
        assert!(debug_str.contains("task_id"));
    }

    /// 测试 PrefillCandidate Debug 实现
    #[test]
    fn test_prefill_candidate_debug() {
        let candidate = PrefillCandidate {
            batch_index: 1,
            sequence_index: 10,
            remaining: 20,
        };
        let debug_str = format!("{:?}", candidate);
        assert!(debug_str.contains("PrefillCandidate"));
    }

    /// 测试 PlanBuilder 多次调用独立性
    #[test]
    fn test_plan_builder_multiple_calls_independent() {
        let builder = PlanBuilder::new(32, 1024, 4);

        let batch_list1 = vec![SlotState::new_prefill_state(0, 100)];
        let batch_list2 = vec![SlotState::new_decode_state(0, 0)];

        let plan1 = builder.build_plan(&batch_list1);
        let plan2 = builder.build_plan(&batch_list2);

        assert_eq!(plan1.mode, BatchMode::Prefill);
        assert_eq!(plan2.mode, BatchMode::Decode);
    }

    /// 测试 PlanBuilder 大量 Decode 候选
    #[test]
    fn test_plan_builder_many_decode_candidates() {
        let builder = PlanBuilder::new(10, 1024, 4); // max_decode_size = 10

        let mut batch_list = Vec::new();
        for i in 0..100 {
            let mut state = SlotState::new_decode_state(i, i);
            state.phase = Phase::Decode;
            batch_list.push(state);
        }

        let plan = builder.build_plan(&batch_list);
        assert_eq!(plan.decode_size, 10); // 应该被限制
    }

    /// 测试 PlanBuilder 大量 Prefill 候选
    #[test]
    fn test_plan_builder_many_prefill_candidates() {
        let builder = PlanBuilder::new(32, 200, 4); // max_prefill_size = 200

        let mut batch_list = Vec::new();
        for i in 0..10 {
            batch_list.push(SlotState::new_prefill_state(i * 100, 50));
        }

        let plan = builder.build_plan(&batch_list);
        assert!(plan.prefill_size <= 200);
    }
}
