use crate::runtime::state::sequence::{DecodeList, SequenceSlice};

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
