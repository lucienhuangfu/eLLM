use crate::kernel::common::heap::FixedMinHeap;
use crate::num_traits::Exp;
use std::ops::{AddAssign, Div, Sub};

// Router 只保留 top-k 个专家，故权重在选出的子集内归一化（对应 HF 的
// `norm_topk_prob = true`）——本仓库唯一使用的语义，旧的二态开关已删。
pub fn experts_topk_softmax_norm<
    T: Exp + Default + AddAssign + PartialOrd + Copy + Sub<Output = T> + Div<Output = T>,
>(
    input_ptr: *const T,
    topk_values_ptr: *mut T,
    topk_indices_ptr: *mut usize,
    num_experts: usize,
    num_topk: usize,
) {
    unsafe {
        let topk_len = num_topk.min(num_experts);
        if topk_len == 0 {
            return;
        }

        let mut heap = FixedMinHeap::new(topk_values_ptr, topk_indices_ptr, topk_len);
        for expert_idx in 0..num_experts {
            let value = *input_ptr.add(expert_idx);
            heap.push(value, expert_idx);
        }
        heap.sort_desc();
        let len = heap.len();
        if len == 0 {
            return;
        }

        // The caller only keeps top-k experts, so normalize within the selected subset.
        // This avoids a full softmax over all experts and keeps the hot path compact.
        let max_value = *topk_values_ptr;
        let mut norm_sum = T::default();
        for i in 0..len {
            let value = *topk_values_ptr.add(i);
            let normalized = (value - max_value).exp();
            *topk_values_ptr.add(i) = normalized;
            norm_sum += normalized;
        }
        for i in 0..len {
            let prob = *topk_values_ptr.add(i) / norm_sum;
            *topk_values_ptr.add(i) = prob;
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    #[test]
    fn test_experts_topk_softmax_norm_basic() {
        const NUM_EXPERTS: usize = 16;
        const NUM_TOPK: usize = 4;
        const NUM_TOKEN: usize = 3;
        const INDEX_TOKEN: usize = 1;
        let data = [
            0.5, -1.0, 2.5, 3.0, 7.5, 6.5, -2.0, 10.0, 4.0, 8.0, 1.0, 9.5, -3.5, 5.5, 11.0, -0.25,
        ];
        let mut topk_vals = [0.0f32; NUM_TOPK];
        let mut topk_idx = [0; NUM_TOPK];

        unsafe {
            super::experts_topk_softmax_norm(
                data.as_ptr(),
                topk_vals.as_mut_ptr(),
                topk_idx.as_mut_ptr(),
                NUM_EXPERTS,
                NUM_TOPK,
            );
        }

        let mut expected: Vec<(usize, f32)> = data
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, val)| (idx, val))
            .collect();
        expected.sort_by(|a, b| b.1.total_cmp(&a.1));

        let denom: f32 = expected.iter().take(NUM_TOPK).map(|(_, v)| v.exp()).sum();
        for i in 0..(NUM_TOPK) {
            let (idx, val) = expected[i];
            assert_eq!(topk_idx[i], idx);
            assert_relative_eq!(topk_vals[i], val.exp() / denom, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_experts_topk_softmax_norm_topk_exceeds_num_experts() {
        const NUM_EXPERTS: usize = 3;
        const NUM_TOPK: usize = 5;
        let data = [1.0f32, -0.75, 2.5];
        let mut topk_vals = [0.0f32; NUM_TOPK];
        let mut topk_idx = [0; NUM_TOPK];

        unsafe {
            super::experts_topk_softmax_norm(
                data.as_ptr(),
                topk_vals.as_mut_ptr(),
                topk_idx.as_mut_ptr(),
                NUM_EXPERTS,
                NUM_TOPK,
            );
        }

        let denom: f32 = data.iter().map(|v| v.exp()).sum();

        let mut expected: Vec<(usize, f32)> = data.iter().copied().enumerate().collect();
        expected.sort_by(|a, b| b.1.total_cmp(&a.1));

        for (i, (idx, val)) in expected.iter().enumerate() {
            assert_eq!(topk_idx[i], *idx);
            assert_relative_eq!(topk_vals[i], val.exp() / denom, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_experts_topk_softmax_norm_normalizes_within_topk() {
        // 子集内归一化：top-k 概率之和恒为 1，与「全专家 softmax」的
        // 旧 `norm_topk_prob = false` 路径不再共存。
        const NUM_EXPERTS: usize = 4;
        const NUM_TOPK: usize = 2;
        let data = [0.0f32, 1.0, 2.0, 3.0];
        let mut topk_vals = [0.0f32; NUM_TOPK];
        let mut topk_idx = [0; NUM_TOPK];

        unsafe {
            super::experts_topk_softmax_norm(
                data.as_ptr(),
                topk_vals.as_mut_ptr(),
                topk_idx.as_mut_ptr(),
                NUM_EXPERTS,
                NUM_TOPK,
            );
        }

        let denom: f32 = data[2..4].iter().map(|v| v.exp()).sum();
        let expected = [
            (3usize, data[3].exp() / denom),
            (2usize, data[2].exp() / denom),
        ];

        for (i, (idx, prob)) in expected.iter().enumerate() {
            assert_eq!(topk_idx[i], *idx);
            assert_relative_eq!(topk_vals[i], *prob, epsilon = 1e-6);
        }
    }
}
