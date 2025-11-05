#![allow(non_snake_case)]
#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

/// 预计算 AVX2 bitonic 排序网络的控制参数。
/// 三个排列向量覆盖跨度 2、4、8 的比较阶段，六个混合掩码用于降序排序时在高/低半区间选择结果。
#[inline(always)]
unsafe fn bitonic_perm_masks() -> ([__m256i; 3], [__m256i; 6]) {
    (
        [
            _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6),
            _mm256_setr_epi32(2, 3, 0, 1, 6, 7, 4, 5),
            _mm256_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3),
        ],
        [
            _mm256_setr_epi32(
                0xFFFFFFFFu32 as i32,
                0,
                0,
                0xFFFFFFFFu32 as i32,
                0xFFFFFFFFu32 as i32,
                0,
                0,
                0xFFFFFFFFu32 as i32,
            ),
            _mm256_setr_epi32(
                0xFFFFFFFFu32 as i32,
                0xFFFFFFFFu32 as i32,
                0,
                0,
                0,
                0,
                0xFFFFFFFFu32 as i32,
                0xFFFFFFFFu32 as i32,
            ),
            _mm256_setr_epi32(
                0xFFFFFFFFu32 as i32,
                0,
                0xFFFFFFFFu32 as i32,
                0,
                0,
                0xFFFFFFFFu32 as i32,
                0,
                0xFFFFFFFFu32 as i32,
            ),
            _mm256_setr_epi32(
                0xFFFFFFFFu32 as i32,
                0xFFFFFFFFu32 as i32,
                0xFFFFFFFFu32 as i32,
                0xFFFFFFFFu32 as i32,
                0,
                0,
                0,
                0,
            ),
            _mm256_setr_epi32(
                0xFFFFFFFFu32 as i32,
                0xFFFFFFFFu32 as i32,
                0,
                0,
                0xFFFFFFFFu32 as i32,
                0xFFFFFFFFu32 as i32,
                0,
                0,
            ),
            _mm256_setr_epi32(
                0xFFFFFFFFu32 as i32,
                0,
                0xFFFFFFFFu32 as i32,
                0,
                0xFFFFFFFFu32 as i32,
                0,
                0xFFFFFFFFu32 as i32,
                0,
            ),
        ],
    )
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn bitonic_sort_f32x8_desc(
    values: __m256,
    indices: __m256i,
) -> (__m256, __m256i) {
    let (permutes, masks) = bitonic_perm_masks();
    let [idx1, idx2, idx4] = permutes;
    let [mask_k2_j1, mask_k4_j2, mask_k4_j1, mask_k8_j4, mask_k8_j2, mask_k8_j1] = masks;
    #[inline(always)]
    unsafe fn cmp_exchange_desc(
        pair: (__m256, __m256i),
        idx: __m256i,
        mask: __m256i,
    ) -> (__m256, __m256i) {
        let (vals, ids) = pair;
        let perm_vals = _mm256_permutevar8x32_ps(vals, idx);
        let perm_ids = _mm256_permutevar8x32_epi32(ids, idx);
        let gt_mask_ps = _mm256_cmp_ps(vals, perm_vals, _CMP_GT_OQ);
        let gt_mask = _mm256_castps_si256(gt_mask_ps);
        let hi_vals = _mm256_blendv_ps(perm_vals, vals, gt_mask_ps);
        let lo_vals = _mm256_blendv_ps(vals, perm_vals, gt_mask_ps);
        let hi_ids = _mm256_blendv_epi8(perm_ids, ids, gt_mask);
        let lo_ids = _mm256_blendv_epi8(ids, perm_ids, gt_mask);
        let stage_mask_ps = _mm256_castsi256_ps(mask);
        let sorted_vals = _mm256_blendv_ps(lo_vals, hi_vals, stage_mask_ps);
        let sorted_ids = _mm256_blendv_epi8(lo_ids, hi_ids, mask);
        (sorted_vals, sorted_ids)
    }
    let mut state = (values, indices);
    state = cmp_exchange_desc(state, idx1, mask_k2_j1);
    state = cmp_exchange_desc(state, idx2, mask_k4_j2);
    state = cmp_exchange_desc(state, idx1, mask_k4_j1);
    state = cmp_exchange_desc(state, idx4, mask_k8_j4);
    state = cmp_exchange_desc(state, idx2, mask_k8_j2);
    state = cmp_exchange_desc(state, idx1, mask_k8_j1);
    state
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::arch::is_x86_feature_detected;
    use std::arch::x86_64::*;
    #[test]
    fn bitonic_sort_desc_orders_values_and_indices() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2-dependent test");
            return;
        }
        unsafe {
            let values = _mm256_setr_ps(1.0, 3.0, 2.0, 7.0, 5.0, 4.0, 6.0, 0.0);
            let indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            let (sorted_vals, sorted_ids) = bitonic_sort_f32x8_desc(values, indices);
            let mut vals_buf = [0.0f32; 8];
            let mut idx_buf = [0i32; 8];
            _mm256_storeu_ps(vals_buf.as_mut_ptr(), sorted_vals);
            _mm256_storeu_si256(idx_buf.as_mut_ptr() as *mut __m256i, sorted_ids);
            assert_eq!(vals_buf, [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);
            assert_eq!(idx_buf, [3, 6, 4, 5, 1, 2, 0, 7]);
        }
    }
}
