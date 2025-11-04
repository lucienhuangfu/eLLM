#![allow(non_snake_case)]
#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

/// 对一个 __m256i (8 x i32) 做 bitonic sort（降序），返回排序后的值和对应索引。
#[target_feature(enable = "avx2")]
unsafe fn bitonic_sort_i32x8_desc(a: __m256i) -> (__m256i, __m256i) {
    // 预计算 permutation 索引
    let idx1 = _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6);
    let idx2 = _mm256_setr_epi32(2, 3, 0, 1, 6, 7, 4, 5);
    let idx4 = _mm256_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3);

    // 各阶段 mask（和升序一样）
    let mask_k2_j1 = _mm256_setr_epi32(
        0xFFFFFFFFu32 as i32,
        0,
        0,
        0xFFFFFFFFu32 as i32,
        0xFFFFFFFFu32 as i32,
        0,
        0,
        0xFFFFFFFFu32 as i32,
    );
    let mask_k4_j2 = _mm256_setr_epi32(
        0xFFFFFFFFu32 as i32,
        0xFFFFFFFFu32 as i32,
        0,
        0,
        0,
        0,
        0xFFFFFFFFu32 as i32,
        0xFFFFFFFFu32 as i32,
    );
    let mask_k4_j1 = _mm256_setr_epi32(
        0xFFFFFFFFu32 as i32,
        0,
        0xFFFFFFFFu32 as i32,
        0,
        0,
        0xFFFFFFFFu32 as i32,
        0,
        0xFFFFFFFFu32 as i32,
    );
    let mask_k8_j4 = _mm256_setr_epi32(
        0xFFFFFFFFu32 as i32,
        0xFFFFFFFFu32 as i32,
        0xFFFFFFFFu32 as i32,
        0xFFFFFFFFu32 as i32,
        0,
        0,
        0,
        0,
    );
    let mask_k8_j2 = _mm256_setr_epi32(
        0xFFFFFFFFu32 as i32,
        0xFFFFFFFFu32 as i32,
        0,
        0,
        0xFFFFFFFFu32 as i32,
        0xFFFFFFFFu32 as i32,
        0,
        0,
    );
    let mask_k8_j1 = _mm256_setr_epi32(
        0xFFFFFFFFu32 as i32,
        0,
        0xFFFFFFFFu32 as i32,
        0,
        0xFFFFFFFFu32 as i32,
        0,
        0xFFFFFFFFu32 as i32,
        0,
    );

    // 降序版的 compare-exchange：
    // 升序时选 lo，降序时选 hi
    #[inline(always)]
    unsafe fn cmp_exchange_desc(
        pair: (__m256i, __m256i),
        idx: __m256i,
        mask: __m256i,
    ) -> (__m256i, __m256i) {
        let (vals, indices) = pair;
        let perm_vals = _mm256_permutevar8x32_epi32(vals, idx);
        let perm_indices = _mm256_permutevar8x32_epi32(indices, idx);
        let gt_mask = _mm256_cmpgt_epi32(vals, perm_vals);
        let hi_vals = _mm256_blendv_epi8(perm_vals, vals, gt_mask);
        let lo_vals = _mm256_blendv_epi8(vals, perm_vals, gt_mask);
        let hi_indices = _mm256_blendv_epi8(perm_indices, indices, gt_mask);
        let lo_indices = _mm256_blendv_epi8(indices, perm_indices, gt_mask);
        let sorted_vals = _mm256_blendv_epi8(lo_vals, hi_vals, mask);
        let sorted_indices = _mm256_blendv_epi8(lo_indices, hi_indices, mask);
        (sorted_vals, sorted_indices)
    }

    // 初始化状态：值和对应的索引
    let mut state = (a, _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    // 执行 bitonic 网络的阶段
    state = cmp_exchange_desc(state, idx1, mask_k2_j1);
    state = cmp_exchange_desc(state, idx2, mask_k4_j2);
    state = cmp_exchange_desc(state, idx1, mask_k4_j1);
    state = cmp_exchange_desc(state, idx4, mask_k8_j4);
    state = cmp_exchange_desc(state, idx2, mask_k8_j2);
    state = cmp_exchange_desc(state, idx1, mask_k8_j1);

    state
}

/// [i32;8] <-> __m256i
#[target_feature(enable = "avx2")]
unsafe fn load_i32x8(arr: [i32; 8]) -> __m256i {
    _mm256_setr_epi32(
        arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7],
    )
}
#[target_feature(enable = "avx2")]
unsafe fn store_i32x8(v: __m256i) -> [i32; 8] {
    let mut out = [0i32; 8];
    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, v);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_bitonic_sort_i32x8_desc() {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                let a = [23, -1, 5, 100, 0, 7, -20, 50];
                let v = load_i32x8(a);
                let (sorted_v, sorted_idx) = bitonic_sort_i32x8_desc(v);
                let out = store_i32x8(sorted_v);
                let indices = store_i32x8(sorted_idx);
                let mut expected = a;
                expected.sort_by(|a, b| b.cmp(a)); // 降序
                assert_eq!(out, expected);
                let mut seen = [false; 8];
                for (pos, &idx) in indices.iter().enumerate() {
                    assert!((0..=7).contains(&idx));
                    let idx_usize = idx as usize;
                    assert!(!seen[idx_usize]);
                    seen[idx_usize] = true;
                    assert_eq!(out[pos], a[idx_usize]);
                }
            }
        }
    }
}
