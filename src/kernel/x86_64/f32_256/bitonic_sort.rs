#![allow(non_snake_case)]
#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

/// 对一个 __m256i (8 x i32) 做 bitonic sort（降序）。
#[target_feature(enable = "avx2")]
unsafe fn bitonic_sort_i32x8_desc(mut a: __m256i) -> __m256i {
    // 预计算 permutation 索引
    let idx1 = _mm256_setr_epi32(1,0,3,2,5,4,7,6);
    let idx2 = _mm256_setr_epi32(2,3,0,1,6,7,4,5);
    let idx4 = _mm256_setr_epi32(4,5,6,7,0,1,2,3);

    // 各阶段 mask（和升序一样）
    let mask_k2_j1 = _mm256_setr_epi32(
        0xFFFFFFFFu32 as i32, 0, 0, 0xFFFFFFFFu32 as i32,
        0xFFFFFFFFu32 as i32, 0, 0, 0xFFFFFFFFu32 as i32);
    let mask_k4_j2 = _mm256_setr_epi32(
        0xFFFFFFFFu32 as i32, 0xFFFFFFFFu32 as i32, 0, 0,
        0, 0, 0xFFFFFFFFu32 as i32, 0xFFFFFFFFu32 as i32);
    let mask_k4_j1 = _mm256_setr_epi32(
        0xFFFFFFFFu32 as i32, 0, 0xFFFFFFFFu32 as i32, 0,
        0, 0xFFFFFFFFu32 as i32, 0, 0xFFFFFFFFu32 as i32);
    let mask_k8_j4 = _mm256_setr_epi32(
        0xFFFFFFFFu32 as i32,0xFFFFFFFFu32 as i32,0xFFFFFFFFu32 as i32,0xFFFFFFFFu32 as i32,
        0,0,0,0);
    let mask_k8_j2 = _mm256_setr_epi32(
        0xFFFFFFFFu32 as i32,0xFFFFFFFFu32 as i32,0,0,
        0xFFFFFFFFu32 as i32,0xFFFFFFFFu32 as i32,0,0);
    let mask_k8_j1 = _mm256_setr_epi32(
        0xFFFFFFFFu32 as i32,0,0xFFFFFFFFu32 as i32,0,
        0xFFFFFFFFu32 as i32,0,0xFFFFFFFFu32 as i32,0);

    // 降序版的 compare-exchange：
    // 升序时选 lo，降序时选 hi
    #[inline(always)]
    unsafe fn cmp_exchange_desc(a: __m256i, idx: __m256i, mask: __m256i) -> __m256i {
        let t = _mm256_permutevar8x32_epi32(a, idx);
        let lo = _mm256_min_epi32(a, t);
        let hi = _mm256_max_epi32(a, t);
        // 降序时：mask=1 的 lane 取 hi，否则取 lo
        _mm256_blendv_epi8(lo, hi, mask)
    }

    // 执行 bitonic 网络的阶段
    a = cmp_exchange_desc(a, idx1, mask_k2_j1);
    a = cmp_exchange_desc(a, idx2, mask_k4_j2);
    a = cmp_exchange_desc(a, idx1, mask_k4_j1);
    a = cmp_exchange_desc(a, idx4, mask_k8_j4);
    a = cmp_exchange_desc(a, idx2, mask_k8_j2);
    a = cmp_exchange_desc(a, idx1, mask_k8_j1);

    a
}

/// [i32;8] <-> __m256i
#[target_feature(enable = "avx2")]
unsafe fn load_i32x8(arr: [i32;8]) -> __m256i {
    _mm256_setr_epi32(arr[0],arr[1],arr[2],arr[3],arr[4],arr[5],arr[6],arr[7])
}
#[target_feature(enable = "avx2")]
unsafe fn store_i32x8(v: __m256i) -> [i32;8] {
    let mut out = [0i32;8];
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
                let sorted_v = bitonic_sort_i32x8_desc(v);
                let out = store_i32x8(sorted_v);
                let mut expected = a;
                expected.sort_by(|a,b| b.cmp(a)); // 降序
                assert_eq!(out, expected);
            }
        }
    }
}
