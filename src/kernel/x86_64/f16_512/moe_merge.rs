// === kernel/x86_64/f16_512/moe_merge.rs ===
#![allow(non_snake_case)]

use std::arch::x86_64::{_mm512_add_ph, _mm512_loadu_ph, _mm512_storeu_ph};
use std::f16;

/// 行内加法：dst[i] += add[i], i in [0, len)
///
/// - dst_ptr: 输出行（OUT[t, :]），原位累加
/// - add_ptr: input[t, s, :] 这行
/// - len:     hidden_size
#[target_feature(enable = "avx512fp16")]
pub unsafe fn moe_merge_add(dst_ptr: *mut f16, add_ptr: *const f16, len: usize) {
    let mut i = 0usize;

    // 向量部分
    while i + 32 <= len {
        let v_dst = _mm512_loadu_ph(dst_ptr.add(i));
        let v_add = _mm512_loadu_ph(add_ptr.add(i));
        let v_res = _mm512_add_ph(v_dst, v_add);
        _mm512_storeu_ph(dst_ptr.add(i), v_res);
        i += 32;
    }

    // 尾部
    while i < len {
        let d = *dst_ptr.add(i);
        let a = *add_ptr.add(i);
        *dst_ptr.add(i) = d + a;
        i += 1;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::arch::is_x86_feature_detected;
    use std::mem;

    fn approx_eq(a: f16, b: f16, tol: f32) -> bool {
        let da = a as f32;
        let db = b as f32;
        (da - db).abs() <= tol
    }

    /// 标量参考：dst[i] += add[i]
    fn reference_merge_add(dst: &mut [f16], add: &[f16]) {
        assert_eq!(dst.len(), add.len());
        for i in 0..dst.len() {
            let d = dst[i] as f32;
            let a = add[i] as f32;
            dst[i] = (d + a) as f16;
        }
    }

    #[test]
    fn test_moe_merge_add_len_multiple_of_32() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("Skipping test_moe_merge_add_len_multiple_of_32: avx512fp16 not detected");
            return;
        }

        const LEN: usize = 64;

        let mut dst: Vec<f16> = (0..LEN).map(|i| (0.1f32 * (i as f32)) as f16).collect();
        let add: Vec<f16> = (0..LEN)
            .map(|i| (-0.05f32 * (i as f32) + 0.3f32) as f16)
            .collect();

        let mut dst_ref = dst.clone();
        reference_merge_add(&mut dst_ref, &add);

        unsafe {
            moe_merge_add(dst.as_mut_ptr(), add.as_ptr(), LEN);
        }

        for i in 0..LEN {
            let g = dst[i];
            let e = dst_ref[i];
            assert!(
                approx_eq(g, e, 1e-2),
                "mismatch (len=64) at {}: got {:?} (f32={}), exp {:?} (f32={})",
                i,
                g,
                g as f32,
                e,
                e as f32,
            );
        }
    }

    #[test]
    fn test_moe_merge_add_len_not_multiple_of_32() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!(
                "Skipping test_moe_merge_add_len_not_multiple_of_32: avx512fp16 not detected"
            );
            return;
        }

        // 不是 32 的倍数，覆盖尾巴
        const LEN: usize = 45;

        let mut dst: Vec<f16> = (0..LEN)
            .map(|i| (0.07f32 * (i as f32) - 0.2f32) as f16)
            .collect();
        let add: Vec<f16> = (0..LEN)
            .map(|i| (0.11f32 * (i as f32) + 0.4f32) as f16)
            .collect();

        let mut dst_ref = dst.clone();
        reference_merge_add(&mut dst_ref, &add);

        unsafe {
            moe_merge_add(dst.as_mut_ptr(), add.as_ptr(), LEN);
        }

        for i in 0..LEN {
            let g = dst[i];
            let e = dst_ref[i];
            assert!(
                approx_eq(g, e, 1e-2),
                "mismatch (len=45) at {}: got {:?} (f32={}), exp {:?} (f32={})",
                i,
                g,
                g as f32,
                e,
                e as f32,
            );
        }
    }
}
