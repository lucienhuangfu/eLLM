// === kernel/x86_64/f16_512/moe_down.rs ===
#![allow(non_snake_case)]

use std::arch::x86_64::{_mm512_fmadd_ph, _mm512_loadu_ph, _mm512_set1_ph, _mm512_storeu_ph};
use std::f16;

/// out[i] += factor * acc[i], i in [0, len)
///
/// - out_ptr:  OUT[b, slot, n0..] 的起点
/// - acc_ptr:  acc_tile 中某一行（对应 token r）
/// - factor:   对应 expert 的路由权重
/// - len:      本次 tile 覆盖的列数（n_blk），可为任意 >=1
#[target_feature(enable = "avx512fp16")]
pub unsafe fn moe_down_scale_add(out_ptr: *mut f16, acc_ptr: *const f16, factor: f16, len: usize) {
    let mut i = 0usize;

    let v_factor = _mm512_set1_ph(factor);

    // 向量部分：每次处理 32 个元素
    while i + 32 <= len {
        let v_out = _mm512_loadu_ph(out_ptr.add(i));
        let v_acc = _mm512_loadu_ph(acc_ptr.add(i));
        // out = out + factor * acc
        let v_res = _mm512_fmadd_ph(v_factor, v_acc, v_out);
        _mm512_storeu_ph(out_ptr.add(i), v_res);
        i += 32;
    }

    // 尾部：len 不是 32 的倍数时，处理剩余元素
    while i < len {
        let o = *out_ptr.add(i);
        let a = *acc_ptr.add(i);
        *out_ptr.add(i) = o + factor * a;
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

    /// 标量参考实现：out[i] += factor * acc[i]
    fn reference_scale_add(out: &mut [f16], acc: &[f16], factor: f32) {
        assert_eq!(out.len(), acc.len());
        for i in 0..out.len() {
            let o = out[i] as f32;
            let a = acc[i] as f32;
            let r = o + factor * a;
            out[i] = r as f16;
        }
    }

    #[test]
    fn test_moe_down_scale_add_len_multiple_of_32() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!(
                "Skipping test_moe_down_scale_add_len_multiple_of_32: avx512fp16 not detected"
            );
            return;
        }

        const LEN: usize = 64;

        let factor_f32 = 0.75f32;
        let factor = factor_f32 as f16;

        // 构造 out / acc
        let mut out: Vec<f16> = (0..LEN).map(|i| (0.1f32 * (i as f32)) as f16).collect();
        let acc: Vec<f16> = (0..LEN)
            .map(|i| (0.05f32 * (i as f32) - 0.3f32) as f16)
            .collect();

        // 参考结果
        let mut out_ref = out.clone();
        reference_scale_add(&mut out_ref, &acc, factor_f32);

        // 调用 AVX512 实现
        unsafe {
            moe_down_scale_add(out.as_mut_ptr(), acc.as_ptr(), factor, LEN);
        }

        for i in 0..LEN {
            let g = out[i];
            let e = out_ref[i];
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
    fn test_moe_down_scale_add_len_not_multiple_of_32() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!(
                "Skipping test_moe_down_scale_add_len_not_multiple_of_32: avx512fp16 not detected"
            );
            return;
        }

        // 选一个不是 32 倍数的长度，覆盖尾巴路径
        const LEN: usize = 40;

        let factor_f32 = -1.25f32;
        let factor = factor_f32 as f16;

        let mut out: Vec<f16> = (0..LEN)
            .map(|i| (0.2f32 * (i as f32) + 0.5f32) as f16)
            .collect();
        let acc: Vec<f16> = (0..LEN)
            .map(|i| (-0.1f32 * (i as f32) + 0.7f32) as f16)
            .collect();

        let mut out_ref = out.clone();
        reference_scale_add(&mut out_ref, &acc, factor_f32);

        unsafe {
            moe_down_scale_add(out.as_mut_ptr(), acc.as_ptr(), factor, LEN);
        }

        for i in 0..LEN {
            let g = out[i];
            let e = out_ref[i];
            assert!(
                approx_eq(g, e, 1e-2),
                "mismatch (len=40) at {}: got {:?} (f32={}), exp {:?} (f32={})",
                i,
                g,
                g as f32,
                e,
                e as f32,
            );
        }
    }
}
