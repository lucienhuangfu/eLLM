// === kernel/x86_64/f16_512/moe_merge.rs ===
#![allow(non_snake_case)]

use std::arch::x86_64::{
    _mm512_add_ph, _mm512_loadu_ph, _mm512_storeu_ph,
};
use std::f16;

/// 行内加法：dst[i] += add[i], i in [0, len)
///
/// - dst_ptr: 输出行（OUT[t, :]），原位累加
/// - add_ptr: input[t, s, :] 这行
/// - len:     hidden_size
#[target_feature(enable = "avx512fp16")]
pub unsafe fn moe_merge_add(
    dst_ptr: *mut f16,
    add_ptr: *const f16,
    len: usize,
) {
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
}#[cfg(test)]
mod tests {
    use super::*;
    use std::arch::is_x86_feature_detected;
    use std::mem;

    use crate::kernel::generic::from_f32::FromF32; // 提供 f16::from_f32

    #[inline]
    fn f16_from_f32(x: f32) -> f16 {
        f16::from_f32(x)
    }

    /// 手写 f16 -> f32，只在测试里做误差比较 / 打印用
    #[inline]
    fn f32_from_f16(x: f16) -> f32 {
        let bits: u16 = unsafe { mem::transmute(x) };
        let sign = ((bits & 0x8000) as u32) << 16;
        let exp = (bits & 0x7C00) >> 10;
        let mant = bits & 0x03FF;

        let f_bits: u32 = if exp == 0 {
            if mant == 0 {
                sign
            } else {
                let mut e: i32 = -14;
                let mut m = mant as u32;
                while (m & 0x0400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x03FF;
                let exp_f = (e + 127) as u32;
                sign | (exp_f << 23) | (m << 13)
            }
        } else if exp == 0x1F {
            let exp_f = 0xFFu32;
            sign | (exp_f << 23) | ((mant as u32) << 13)
        } else {
            let exp_f = (exp as i32 - 15 + 127) as u32;
            sign | (exp_f << 23) | ((mant as u32) << 13)
        };

        f32::from_bits(f_bits)
    }

    fn approx_eq(a: f16, b: f16, tol: f32) -> bool {
        let da = f32_from_f16(a);
        let db = f32_from_f16(b);
        (da - db).abs() <= tol
    }

    /// 标量参考：dst[i] += add[i]
    fn reference_merge_add(dst: &mut [f16], add: &[f16]) {
        assert_eq!(dst.len(), add.len());
        for i in 0..dst.len() {
            let d = f32_from_f16(dst[i]);
            let a = f32_from_f16(add[i]);
            dst[i] = f16_from_f32(d + a);
        }
    }

    #[test]
    fn test_moe_merge_add_len_multiple_of_32() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("Skipping test_moe_merge_add_len_multiple_of_32: avx512fp16 not detected");
            return;
        }

        const LEN: usize = 64;

        let mut dst: Vec<f16> = (0..LEN)
            .map(|i| f16_from_f32(0.1f32 * (i as f32)))
            .collect();
        let add: Vec<f16> = (0..LEN)
            .map(|i| f16_from_f32(-0.05f32 * (i as f32) + 0.3f32))
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
                f32_from_f16(g),
                e,
                f32_from_f16(e),
            );
        }
    }

    #[test]
    fn test_moe_merge_add_len_not_multiple_of_32() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("Skipping test_moe_merge_add_len_not_multiple_of_32: avx512fp16 not detected");
            return;
        }

        // 不是 32 的倍数，覆盖尾巴
        const LEN: usize = 45;

        let mut dst: Vec<f16> = (0..LEN)
            .map(|i| f16_from_f32(0.07f32 * (i as f32) - 0.2f32))
            .collect();
        let add: Vec<f16> = (0..LEN)
            .map(|i| f16_from_f32(0.11f32 * (i as f32) + 0.4f32))
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
                f32_from_f16(g),
                e,
                f32_from_f16(e),
            );
        }
    }
}