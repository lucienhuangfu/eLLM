// === kernel/x86_64/f16_512/matmul_rms_complex.rs ===
#![allow(non_snake_case)]

use std::arch::x86_64::{
    _mm512_fmadd_ph,
    _mm512_loadu_ph,   // ← 改这里
    _mm512_set1_ph,
    _mm512_storeu_ph,  // ← 改这里
};
use std::f16;

use crate::kernel::x86_64::f16_512::complex_mul::complex_mul;
use crate::kernel::x86_64::f16_512::rms_norm::rms_norm;

/// 3×32 in-place 累加：每个 kc 面板把 A×B_panel 直接加到 C 上并写回
///
/// 形状约定:
/// - MR = 3, NR = 32
/// - A_tile: 3×kc，行距 = lda = K
/// - B_panel: kc×32（行主，每行 32 连续）
/// - C_tile: 3×32，行距 = ldc = N（整行跨度）
/// - kc     = 当前 K block 大小
///
/// 注意：不清零，不做 first；要求上层在 K 循环外保证 C 的初始状态。
#[target_feature(enable = "avx512fp16")]
pub unsafe fn matmul_update_inplace_3x32_accum(
    a: *const f16,       // 3×kc
    b_panel: *const f16, // kc×32
    c: *mut f16,         // 3×32
    lda: usize,          // A 行距 = K
    ldc: usize,          // C 行距 = 当前矩阵的 N
    kc: usize,           // 当前 K block 长度
) {
    let mr = 3usize;
    let nr = 32usize;
    debug_assert_eq!(mr, 3);
    debug_assert_eq!(nr, 32);

    let bstride = nr; // B_panel 每行 32 连续元素

    // A 三行基址
    let a0 = a;
    let a1 = a.add(lda);
    let a2 = a.add(2 * lda);

    // 从 C 载入旧值（3 行 × 32 列）
    let mut c0 = _mm512_loadu_ph(c.add(0 * ldc));
    let mut c1 = _mm512_loadu_ph(c.add(1 * ldc));
    let mut c2 = _mm512_loadu_ph(c.add(2 * ldc));

    // 主循环：kc 方向做广播 × 向量 FMA
    for k in 0..kc {
        let b = _mm512_loadu_ph(b_panel.add(k * bstride));

        let a0k = _mm512_set1_ph(*a0.add(k));
        let a1k = _mm512_set1_ph(*a1.add(k));
        let a2k = _mm512_set1_ph(*a2.add(k));

        c0 = _mm512_fmadd_ph(a0k, b, c0);
        c1 = _mm512_fmadd_ph(a1k, b, c1);
        c2 = _mm512_fmadd_ph(a2k, b, c2);
    }

    _mm512_storeu_ph(c.add(0 * ldc), c0);
    _mm512_storeu_ph(c.add(1 * ldc), c1);
    _mm512_storeu_ph(c.add(2 * ldc), c2);
}

/// 3×128 收尾：在 C 上 **原地** 做 RMSNorm(weight=1) + RoPE
///
/// - c:        指向 3×128 tile 左上角
/// - rope_ptr: 长度 128 的 [cos0, sin0, cos1, sin1, ...]
/// - ldc:      C 的行距（N）
/// - eps:      数值稳定项
#[target_feature(enable = "avx512fp16")]
pub unsafe fn matmul_finalize_rmsnorm_rope_inplace_3x128(
    c: *mut f16,          // 3×128，行距=ldc
    rope_ptr: *const f16, // 128 个交错相位
    ldc: usize,
    eps: f16,
) {
    let mr = 3usize;
    let nr = 128usize;
    debug_assert_eq!(mr, 3);
    debug_assert_eq!(nr, 128);

    // 对 3 行分别做 in-place RMSNorm（长度 128）
    rms_norm(c.add(0 * ldc), c.add(0 * ldc), 128, eps);
    rms_norm(c.add(1 * ldc), c.add(1 * ldc), 128, eps);
    rms_norm(c.add(2 * ldc), c.add(2 * ldc), 128, eps);

    // 然后对 3 行分别做 in-place RoPE 复数乘（长度 128）
    complex_mul(c.add(0 * ldc), rope_ptr, c.add(0 * ldc), 128);
    complex_mul(c.add(1 * ldc), rope_ptr, c.add(1 * ldc), 128);
    complex_mul(c.add(2 * ldc), rope_ptr, c.add(2 * ldc), 128);
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::arch::is_x86_feature_detected;
    use std::mem;
    use std::ptr;
    use std::slice;

    // 你们自己的对齐分配器
    use crate::mem_mgr::allocator::allocate_init;

    fn approx_eq(a: f16, b: f16, tol: f32) -> bool {
        let da = a as f32;
        let db = b as f32;
        (da - db).abs() <= tol
    }

    /// 标量参考实现：在 f16 域里做 C = C + A×B（3×kc * kc×32）
    fn reference_matmul_accum_3x32(
        a: &[f16],     // 3×kc, lda = kc
        b: &[f16],     // kc×32, 行主
        c: &mut [f16], // 3×32, ldc = 32
        kc: usize,
    ) {
        let lda = kc;
        let ldc = 32;

        for m in 0..3 {
            for n in 0..32 {
                let mut acc = c[m * ldc + n];
                for k in 0..kc {
                    let av = a[m * lda + k];
                    let bv = b[k * 32 + n];
                    acc = acc + av * bv;
                }
                c[m * ldc + n] = acc;
            }
        }
    }

    #[test]
    fn test_matmul_update_inplace_3x32_accum_basic() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!(
                "Skipping test_matmul_update_inplace_3x32_accum_basic: avx512fp16 not detected"
            );
            return;
        }

        const KC: usize = 7;
        const LDA: usize = KC;
        const LDC: usize = 32;

        // -------- 构造 A/B/C：一份对齐指针给 AVX，用 Vec 做参考 ----------

        // A：对齐版指针
        let a_ptr = allocate_init::<f16>(3 * LDA, 0.0 as f16);
        // A：标量参考版
        let mut a_ref = vec![0.0 as f16; 3 * LDA];

        for m in 0..3 {
            for k in 0..KC {
                let v = (m as f32) * 0.1 + (k as f32) * 0.01;
                let val = v as f16;
                a_ref[m * LDA + k] = val;
                unsafe {
                    ptr::write(a_ptr.add(m * LDA + k), val);
                }
            }
        }

        // B
        let b_ptr = allocate_init::<f16>(KC * 32, 0.0 as f16);
        let mut b_ref = vec![0.0 as f16; KC * 32];
        for k in 0..KC {
            for n in 0..32 {
                let v = (k as f32) * 0.02 + (n as f32) * 0.001;
                let val = v as f16;
                b_ref[k * 32 + n] = val;
                unsafe {
                    ptr::write(b_ptr.add(k * 32 + n), val);
                }
            }
        }

        // C
        let c_init = 0.5 as f16;
        let c_ptr = allocate_init::<f16>(3 * LDC, c_init);
        let mut c_ref = vec![c_init; 3 * LDC];

        // 参考计算
        reference_matmul_accum_3x32(&a_ref, &b_ref, &mut c_ref, KC);

        // 调用 AVX 内核
        unsafe {
            matmul_update_inplace_3x32_accum(a_ptr, b_ptr, c_ptr, LDA, LDC, KC);
        }

        // 从对齐指针构造 slice 读回结果
        let c_slice = unsafe { slice::from_raw_parts(c_ptr, 3 * LDC) };

        for i in 0..c_slice.len() {
            assert!(
                approx_eq(c_slice[i], c_ref[i], 1e-2),
                "mismatch at index {}: got {:?}, expected {:?} (f32: {} vs {})",
                i,
                c_slice[i],
                c_ref[i],
                c_slice[i] as f32,
                c_ref[i] as f32,
            );
        }
    }

    /// 把 f16 当成 u16 比特来比较（测试 finalize 用）
    #[inline]
    fn bits_from_f16(x: f16) -> u16 {
        unsafe { mem::transmute::<f16, u16>(x) }
    }

    #[test]
    fn test_matmul_finalize_rmsnorm_rope_inplace_3x128() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!(
                "Skipping test_matmul_finalize_rmsnorm_rope_inplace_3x128: avx512fp16 not detected"
            );
            return;
        }

        const LDC: usize = 128;
        let eps = 1e-5 as f16;

        // -------- C / rope 全部用 allocate_init，保证传给 AVX 的都是对齐指针 --------

        // C：对齐指针，用于被测路径
        let c_ptr = allocate_init::<f16>(3 * LDC, 0.0 as f16);
        // C_ref：对齐指针，用于“手动调用 rms_norm+complex_mul”的参考路径
        let c_ref_ptr = allocate_init::<f16>(3 * LDC, 0.0 as f16);

        for row in 0..3 {
            for col in 0..LDC {
                let v = 0.01f32 * (row as f32) + 0.001f32 * (col as f32);
                let val = v as f16;
                let idx = row * LDC + col;
                unsafe {
                    ptr::write(c_ptr.add(idx), val);
                    ptr::write(c_ref_ptr.add(idx), val);
                }
            }
        }

        // rope：对齐指针
        let rope_ptr = allocate_init::<f16>(LDC, 0.0 as f16);
        let num_pairs = LDC / 2;
        for j in 0..num_pairs {
            let theta = 0.01f32 * (j as f32);
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            unsafe {
                ptr::write(rope_ptr.add(2 * j), cos_t as f16);
                ptr::write(rope_ptr.add(2 * j + 1), sin_t as f16);
            }
        }

        // 手动参考路径：逐行调用 rms_norm + complex_mul
        unsafe {
            // row 0
            rms_norm(c_ref_ptr.add(0 * LDC), c_ref_ptr.add(0 * LDC), 128, eps);
            complex_mul(
                c_ref_ptr.add(0 * LDC),
                rope_ptr,
                c_ref_ptr.add(0 * LDC),
                128,
            );

            // row 1
            rms_norm(c_ref_ptr.add(1 * LDC), c_ref_ptr.add(1 * LDC), 128, eps);
            complex_mul(
                c_ref_ptr.add(1 * LDC),
                rope_ptr,
                c_ref_ptr.add(1 * LDC),
                128,
            );

            // row 2
            rms_norm(c_ref_ptr.add(2 * LDC), c_ref_ptr.add(2 * LDC), 128, eps);
            complex_mul(
                c_ref_ptr.add(2 * LDC),
                rope_ptr,
                c_ref_ptr.add(2 * LDC),
                128,
            );
        }

        // 被测路径：调用你的 finalize 包装函数
        unsafe {
            matmul_finalize_rmsnorm_rope_inplace_3x128(c_ptr, rope_ptr, LDC, eps);
        }

        // 读回结果，对齐指针 → slice
        let c_slice = unsafe { slice::from_raw_parts(c_ptr, 3 * LDC) };
        let c_ref_slice = unsafe { slice::from_raw_parts(c_ref_ptr, 3 * LDC) };

        // 逐元素按 bit 检查完全一致
        for i in 0..c_slice.len() {
            let got = bits_from_f16(c_slice[i]);
            let exp = bits_from_f16(c_ref_slice[i]);
            assert!(
                got == exp,
                "mismatch at index {}: got_bits=0x{:04X}, exp_bits=0x{:04X}",
                i,
                got,
                exp,
            );
        }
    }
}
