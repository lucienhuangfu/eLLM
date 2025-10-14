// === kernel/x86_64/f16_512/matmul_rms_complex.rs ===
#![allow(non_snake_case)]

use std::arch::x86_64::{
    _mm512_fmadd_ph, _mm512_load_ph, _mm512_set1_ph, _mm512_store_ph,
};
use std::f16;

use crate::init::matmul_params::matmulParams;

// 复用已有的 AVX-512 FP16 实现
use crate::kernel::x86_64::f16_512::rms_norm::rms_norm;
use crate::kernel::x86_64::f16_512::complex_mul::complex_mul;

/// 3×128 in-place 累加：每个 kc 面板把 A×B_panel 直接加到 C 上并写回
///
/// 形状约定:
/// - MR = 3, NR = 128
/// - A_tile: 3×kc，行距 = lda = K
/// - B_panel: kc×128（行主，每行 128 连续）
/// - C_tile: 3×128，行距 = ldc = N
/// - kc     = param.column_step_macro
///
/// 注意：不清零，不做 first；要求上层在 run 前保证 C 的初始状态。

#[target_feature(enable = "avx512fp16")]
pub unsafe fn matmul_update_inplace_3x128_accum(
    a: *const f16,        // 3×kc
    b_panel: *const f16,  // kc×128
    c: *mut f16,          // 3×128
    param: &matmulParams,
) {
    debug_assert_eq!(param.a_row_step_micro, 3);
    debug_assert_eq!(param.b_row_step_micro, 128);

    let lda     = param.a_row_step_macro;
    let kc      = param.column_step_macro;
    let ldc     = param.b_row_step_macro;
    let bstride = 128usize;

    // A 三行基址
    let a0 = a;
    let a1 = a.add(lda);
    let a2 = a.add(2 * lda);

    // 从 C 载入旧值
    let mut c0_0 = _mm512_load_ph(c.add(0*ldc +  0));
    let mut c0_1 = _mm512_load_ph(c.add(0*ldc + 32));
    let mut c0_2 = _mm512_load_ph(c.add(0*ldc + 64));
    let mut c0_3 = _mm512_load_ph(c.add(0*ldc + 96));

    let mut c1_0 = _mm512_load_ph(c.add(1*ldc +  0));
    let mut c1_1 = _mm512_load_ph(c.add(1*ldc + 32));
    let mut c1_2 = _mm512_load_ph(c.add(1*ldc + 64));
    let mut c1_3 = _mm512_load_ph(c.add(1*ldc + 96));

    let mut c2_0 = _mm512_load_ph(c.add(2*ldc +  0));
    let mut c2_1 = _mm512_load_ph(c.add(2*ldc + 32));
    let mut c2_2 = _mm512_load_ph(c.add(2*ldc + 64));
    let mut c2_3 = _mm512_load_ph(c.add(2*ldc + 96));

    // 主循环
    for k in 0..kc {
        let b0 = _mm512_load_ph(b_panel.add(k*bstride +  0));
        let b1 = _mm512_load_ph(b_panel.add(k*bstride + 32));
        let b2 = _mm512_load_ph(b_panel.add(k*bstride + 64));
        let b3 = _mm512_load_ph(b_panel.add(k*bstride + 96));

        let a0k = _mm512_set1_ph(*a0.add(k));
        let a1k = _mm512_set1_ph(*a1.add(k));
        let a2k = _mm512_set1_ph(*a2.add(k));

        c0_0 = _mm512_fmadd_ph(a0k, b0, c0_0);
        c0_1 = _mm512_fmadd_ph(a0k, b1, c0_1);
        c0_2 = _mm512_fmadd_ph(a0k, b2, c0_2);
        c0_3 = _mm512_fmadd_ph(a0k, b3, c0_3);

        c1_0 = _mm512_fmadd_ph(a1k, b0, c1_0);
        c1_1 = _mm512_fmadd_ph(a1k, b1, c1_1);
        c1_2 = _mm512_fmadd_ph(a1k, b2, c1_2);
        c1_3 = _mm512_fmadd_ph(a1k, b3, c1_3);

        c2_0 = _mm512_fmadd_ph(a2k, b0, c2_0);
        c2_1 = _mm512_fmadd_ph(a2k, b1, c2_1);
        c2_2 = _mm512_fmadd_ph(a2k, b2, c2_2);
        c2_3 = _mm512_fmadd_ph(a2k, b3, c2_3);
    }

    // 写回
    _mm512_store_ph(c.add(0*ldc +  0), c0_0);
    _mm512_store_ph(c.add(0*ldc + 32), c0_1);
    _mm512_store_ph(c.add(0*ldc + 64), c0_2);
    _mm512_store_ph(c.add(0*ldc + 96), c0_3);

    _mm512_store_ph(c.add(1*ldc +  0), c1_0);
    _mm512_store_ph(c.add(1*ldc + 32), c1_1);
    _mm512_store_ph(c.add(1*ldc + 64), c1_2);
    _mm512_store_ph(c.add(1*ldc + 96), c1_3);

    _mm512_store_ph(c.add(2*ldc +  0), c2_0);
    _mm512_store_ph(c.add(2*ldc + 32), c2_1);
    _mm512_store_ph(c.add(2*ldc + 64), c2_2);
    _mm512_store_ph(c.add(2*ldc + 96), c2_3);
}

/// 收尾：在 C 上 **原地** 做 RMSNorm(weight=1) + RoPE
///
/// - rope_ptr: 长度 128 的 [cos0, sin0, cos1, sin1, ...]
/// - eps: 数值稳定项

#[target_feature(enable = "avx512fp16")]
pub unsafe fn matmul_finalize_rmsnorm_rope_inplace_3x128(
    c: *mut f16,          // 3×128，行距=ldc
    rope_ptr: *const f16, // 128 个交错相位
    eps: f16,
    p: &matmulParams,
) {
    let ldc = p.b_row_step_macro;
    debug_assert_eq!(p.a_row_step_micro, 3);
    debug_assert_eq!(p.b_row_step_micro, 128);

    // 权重恒 1
    const F16_ONE_BITS: u16 = 0x3C00;
    static ONES128: [f16; 128] = [f16::from_bits(F16_ONE_BITS); 128];

    rms_norm(c.add(0*ldc), c.add(0*ldc), 128, ONES128.as_ptr(), eps);
    rms_norm(c.add(1*ldc), c.add(1*ldc), 128, ONES128.as_ptr(), eps);
    rms_norm(c.add(2*ldc), c.add(2*ldc), 128, ONES128.as_ptr(), eps);

    complex_mul(c.add(0*ldc), rope_ptr, c.add(0*ldc), 128);
    complex_mul(c.add(1*ldc), rope_ptr, c.add(1*ldc), 128);
    complex_mul(c.add(2*ldc), rope_ptr, c.add(2*ldc), 128);
}