// === kernel/x86_64/f16_512/matmul_rms_complex.rs ===
#![allow(non_snake_case)]

use std::arch::x86_64::{
    _mm512_fmadd_ph, _mm512_load_ph, _mm512_set1_ph, _mm512_store_ph,
};
use std::f16;

use crate::init::matmul_params::MatMulParams;

// 复用已有的 AVX-512 FP16 实现
use crate::kernel::x86_64::f16_512::rms_norm::rms_norm;
use crate::kernel::x86_64::f16_512::complex_mul::complex_mul;

/// 3×32 in-place 累加：每个 kc 面板把 A×B_panel 直接加到 C 上并写回
///
/// 形状约定:
/// - MR = 3, NR = 32
/// - A_tile: 3×kc，行距 = lda = K
/// - B_panel: kc×32（行主，每行 32 连续）
/// - C_tile: 3×32，行距 = ldc = N（整行跨度，和 K/Q/V 自己的 N 有关）
/// - kc     = param.column_step_macro
///
/// 注意：不清零，不做 first；要求上层在 run 前保证 C 的初始状态。
#[target_feature(enable = "avx512fp16")]
pub unsafe fn matmul_update_inplace_3x128_accum( // 名字暂时不改，内部已经是 3×32
    a: *const f16,        // 3×kc
    b_panel: *const f16,  // kc×32
    c: *mut f16,          // 3×32
    param: &MatMulParams,
) {
    // 微核尺寸：3 行 × 32 列
    debug_assert_eq!(param.a_row_step_micro, 3);
    debug_assert_eq!(param.b_row_step_micro, 32);

    let lda     = param.a_row_step_macro;   // A 行距 = K
    let kc      = param.column_step_macro;  // 当前 kc
    let ldc     = param.b_row_step_macro;   // C 行距 = 该矩阵的 N
    let bstride = 32usize;                  // B_panel 每行 32 连续元素

    // A 三行基址
    let a0 = a;
    let a1 = a.add(lda);
    let a2 = a.add(2 * lda);

    // 从 C 载入旧值（3 行 × 32 列，各占 1 个 ZMM）
    let mut c0 = _mm512_load_ph(c.add(0 * ldc));
    let mut c1 = _mm512_load_ph(c.add(1 * ldc));
    let mut c2 = _mm512_load_ph(c.add(2 * ldc));

    // 主循环：对 kc 方向做标量广播 × 向量 FMA
    for k in 0..kc {
        // 这一行 B 的 32 宽向量
        let b = _mm512_load_ph(b_panel.add(k * bstride));

        // A 三行各取一个标量并广播
        let a0k = _mm512_set1_ph(*a0.add(k));
        let a1k = _mm512_set1_ph(*a1.add(k));
        let a2k = _mm512_set1_ph(*a2.add(k));

        // C += A_row * B_row
        c0 = _mm512_fmadd_ph(a0k, b, c0);
        c1 = _mm512_fmadd_ph(a1k, b, c1);
        c2 = _mm512_fmadd_ph(a2k, b, c2);
    }

    // 写回 C 的 3×32 tile
    _mm512_store_ph(c.add(0 * ldc), c0);
    _mm512_store_ph(c.add(1 * ldc), c1);
    _mm512_store_ph(c.add(2 * ldc), c2);
}

/// 收尾：在 C 上 **原地** 做 RMSNorm(weight=1) + RoPE（3×32 tile）
///
/// - rope_ptr: 长度 32 的 [cos0, sin0, cos1, sin1, ...]（对应这个 32 维子块）
/// - eps: 数值稳定项
///
/// 注意：这个内核只看当前 3×32 子块：
/// - C 的行距用 `ldc = p.b_row_step_macro`
/// - K/Q/V 的总 N 可以都不一样，上层只要给对每个矩阵自己的 ldc 和 tile 起始指针即可。
#[target_feature(enable = "avx512fp16")]
pub unsafe fn matmul_finalize_rmsnorm_rope_inplace_3x128( // 同上，内部是 3×32
    c: *mut f16,          // 3×32，行距=ldc
    rope_ptr: *const f16, // 32 个交错相位
    eps: f16,
    p: &MatMulParams,
) {
    let ldc = p.b_row_step_macro;
    debug_assert_eq!(p.a_row_step_micro, 3);
    debug_assert_eq!(p.b_row_step_micro, 32);

    // 权重恒 1（长度 32，对这块 32 维做 RMS）
    const F16_ONE_BITS: u16 = 0x3C00;
    static ONES32: [f16; 32] = [f16::from_bits(F16_ONE_BITS); 32];

    // 对 3 行分别做 in-place RMSNorm（长度 32）
    rms_norm(c.add(0 * ldc), c.add(0 * ldc), 32, ONES32.as_ptr(), eps);
    rms_norm(c.add(1 * ldc), c.add(1 * ldc), 32, ONES32.as_ptr(), eps);
    rms_norm(c.add(2 * ldc), c.add(2 * ldc), 32, ONES32.as_ptr(), eps);

    // 然后对 3 行分别做 in-place RoPE 复数乘（长度 32）
    complex_mul(c.add(0 * ldc), rope_ptr, c.add(0 * ldc), 32);
    complex_mul(c.add(1 * ldc), rope_ptr, c.add(1 * ldc), 32);
    complex_mul(c.add(2 * ldc), rope_ptr, c.add(2 * ldc), 32);
}