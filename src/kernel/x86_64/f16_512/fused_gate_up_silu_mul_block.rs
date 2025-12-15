
#![allow(non_snake_case)]

use std::arch::x86_64::{
    _mm512_fmadd_ph, _mm512_loadu_ph, _mm512_mul_ph, _mm512_set1_ph,
    _mm512_storeu_ph, _mm512_setzero_ph,
};
use std::f16;

use crate::init::matmul_params::matmulParams;
use crate::kernel::x86_64::f16_512::activation::sigmoid512;

/// 逐 kc 累加：把 A×W_gate 与 A×W_up 的部分和累加到各自的 3×32 累加缓冲。
/// 约定：累加缓冲 gate_acc / up_acc 的行距 = 32（便于矢量读写）。
/// matmulParams 映射：
/// - a_row_step_macro = lda (= K of A)
/// - b_row_step_macro = ldc_acc (= 32)   <-- 用于 acc 的行距
/// - column_step_macro = kc
/// - a_row_step_micro = MR (=3)
/// - b_row_step_micro = NR (=32)
#[target_feature(enable = "avx512fp16")]
pub unsafe fn fused_update_gate_up_acc_block(
    a: *const f16,            // A tile: 3×kc, row stride = lda
    b_gate_panel: *const f16, // W_gate panel: kc×32 (row-major, row=kc, stride=32)
    b_up_panel: *const f16,   // W_up   panel: kc×32
    gate_acc: *mut f16,       // acc buffer: 3×32, row stride = 32
    up_acc: *mut f16,         // acc buffer: 3×32, row stride = 32
    param: &matmulParams,
) {
    debug_assert_eq!(param.a_row_step_micro, 3);
    debug_assert_eq!(param.b_row_step_micro, 32);
    debug_assert!(param.column_step_macro > 0);

    let lda = param.a_row_step_macro;   // A row stride (=K)
    let ldc_acc = param.b_row_step_macro; // = 32 (for acc)
    let kc  = param.column_step_macro;  // panel width
    let b_stride = 32usize;

    // A rows
    let a0 = a;
    let a1 = a.add(lda);
    let a2 = a.add(2 * lda);

    // load acc
    let mut g0 = _mm512_loadu_ph(gate_acc.add(0 * ldc_acc));
    let mut g1 = _mm512_loadu_ph(gate_acc.add(1 * ldc_acc));
    let mut g2 = _mm512_loadu_ph(gate_acc.add(2 * ldc_acc));

    let mut u0 = _mm512_loadu_ph(up_acc.add(0 * ldc_acc));
    let mut u1 = _mm512_loadu_ph(up_acc.add(1 * ldc_acc));
    let mut u2 = _mm512_loadu_ph(up_acc.add(2 * ldc_acc));

    // main loop over kc
    for k in 0..kc {
        let bg = _mm512_loadu_ph(b_gate_panel.add(k * b_stride));
        let bu = _mm512_loadu_ph(b_up_panel  .add(k * b_stride));

        let a0k = _mm512_set1_ph(*a0.add(k));
        let a1k = _mm512_set1_ph(*a1.add(k));
        let a2k = _mm512_set1_ph(*a2.add(k));

        g0 = _mm512_fmadd_ph(a0k, bg, g0);
        g1 = _mm512_fmadd_ph(a1k, bg, g1);
        g2 = _mm512_fmadd_ph(a2k, bg, g2);

        u0 = _mm512_fmadd_ph(a0k, bu, u0);
        u1 = _mm512_fmadd_ph(a1k, bu, u1);
        u2 = _mm512_fmadd_ph(a2k, bu, u2);
    }

    // store acc
    _mm512_storeu_ph(gate_acc.add(0 * ldc_acc), g0);
    _mm512_storeu_ph(gate_acc.add(1 * ldc_acc), g1);
    _mm512_storeu_ph(gate_acc.add(2 * ldc_acc), g2);

    _mm512_storeu_ph(up_acc.add(0 * ldc_acc), u0);
    _mm512_storeu_ph(up_acc.add(1 * ldc_acc), u1);
    _mm512_storeu_ph(up_acc.add(2 * ldc_acc), u2);
}

/// 整个 K 累加完成后的收尾： C = SiLU(gate_acc) ⊙ up_acc
/// 注意：这里写回 **C**，其行距应为 **N**，因此 matmulParams.b_row_step_macro 必须传入 N。

#[target_feature(enable = "avx512fp16")]
pub unsafe fn fused_finalize_gate_up_silu_mul_block(
    gate_acc: *const f16, // 3×32, row stride = 32
    up_acc: *const f16,   // 3×32, row stride = 32
    c: *mut f16,          // 3×32, row stride = ldc_out (=N)
    param: &matmulParams,
) {
    debug_assert_eq!(param.a_row_step_micro, 3);
    debug_assert_eq!(param.b_row_step_micro, 32);

    let ldc_out = param.b_row_step_macro; // <-- N (C 的行距)
    let ldc_acc = 32usize;

    // load acc
    let g0 = _mm512_loadu_ph(gate_acc.add(0 * ldc_acc));
    let g1 = _mm512_loadu_ph(gate_acc.add(1 * ldc_acc));
    let g2 = _mm512_loadu_ph(gate_acc.add(2 * ldc_acc));

    let u0 = _mm512_loadu_ph(up_acc.add(0 * ldc_acc));
    let u1 = _mm512_loadu_ph(up_acc.add(1 * ldc_acc));
    let u2 = _mm512_loadu_ph(up_acc.add(2 * ldc_acc));

    // SiLU(g) = g * sigmoid(g)
    let s0 = sigmoid512(g0);
    let s1 = sigmoid512(g1);
    let s2 = sigmoid512(g2);

    let sg0 = _mm512_mul_ph(g0, s0);
    let sg1 = _mm512_mul_ph(g1, s1);
    let sg2 = _mm512_mul_ph(g2, s2);

    // write C with ldc_out (= N)
    _mm512_storeu_ph(c.add(0 * ldc_out), _mm512_mul_ph(sg0, u0));
    _mm512_storeu_ph(c.add(1 * ldc_out), _mm512_mul_ph(sg1, u1));
    _mm512_storeu_ph(c.add(2 * ldc_out), _mm512_mul_ph(sg2, u2));
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::arch::is_x86_feature_detected;

    // f16 <-> f32 转换（与你现有风格一致）
    #[inline] fn f16v(v: f32) -> f16 { let h = half::f16::from_f32(v); f16::from_bits(h.to_bits()) }
    #[inline] fn to_f32(x: f16) -> f32 { half::f16::from_bits(x.to_bits()).to_f32() }
    fn all_close(a: &[f16], b: &[f16], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x,y)| (to_f32(*x)-to_f32(*y)).abs() <= tol)
    }

    // 用例 1：单次 kc=K，A/W_gate/W_up 全 1
    // 期望：G = K, U = K => C = SiLU(K) * K
    #[test]
    fn test_fused_update_finalize_k64_single_pass_all_ones() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("Skipping: CPU lacks avx512fp16");
            return;
        }
        unsafe {
            let mr=3usize; let nr=32usize;
            let kc=64usize; let k_total=64usize; // kc == K
            let lda=k_total;         // A 行距
            let ldc_acc=32usize;     // 累加缓冲行距（固定 32）
            let ldc_out=32usize;     // 输出 C 的行距（本 tile 的 N=32）

            // A: 3×K，全 1
            let a: Vec<f16> = (0..mr*lda).map(|_| f16v(1.0)).collect();

            // 面板：kc×32，全 1
            let gate_panel: Vec<f16> = (0..kc*nr).map(|_| f16v(1.0)).collect();
            let up_panel:   Vec<f16> = (0..kc*nr).map(|_| f16v(1.0)).collect();

            // 累加缓冲：3×32，清零
            let mut gate_acc: Vec<f16> = vec![f16v(0.0); mr*ldc_acc];
            let mut up_acc:   Vec<f16> = vec![f16v(0.0); mr*ldc_acc];

            // 调用期参数（update）：lda=K, ldc_acc=32, kc
            let param_update = matmulParams {
                a_row_step_macro: lda,
                b_row_step_macro: ldc_acc,
                column_step_macro: kc,
                a_row_step_micro: mr,
                b_row_step_micro: nr,
            };

            // 1) 单次 update（kc=K）
            fused_update_gate_up_acc_block(
                a.as_ptr(),
                gate_panel.as_ptr(),
                up_panel.as_ptr(),
                gate_acc.as_mut_ptr(),
                up_acc.as_mut_ptr(),
                &param_update,
            );

            // 输出 C：3×32
            let mut c: Vec<f16> = vec![f16v(0.0); mr*ldc_out];

            // 调用期参数（finalize）：ldc_out = 32
            let param_finalize = matmulParams {
                a_row_step_macro: lda,       // 未用
                b_row_step_macro: ldc_out,   // C 的行距（=N_tile=32）
                column_step_macro: kc,       // 未用
                a_row_step_micro: mr,
                b_row_step_micro: nr,
            };

            // 2) finalize
            fused_finalize_gate_up_silu_mul_block(
                gate_acc.as_ptr(),
                up_acc.as_ptr(),
                c.as_mut_ptr(),
                &param_finalize,
            );

            // 期望值
            let g = k_total as f32;
            let u = k_total as f32;
            let expected_val = (g / (1.0 + (-g).exp())) * g; // SiLU(g) * u, 且 u=g
            let expected: Vec<f16> = (0..mr*ldc_out).map(|_| f16v(expected_val)).collect();

            assert!(all_close(&c, &expected, 1e-2), "mismatch: got ~{:?}, expect ~{}", to_f32(c[0]), expected_val);
        }
    }

    // 用例 2：多次 kc 累加（两次 kc=64，K=128）
    // gate_panel=1.0，up_panel=2.0 => G=128, U=256 => C = SiLU(128) * 256
    #[test]
    fn test_fused_update_finalize_k128_two_pass_accumulate() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("Skipping: CPU lacks avx512fp16");
            return;
        }
        unsafe {
            let mr=3usize; let nr=32usize;
            let kc=64usize; let k_total=128usize; // 两次 update
            let lda=k_total;
            let ldc_acc=32usize;
            let ldc_out=32usize;

            // A: 3×K，全 1（注意 A 的行距=K）
            let a: Vec<f16> = (0..mr*lda).map(|_| f16v(1.0)).collect();

            // 面板：kc×32
            let gate_panel: Vec<f16> = (0..kc*nr).map(|_| f16v(1.0)).collect();  // 1.0
            let up_panel:   Vec<f16> = (0..kc*nr).map(|_| f16v(2.0)).collect();  // 2.0

            // 累加缓冲
            let mut gate_acc: Vec<f16> = vec![f16v(0.0); mr*ldc_acc];
            let mut up_acc:   Vec<f16> = vec![f16v(0.0); mr*ldc_acc];

            // 参数（update）
            let param_update = matmulParams {
                a_row_step_macro: lda,
                b_row_step_macro: ldc_acc,
                column_step_macro: kc,
                a_row_step_micro: mr,
                b_row_step_micro: nr,
            };

            // 两次 update（模拟 k0=0..64, 64..128 的两片）
            fused_update_gate_up_acc_block(
                a.as_ptr().add(0),             // A 的 k 偏移在内核里用列索引处理，这里直接同一 a_tile
                gate_panel.as_ptr(),
                up_panel.as_ptr(),
                gate_acc.as_mut_ptr(),
                up_acc.as_mut_ptr(),
                &param_update,
            );
            fused_update_gate_up_acc_block(
                a.as_ptr().add(64),            // A 第二段的列在 update 内核通过列 k 访问；这里传 a+64 起点即可
                gate_panel.as_ptr(),           // 为了简单，仍然用同一面板数据（值相同即可）
                up_panel.as_ptr(),
                gate_acc.as_mut_ptr(),
                up_acc.as_mut_ptr(),
                &param_update,
            );

            // 输出 C：3×32
            let mut c: Vec<f16> = vec![f16v(0.0); mr*ldc_out];

            // 参数（finalize）
            let param_finalize = matmulParams {
                a_row_step_macro: lda,      // 未用
                b_row_step_macro: ldc_out,  // C 的行距
                column_step_macro: kc,      // 未用
                a_row_step_micro: mr,
                b_row_step_micro: nr,
            };

            fused_finalize_gate_up_silu_mul_block(
                gate_acc.as_ptr(),
                up_acc.as_ptr(),
                c.as_mut_ptr(),
                &param_finalize,
            );

            // 期望：G = 128, U = 256
            let g = 128.0_f32;
            let u = 256.0_f32;
            let expected_val = (g / (1.0 + (-g).exp())) * u; // SiLU(g) * U
            let expected: Vec<f16> = (0..mr*ldc_out).map(|_| f16v(expected_val)).collect();

            assert!(all_close(&c, &expected, 5e-2), "mismatch: got ~{:?}, expect ~{}", to_f32(c[0]), expected_val);
        }
    }
}