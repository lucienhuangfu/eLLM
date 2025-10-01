// === kernel/x86_64/f16_512/matmul_block.rs ===
#![allow(non_snake_case)]

use std::arch::x86_64::{_mm512_fmadd_ph, _mm512_loadu_ph, _mm512_set1_ph, _mm512_storeu_ph};
use std::f16;

use crate::init::matmul_params::MatMulParams;

/// 广播式 3x32 FP16 AVX-512 微核：
/// 约定把 (lda/ldc/kc) 映射进 MatMulParams 的 5 字段中：
/// - a_row_step_micro = MR (=3)
/// - b_row_step_micro = NR (=32)
/// - column_step_macro = kc
/// - a_row_step_macro = lda
/// - b_row_step_macro = ldc
///
/// A_tile: 3 x kc（行主，行距=lda）
/// B_panel: kc x 32（行主打包，每行 32 连续）
/// C_tile: 3 x 32（行主，行距=ldc）
#[inline(always)]
pub unsafe fn matmul_block(
    a: *const f16,        // A tile base: 3xkc
    b_panel: *const f16,  // packed B panel: kc x 32
    c: *mut f16,          // C tile base: 3x32
    param: &MatMulParams,
) {
    // 形状校验
    debug_assert_eq!(param.a_row_step_micro, 3);
    debug_assert_eq!(param.b_row_step_micro, 32);
    debug_assert!(param.column_step_macro > 0);

    // 取 stride/尺寸（元素计）
    let lda = param.a_row_step_macro;    // A 行距
    let ldc = param.b_row_step_macro;    // C 行距
    let kc  = param.column_step_macro;   // K 面板长度
    let b_stride = 32usize;              // B_panel 每行 32

    // A 三行基址
    let a0 = a;
    let a1 = a.add(lda);
    let a2 = a.add(2 * lda);

    // 读入 C 累加器
    let mut c_row0 = _mm512_loadu_ph(c.add(0 * ldc));
    let mut c_row1 = _mm512_loadu_ph(c.add(1 * ldc));
    let mut c_row2 = _mm512_loadu_ph(c.add(2 * ldc));

    // 主循环：按 k 广播 A 的标量 * B_panel 的向量做 FMA
    for k in 0..kc {
        let bvec = _mm512_loadu_ph(b_panel.add(k * b_stride));
        let a0b = _mm512_set1_ph(*a0.add(k));
        let a1b = _mm512_set1_ph(*a1.add(k));
        let a2b = _mm512_set1_ph(*a2.add(k));

        c_row0 = _mm512_fmadd_ph(a0b, bvec, c_row0);
        c_row1 = _mm512_fmadd_ph(a1b, bvec, c_row1);
        c_row2 = _mm512_fmadd_ph(a2b, bvec, c_row2);
    }

    // 写回
    _mm512_storeu_ph(c.add(0 * ldc), c_row0);
    _mm512_storeu_ph(c.add(1 * ldc), c_row1);
    _mm512_storeu_ph(c.add(2 * ldc), c_row2);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::arch::is_x86_feature_detected;
    use std::f16;

    #[inline] fn f16v(v: f32) -> f16 { let h = half::f16::from_f32(v); f16::from_bits(h.to_bits()) }
    #[inline] fn to_f32(x: f16) -> f32 { half::f16::from_bits(x.to_bits()).to_f32() }
    fn all_close(a: &[f16], b: &[f16], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x,y)| (to_f32(*x)-to_f32(*y)).abs() <= tol)
    }

    #[test]
    fn test_broadcast_microkernel_3x32_k64_ones() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("Skipping: CPU lacks avx512fp16");
            return;
        }
        unsafe {
            let mr=3; let nr=32; let kc=64;
            let lda=kc; let ldc=nr;

            let param = MatMulParams {
                a_row_step_macro: lda,
                b_row_step_macro: ldc,
                column_step_macro: kc,
                a_row_step_micro: mr,
                b_row_step_micro: nr,
            };

            let a: Vec<f16> = (0..mr*kc).map(|_| f16v(1.0)).collect();
            let b_panel: Vec<f16> = (0..kc*nr).map(|_| f16v(1.0)).collect();
            let mut c: Vec<f16> = (0..mr*ldc).map(|_| f16v(0.0)).collect();

            matmul_block(a.as_ptr(), b_panel.as_ptr(), c.as_mut_ptr(), &param);

            let expected: Vec<f16> = (0..mr*ldc).map(|_| f16v(kc as f32)).collect();
            assert!(all_close(&c, &expected, 1e-3));
        }
    }
}