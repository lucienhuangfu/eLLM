#![allow(non_snake_case)]

use std::f16;

use crate::kernel::common::matmul_params::MatMulParams;
use crate::kernel::x86_64::f16_amx::matmul_block::matmul_block;

/// AMX-FP16 version of
/// `f16_512::matmul_rms_complex::matmul_update_inplace_3x32_accum`.
///
/// This only replaces the matrix update. RMSNorm and RoPE remain better suited
/// to the existing AVX512 vector kernels.
#[target_feature(enable = "amx-tile,amx-fp16")]
pub unsafe fn matmul_update_inplace_3x32_accum(
    a: *const f16,
    b_panel: *const f16,
    c: *mut f16,
    lda: usize,
    ldc: usize,
    kc: usize,
) {
    let param = MatMulParams {
        a_row_step_micro: 3,
        b_row_step_micro: 32,
        column_step_macro: kc,
        a_row_step_macro: lda,
        b_row_step_macro: ldc,
    };

    unsafe { matmul_block(a, b_panel, c, &param) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::arch::is_x86_feature_detected;

    #[test]
    fn test_amx_matmul_update_inplace_3x32_accum() {
        if !is_x86_feature_detected!("amx-tile") || !is_x86_feature_detected!("amx-fp16") {
            eprintln!("skip: amx-tile/amx-fp16 not detected");
            return;
        }

        const MR: usize = 3;
        const NR: usize = 32;
        const KC: usize = 37;
        const LDA: usize = KC;
        const LDC: usize = 48;

        let a: Vec<f16> = (0..MR * LDA)
            .map(|i| (((i * 3) % 23) as f32 * 0.01 - 0.08) as f16)
            .collect();
        let b: Vec<f16> = (0..KC * NR)
            .map(|i| (((i * 5) % 29) as f32 * 0.007 - 0.06) as f16)
            .collect();
        let mut c = vec![0.0f16; MR * LDC];
        for r in 0..MR {
            for n in 0..NR {
                c[r * LDC + n] = (0.1 + (r * NR + n) as f32 * 0.001) as f16;
            }
        }
        let c_init = c.clone();

        unsafe {
            matmul_update_inplace_3x32_accum(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), LDA, LDC, KC);
        }

        for r in 0..MR {
            for n in 0..NR {
                let mut expected = c_init[r * LDC + n] as f32;
                for k in 0..KC {
                    expected += (a[r * LDA + k] as f32) * (b[k * NR + n] as f32);
                }
                let got = c[r * LDC + n] as f32;
                assert!(
                    (got - expected).abs() <= 0.05,
                    "mismatch row {r} col {n}: got {got}, expected {expected}"
                );
            }
        }
    }
}
