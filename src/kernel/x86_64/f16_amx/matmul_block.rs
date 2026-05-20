#![allow(non_snake_case)]

use std::f16;

use crate::common::matmul_params::MatMulParams;
use crate::kernel::x86_64::f16_amx::tile::{ensure_amx_ready, gemm_3x16_to_f32, AMX_MR, AMX_NR};

/// AMX-FP16 version of `f16_512::matmul_block::matmul_block`.
///
/// Shape:
/// - A tile: 3 x kc, row stride = lda
/// - B panel: kc x 32, packed row-major
/// - C tile: 3 x 32, row stride = ldc
///
/// AMX-FP16 accumulates into FP32 tiles. The kernel computes two 3x16
/// halves, adds the existing f16 C values in FP32, then writes f16 back.
#[target_feature(enable = "amx-tile,amx-fp16")]
pub unsafe fn matmul_block(a: *const f16, b_panel: *const f16, c: *mut f16, param: &MatMulParams) {
    debug_assert_eq!(param.a_row_step_micro, AMX_MR);
    debug_assert_eq!(param.b_row_step_micro, 32);
    debug_assert!(param.column_step_macro > 0);

    unsafe { ensure_amx_ready() };

    let lda = param.a_row_step_macro;
    let ldc = param.b_row_step_macro;
    let kc = param.column_step_macro;
    let mut acc = [[0.0f32; AMX_NR]; AMX_MR];

    for half in 0..2 {
        unsafe {
            gemm_3x16_to_f32(
                a,
                b_panel.add(half * AMX_NR),
                acc.as_mut_ptr().cast(),
                lda,
                32,
                kc,
            );
        }

        for r in 0..AMX_MR {
            for n in 0..AMX_NR {
                let dst = unsafe { c.add(r * ldc + half * AMX_NR + n) };
                unsafe {
                    *dst = ((*dst as f32) + acc[r][n]) as f16;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::arch::is_x86_feature_detected;

    fn f16v(x: f32) -> f16 {
        x as f16
    }

    fn assert_close(got: f16, expected: f32, tol: f32) {
        let got = got as f32;
        assert!(
            (got - expected).abs() <= tol,
            "got {got}, expected {expected}, diff {}",
            (got - expected).abs()
        );
    }

    #[test]
    fn test_amx_matmul_block_3x32_k64() {
        if !is_x86_feature_detected!("amx-tile") || !is_x86_feature_detected!("amx-fp16") {
            eprintln!("skip: amx-tile/amx-fp16 not detected");
            return;
        }

        const MR: usize = 3;
        const NR: usize = 32;
        const KC: usize = 64;
        const LDA: usize = KC;
        const LDC: usize = NR;

        let mut a = vec![0.0f16; MR * LDA];
        let mut b = vec![0.0f16; KC * NR];
        let mut c = vec![0.0f16; MR * LDC];

        for r in 0..MR {
            for k in 0..KC {
                a[r * LDA + k] = f16v(((r * 7 + k * 3) % 29) as f32 * 0.01 - 0.12);
            }
        }
        for k in 0..KC {
            for n in 0..NR {
                b[k * NR + n] = f16v(((k * 5 + n * 11) % 31) as f32 * 0.008 - 0.1);
            }
        }
        for r in 0..MR {
            for n in 0..NR {
                c[r * LDC + n] = f16v(0.25 + (r * NR + n) as f32 * 0.0003);
            }
        }

        let c_init = c.clone();
        let params = MatMulParams {
            a_row_step_micro: MR,
            b_row_step_micro: NR,
            column_step_macro: KC,
            a_row_step_macro: LDA,
            b_row_step_macro: LDC,
        };

        unsafe {
            matmul_block(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), &params);
        }

        for r in 0..MR {
            for n in 0..NR {
                let mut expected = c_init[r * LDC + n] as f32;
                for k in 0..KC {
                    expected += (a[r * LDA + k] as f32) * (b[k * NR + n] as f32);
                }
                assert_close(c[r * LDC + n], expected, 0.08);
            }
        }
    }

    #[test]
    fn test_amx_matmul_block_handles_k_tail() {
        if !is_x86_feature_detected!("amx-tile") || !is_x86_feature_detected!("amx-fp16") {
            eprintln!("skip: amx-tile/amx-fp16 not detected");
            return;
        }

        const MR: usize = 3;
        const NR: usize = 32;
        const KC: usize = 45;
        const LDA: usize = KC;
        const LDC: usize = NR;

        let a: Vec<f16> = (0..MR * LDA)
            .map(|i| f16v((i % 17) as f32 * 0.01 - 0.05))
            .collect();
        let b: Vec<f16> = (0..KC * NR)
            .map(|i| f16v((i % 19) as f32 * 0.009 - 0.07))
            .collect();
        let mut c = vec![0.0f16; MR * LDC];

        let params = MatMulParams {
            a_row_step_micro: MR,
            b_row_step_micro: NR,
            column_step_macro: KC,
            a_row_step_macro: LDA,
            b_row_step_macro: LDC,
        };

        unsafe {
            matmul_block(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), &params);
        }

        for r in 0..MR {
            for n in 0..NR {
                let mut expected = 0.0f32;
                for k in 0..KC {
                    expected += (a[r * LDA + k] as f32) * (b[k * NR + n] as f32);
                }
                assert_close(c[r * LDC + n], expected, 0.05);
            }
        }
    }
}
