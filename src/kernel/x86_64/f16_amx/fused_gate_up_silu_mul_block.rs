#![allow(non_snake_case)]

use std::f16;

use crate::common::matmul_params::MatMulParams;
use crate::kernel::x86_64::f16_amx::tile::{ensure_amx_ready, gemm_3x16_to_f32, AMX_MR, AMX_NR};

/// AMX-FP16 version of
/// `f16_512::fused_gate_up_silu_mul_block::fused_update_gate_up_acc_block`.
///
/// This file intentionally mirrors the AVX512 file, but only provides the AMX
/// part that actually benefits from tile dot-product: the gate/up GEMM update.
#[target_feature(enable = "amx-tile,amx-fp16")]
pub unsafe fn fused_update_gate_up_acc_block(
    a: *const f16,
    b_gate_panel: *const f16,
    b_up_panel: *const f16,
    gate_acc: *mut f16,
    up_acc: *mut f16,
    param: &MatMulParams,
) {
    debug_assert_eq!(param.a_row_step_micro, AMX_MR);
    debug_assert_eq!(param.b_row_step_micro, 32);
    debug_assert!(param.column_step_macro > 0);

    unsafe { ensure_amx_ready() };

    let kc = param.column_step_macro;
    let lda = param.a_row_step_macro;
    let mut acc = [[0.0f32; AMX_NR]; AMX_MR];

    for half in 0..2 {
        unsafe {
            gemm_3x16_to_f32(
                a,
                b_gate_panel.add(half * AMX_NR),
                acc.as_mut_ptr().cast(),
                lda,
                32,
                kc,
            );
        }
        for r in 0..AMX_MR {
            for n in 0..AMX_NR {
                let dst = unsafe { gate_acc.add(r * 32 + half * AMX_NR + n) };
                unsafe {
                    *dst = ((*dst as f32) + acc[r][n]) as f16;
                }
            }
        }

        unsafe {
            gemm_3x16_to_f32(
                a,
                b_up_panel.add(half * AMX_NR),
                acc.as_mut_ptr().cast(),
                lda,
                32,
                kc,
            );
        }
        for r in 0..AMX_MR {
            for n in 0..AMX_NR {
                let dst = unsafe { up_acc.add(r * 32 + half * AMX_NR + n) };
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

    fn check_acc(
        a: &[f16],
        panel: &[f16],
        before: &[f16],
        after: &[f16],
        lda: usize,
        kc: usize,
        name: &str,
    ) {
        for r in 0..AMX_MR {
            for n in 0..32 {
                let mut expected = before[r * 32 + n] as f32;
                for k in 0..kc {
                    expected += (a[r * lda + k] as f32) * (panel[k * 32 + n] as f32);
                }
                let got = after[r * 32 + n] as f32;
                assert!(
                    (got - expected).abs() <= 0.06,
                    "{name} mismatch row {r} col {n}: got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_amx_fused_update_gate_up_acc_block() {
        if !is_x86_feature_detected!("amx-tile") || !is_x86_feature_detected!("amx-fp16") {
            eprintln!("skip: amx-tile/amx-fp16 not detected");
            return;
        }

        const KC: usize = 35;
        const LDA: usize = 64;

        let mut a = vec![0.0f16; AMX_MR * LDA];
        for r in 0..AMX_MR {
            for k in 0..KC {
                a[r * LDA + k] = (((r * 13 + k * 7) % 43) as f32 * 0.004 - 0.08) as f16;
            }
        }
        let gate_panel: Vec<f16> = (0..KC * 32)
            .map(|i| (((i * 3) % 29) as f32 * 0.005 - 0.05) as f16)
            .collect();
        let up_panel: Vec<f16> = (0..KC * 32)
            .map(|i| (((i * 17) % 31) as f32 * 0.004 - 0.04) as f16)
            .collect();
        let mut gate_acc: Vec<f16> = (0..AMX_MR * 32)
            .map(|i| (0.03 + i as f32 * 0.0002) as f16)
            .collect();
        let mut up_acc: Vec<f16> = (0..AMX_MR * 32)
            .map(|i| (0.04 - i as f32 * 0.0003) as f16)
            .collect();
        let gate_before = gate_acc.clone();
        let up_before = up_acc.clone();

        let params = MatMulParams {
            a_row_step_micro: AMX_MR,
            b_row_step_micro: 32,
            column_step_macro: KC,
            a_row_step_macro: LDA,
            b_row_step_macro: 32,
        };

        unsafe {
            fused_update_gate_up_acc_block(
                a.as_ptr(),
                gate_panel.as_ptr(),
                up_panel.as_ptr(),
                gate_acc.as_mut_ptr(),
                up_acc.as_mut_ptr(),
                &params,
            );
        }

        check_acc(&a, &gate_panel, &gate_before, &gate_acc, LDA, KC, "gate");
        check_acc(&a, &up_panel, &up_before, &up_acc, LDA, KC, "up");
    }
}
