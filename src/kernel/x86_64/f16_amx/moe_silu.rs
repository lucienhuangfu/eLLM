#![allow(non_snake_case)]

use std::f16;

use crate::kernel::x86_64::f16_amx::tile::{ensure_amx_ready, gemm_3x16_to_f32, AMX_MR, AMX_NR};

/// AMX-FP16 version of `f16_512::moe_silu::moe_silu_update_3x32`.
///
/// Only the gate/up GEMM accumulation is an AMX target. The final
/// `SiLU(gate) * up` pass is elementwise and should stay AVX512.
#[target_feature(enable = "amx-tile,amx-fp16")]
pub unsafe fn moe_silu_update_3x32(
    a_tile: *const f16,
    gate_panel: *const f16,
    up_panel: *const f16,
    gate_acc: *mut f16,
    up_acc: *mut f16,
    kc: usize,
) {
    unsafe { ensure_amx_ready() };

    let mut acc = [[0.0f32; AMX_NR]; AMX_MR];

    for half in 0..2 {
        unsafe {
            gemm_3x16_to_f32(
                a_tile,
                gate_panel.add(half * AMX_NR),
                acc.as_mut_ptr().cast(),
                kc,
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
                a_tile,
                up_panel.add(half * AMX_NR),
                acc.as_mut_ptr().cast(),
                kc,
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
        kc: usize,
        name: &str,
    ) {
        for r in 0..AMX_MR {
            for n in 0..32 {
                let mut expected = before[r * 32 + n] as f32;
                for k in 0..kc {
                    expected += (a[r * kc + k] as f32) * (panel[k * 32 + n] as f32);
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
    fn test_amx_moe_silu_update_3x32_gate_and_up() {
        if !is_x86_feature_detected!("amx-tile") || !is_x86_feature_detected!("amx-fp16") {
            eprintln!("skip: amx-tile/amx-fp16 not detected");
            return;
        }

        const KC: usize = 49;
        let a: Vec<f16> = (0..AMX_MR * KC)
            .map(|i| (((i * 7) % 31) as f32 * 0.006 - 0.09) as f16)
            .collect();
        let gate_panel: Vec<f16> = (0..KC * 32)
            .map(|i| (((i * 5) % 37) as f32 * 0.004 - 0.05) as f16)
            .collect();
        let up_panel: Vec<f16> = (0..KC * 32)
            .map(|i| (((i * 11) % 41) as f32 * 0.003 - 0.04) as f16)
            .collect();
        let mut gate_acc: Vec<f16> = (0..AMX_MR * 32)
            .map(|i| (0.02 + i as f32 * 0.0004) as f16)
            .collect();
        let mut up_acc: Vec<f16> = (0..AMX_MR * 32)
            .map(|i| (-0.01 + i as f32 * 0.0005) as f16)
            .collect();
        let gate_before = gate_acc.clone();
        let up_before = up_acc.clone();

        unsafe {
            moe_silu_update_3x32(
                a.as_ptr(),
                gate_panel.as_ptr(),
                up_panel.as_ptr(),
                gate_acc.as_mut_ptr(),
                up_acc.as_mut_ptr(),
                KC,
            );
        }

        check_acc(&a, &gate_panel, &gate_before, &gate_acc, KC, "gate");
        check_acc(&a, &up_panel, &up_before, &up_acc, KC, "up");
    }
}
