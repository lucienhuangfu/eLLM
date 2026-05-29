// === kernel/x86_64/f16_512/moe_silu.rs ===
#![allow(non_snake_case)]

use std::f16;

/// update: MR=3, NR=32 固定
/// - A_tile: 3×kc（紧凑，行距=kc）
/// - gate_panel/up_panel: kc×32（行距=32）
/// - gate_acc/up_acc: 3×32（行距=32）
#[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
#[target_feature(enable = "avx512fp16")]
pub unsafe fn moe_silu_update_3x32(
    a_tile: *const f16,
    gate_panel: *const f16,
    up_panel: *const f16,
    gate_acc: *mut f16,
    up_acc: *mut f16,
    kc: usize,
) {
    let nr = 32usize;
    for row in 0..3 {
        for col in 0..nr {
            let mut gate_sum = *gate_acc.add(row * nr + col) as f32;
            let mut up_sum = *up_acc.add(row * nr + col) as f32;
            for kk in 0..kc {
                let a = *a_tile.add(row * kc + kk) as f32;
                gate_sum += a * (*gate_panel.add(kk * nr + col) as f32);
                up_sum += a * (*up_panel.add(kk * nr + col) as f32);
            }
            *gate_acc.add(row * nr + col) = gate_sum as f16;
            *up_acc.add(row * nr + col) = up_sum as f16;
        }
    }
}

/// finalize row: NR=32 固定
/// C_row[j] = SiLU(gate_row[j]) * up_row[j]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
#[target_feature(enable = "avx512fp16")]
pub unsafe fn moe_silu_finalize_row_32(gate_row: *const f16, up_row: *const f16, c_row: *mut f16) {
    for i in 0..32 {
        let g = *gate_row.add(i) as f32;
        let u = *up_row.add(i) as f32;
        let silu = g / (1.0 + (-g).exp());
        *c_row.add(i) = (silu * u) as f16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn random_vec(len: usize) -> Vec<f16> {
        let mut rng = rand::thread_rng();
        (0..len).map(|_| rng.gen_range(-1.0..1.0) as f16).collect()
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_moe_silu_update_3x32() {
        unsafe {
            let kc = 64;
            let mr = 3;
            let nr = 32;

            let a_tile = random_vec(mr * kc);
            let gate_panel = random_vec(kc * nr);
            let up_panel = random_vec(kc * nr);

            let mut gate_acc = random_vec(mr * nr);
            let mut up_acc = random_vec(mr * nr);

            let mut ref_gate_acc = gate_acc.clone();
            let mut ref_up_acc = up_acc.clone();

            moe_silu_update_3x32(
                a_tile.as_ptr(),
                gate_panel.as_ptr(),
                up_panel.as_ptr(),
                gate_acc.as_mut_ptr(),
                up_acc.as_mut_ptr(),
                kc,
            );

            // Reference
            for k in 0..kc {
                for m in 0..mr {
                    let a_val = a_tile[m * kc + k] as f32;
                    for n in 0..nr {
                        let g_w = gate_panel[k * nr + n] as f32;
                        let u_w = up_panel[k * nr + n] as f32;

                        let idx = m * nr + n;
                        let g_acc = ref_gate_acc[idx] as f32;
                        let u_acc = ref_up_acc[idx] as f32;

                        ref_gate_acc[idx] = (g_acc + a_val * g_w) as f16;
                        ref_up_acc[idx] = (u_acc + a_val * u_w) as f16;
                    }
                }
            }

            let tolerance = 0.15;
            for i in 0..gate_acc.len() {
                let diff_g = (gate_acc[i] as f32 - ref_gate_acc[i] as f32).abs();
                let diff_u = (up_acc[i] as f32 - ref_up_acc[i] as f32).abs();
                assert!(
                    diff_g < tolerance,
                    "Gate diff at {}: {} vs {}",
                    i,
                    gate_acc[i] as f32,
                    ref_gate_acc[i] as f32
                );
                assert!(
                    diff_u < tolerance,
                    "Up diff at {}: {} vs {}",
                    i,
                    up_acc[i] as f32,
                    ref_up_acc[i] as f32
                );
            }
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_moe_silu_finalize_row_32() {
        unsafe {
            let nr = 32;
            let gate_row = random_vec(nr);
            let up_row = random_vec(nr);
            let mut c_row = vec![0.0 as f16; nr];

            moe_silu_finalize_row_32(gate_row.as_ptr(), up_row.as_ptr(), c_row.as_mut_ptr());

            for i in 0..nr {
                let g = gate_row[i] as f32;
                let u = up_row[i] as f32;
                let silu = g / (1.0 + (-g).exp());
                let expected = silu * u;
                let got = c_row[i] as f32;

                assert!(
                    (got - expected).abs() < 0.01,
                    "Finalize diff at {}: {} vs {}",
                    i,
                    got,
                    expected
                );
            }
        }
    }
}
