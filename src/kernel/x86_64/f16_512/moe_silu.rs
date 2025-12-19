// === kernel/x86_64/f16_512/moe_silu.rs ===
#![allow(non_snake_case)]

use std::arch::x86_64::{
    _mm512_fmadd_ph, _mm512_loadu_ph, _mm512_mul_ph, _mm512_set1_ph, _mm512_storeu_ph,
};
use std::f16;

use crate::kernel::x86_64::f16_512::activation::sigmoid512;

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
    let lda = kc;
    let ldc = 32usize;
    let bstride = 32usize;

    let a0 = a_tile;
    let a1 = a_tile.add(lda);
    let a2 = a_tile.add(2 * lda);

    let mut g0 = _mm512_loadu_ph(gate_acc.add(0 * ldc));
    let mut g1 = _mm512_loadu_ph(gate_acc.add(1 * ldc));
    let mut g2 = _mm512_loadu_ph(gate_acc.add(2 * ldc));

    let mut u0 = _mm512_loadu_ph(up_acc.add(0 * ldc));
    let mut u1 = _mm512_loadu_ph(up_acc.add(1 * ldc));
    let mut u2 = _mm512_loadu_ph(up_acc.add(2 * ldc));

    for kk in 0..kc {
        let bg = _mm512_loadu_ph(gate_panel.add(kk * bstride));
        let bu = _mm512_loadu_ph(up_panel.add(kk * bstride));

        let a0k = _mm512_set1_ph(*a0.add(kk));
        let a1k = _mm512_set1_ph(*a1.add(kk));
        let a2k = _mm512_set1_ph(*a2.add(kk));

        g0 = _mm512_fmadd_ph(a0k, bg, g0);
        g1 = _mm512_fmadd_ph(a1k, bg, g1);
        g2 = _mm512_fmadd_ph(a2k, bg, g2);

        u0 = _mm512_fmadd_ph(a0k, bu, u0);
        u1 = _mm512_fmadd_ph(a1k, bu, u1);
        u2 = _mm512_fmadd_ph(a2k, bu, u2);
    }

    _mm512_storeu_ph(gate_acc.add(0 * ldc), g0);
    _mm512_storeu_ph(gate_acc.add(1 * ldc), g1);
    _mm512_storeu_ph(gate_acc.add(2 * ldc), g2);

    _mm512_storeu_ph(up_acc.add(0 * ldc), u0);
    _mm512_storeu_ph(up_acc.add(1 * ldc), u1);
    _mm512_storeu_ph(up_acc.add(2 * ldc), u2);
}

/// finalize row: NR=32 固定
/// C_row[j] = SiLU(gate_row[j]) * up_row[j]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
#[target_feature(enable = "avx512fp16")]
pub unsafe fn moe_silu_finalize_row_32(
    gate_row: *const f16,
    up_row: *const f16,
    c_row: *mut f16,
) {
    let g = _mm512_loadu_ph(gate_row);
    let u = _mm512_loadu_ph(up_row);

    let s = sigmoid512(g);          // sigmoid(g)
    let silu = _mm512_mul_ph(g, s); // g * sigmoid(g)

    _mm512_storeu_ph(c_row, _mm512_mul_ph(silu, u));
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::arch::is_x86_feature_detected;
    use std::mem;

    use crate::kernel::generic::from_f32::FromF32; // f16::from_f32

    #[inline]
    fn f16_from_f32(x: f32) -> f16 {
        f16::from_f32(x)
    }

    /// 手写 f16 -> f32，只用于测试里的误差比较 / 打印
    #[inline]
    fn f32_from_f16(x: f16) -> f32 {
        let bits: u16 = unsafe { mem::transmute(x) };
        let sign = ((bits & 0x8000) as u32) << 16;
        let exp = (bits & 0x7C00) >> 10;
        let mant = bits & 0x03FF;

        let f_bits: u32 = if exp == 0 {
            if mant == 0 {
                sign
            } else {
                let mut e: i32 = -14;
                let mut m = mant as u32;
                while (m & 0x0400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x03FF;
                let exp_f = (e + 127) as u32;
                sign | (exp_f << 23) | (m << 13)
            }
        } else if exp == 0x1F {
            let exp_f = 0xFFu32;
            sign | (exp_f << 23) | ((mant as u32) << 13)
        } else {
            let exp_f = (exp as i32 - 15 + 127) as u32;
            sign | (exp_f << 23) | ((mant as u32) << 13)
        };

        f32::from_bits(f_bits)
    }

    fn approx_eq(a: f16, b: f16, tol: f32) -> bool {
        (f32_from_f16(a) - f32_from_f16(b)).abs() <= tol
    }

    /// 标量参考：gate_acc/up_acc += A(3×kc) × panel(kc×32)
    fn reference_update(
        a: &[f16],              // 3×kc, lda=kc
        gate_panel: &[f16],     // kc×32
        up_panel: &[f16],       // kc×32
        gate_acc: &mut [f16],   // 3×32
        up_acc: &mut [f16],     // 3×32
        kc: usize,
    ) {
        let lda = kc;
        let ldc = 32;

        for r in 0..3 {
            for j in 0..32 {
                let mut g = f32_from_f16(gate_acc[r * ldc + j]);
                let mut u = f32_from_f16(up_acc[r * ldc + j]);
                for kk in 0..kc {
                    let av = f32_from_f16(a[r * lda + kk]);
                    let bg = f32_from_f16(gate_panel[kk * 32 + j]);
                    let bu = f32_from_f16(up_panel[kk * 32 + j]);
                    g += av * bg;
                    u += av * bu;
                }
                gate_acc[r * ldc + j] = f16_from_f32(g);
                up_acc[r * ldc + j] = f16_from_f32(u);
            }
        }
    }

    /// 标量参考：SiLU(g) * u
    fn reference_finalize_row(gate: &[f16], up: &[f16]) -> Vec<f16> {
        let mut out = vec![f16_from_f32(0.0); 32];
        for j in 0..32 {
            let g = f32_from_f16(gate[j]);
            let u = f32_from_f16(up[j]);
            let sig = 1.0f32 / (1.0f32 + (-g).exp());
            let silu = g * sig;
            out[j] = f16_from_f32(silu * u);
        }
        out
    }

    #[test]
    fn test_moe_silu_update_3x32_basic() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("Skipping test_moe_silu_update_3x32_basic: avx512fp16 not detected");
            return;
        }

        const KC: usize = 13; // 故意不是 32 的倍数
        const LDC: usize = 32;

        // A_tile: 3×KC（紧凑行距=KC）
        let mut a = vec![f16_from_f32(0.0); 3 * KC];
        for r in 0..3 {
            for kk in 0..KC {
                let v = 0.01f32 * (r as f32) + 0.002f32 * (kk as f32) - 0.03f32;
                a[r * KC + kk] = f16_from_f32(v);
            }
        }

        // panels: KC×32
        let mut gate_panel = vec![f16_from_f32(0.0); KC * 32];
        let mut up_panel   = vec![f16_from_f32(0.0); KC * 32];
        for kk in 0..KC {
            for j in 0..32 {
                let vg = 0.02f32 * (kk as f32) + 0.003f32 * (j as f32) - 0.1f32;
                let vu = -0.015f32 * (kk as f32) + 0.004f32 * (j as f32) + 0.05f32;
                gate_panel[kk * 32 + j] = f16_from_f32(vg);
                up_panel  [kk * 32 + j] = f16_from_f32(vu);
            }
        }

        // acc 初始化非零，确保是“累加”语义
        let mut gate_acc = vec![f16_from_f32(0.1); 3 * LDC];
        let mut up_acc   = vec![f16_from_f32(-0.2); 3 * LDC];

        let mut gate_ref = gate_acc.clone();
        let mut up_ref   = up_acc.clone();

        // reference
        reference_update(&a, &gate_panel, &up_panel, &mut gate_ref, &mut up_ref, KC);

        // avx
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

        for idx in 0..(3 * LDC) {
            let g = gate_acc[idx];
            let e = gate_ref[idx];
            assert!(
                approx_eq(g, e, 1e-1),
                "gate mismatch at {}: got {:?} (f32={}), exp {:?} (f32={})",
                idx,
                g,
                f32_from_f16(g),
                e,
                f32_from_f16(e),
            );

            let u = up_acc[idx];
            let ue = up_ref[idx];
            assert!(
                approx_eq(u, ue, 1e-1),
                "up mismatch at {}: got {:?} (f32={}), exp {:?} (f32={})",
                idx,
                u,
                f32_from_f16(u),
                ue,
                f32_from_f16(ue),
            );
        }
    }

    #[test]
    fn test_moe_silu_finalize_row_32_basic() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("Skipping test_moe_silu_finalize_row_32_basic: avx512fp16 not detected");
            return;
        }

        // gate/up 一行各 32
        let mut gate = vec![f16_from_f32(0.0); 32];
        let mut up   = vec![f16_from_f32(0.0); 32];
        for j in 0..32 {
            let g = 0.05f32 * (j as f32) - 0.6f32;      // 让正负都有
            let u = -0.02f32 * (j as f32) + 0.3f32;
            gate[j] = f16_from_f32(g);
            up[j]   = f16_from_f32(u);
        }

        let expected = reference_finalize_row(&gate, &up);

        let mut c = vec![f16_from_f32(0.0); 32];
        unsafe {
            moe_silu_finalize_row_32(gate.as_ptr(), up.as_ptr(), c.as_mut_ptr());
        }

        for j in 0..32 {
            let g = c[j];
            let e = expected[j];
            assert!(
                approx_eq(g, e, 2e-1),
                "C mismatch at {}: got {:?} (f32={}), exp {:?} (f32={})",
                j,
                g,
                f32_from_f16(g),
                e,
                f32_from_f16(e),
            );
        }
    }
}