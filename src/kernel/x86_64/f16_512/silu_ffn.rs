// === kernel/x86_64/f16_512/silu_ffn.rs ===
#![allow(non_snake_case)]

use std::arch::x86_64::{
    _mm512_fmadd_ph, _mm512_loadu_ph, _mm512_mul_ph, _mm512_set1_ph, _mm512_storeu_ph,
};
use std::f16;

use crate::init::matmul_params::MatMulParams;
use crate::kernel::x86_64::f16_512::activation::sigmoid512;

/// K 方向逐 kc 累加：
///   gate_acc/up_acc += A_tile(3×kc) × [W_gate/W_up]_panel(kc×32)
///
/// 约定 / MatMulParams 映射：
/// - a_row_step_macro = lda (= K of A)
/// - b_row_step_macro = ldc_acc (= 32)   <-- acc 的行距
/// - column_step_macro = kc
/// - a_row_step_micro = MR (=3)
/// - b_row_step_micro = NR (=32)
///
/// A_tile:       3×kc  （行主，行距=lda）
/// gate_panel:   kc×32（行主，每行 32 连续）
/// up_panel:     kc×32（行主，每行 32 连续）
/// gate_acc/up:  3×32  （行主，行距=32）
#[target_feature(enable = "avx512fp16")]
pub unsafe fn silu_update_3x32(
    a: *const f16,
    gate_panel: *const f16,
    up_panel: *const f16,
    gate_acc: *mut f16,
    up_acc: *mut f16,
    param: &MatMulParams,
) {
    debug_assert_eq!(param.a_row_step_micro, 3);
    debug_assert_eq!(param.b_row_step_micro, 32);
    debug_assert!(param.column_step_macro > 0);

    let lda      = param.a_row_step_macro;   // A 行距 (=K)
    let ldc_acc  = param.b_row_step_macro;   // acc 行距 (=32)
    let kc       = param.column_step_macro;  // panel 宽度
    let b_stride = 32usize;

    // A 三行基址
    let a0 = a;
    let a1 = a.add(lda);
    let a2 = a.add(2 * lda);

    // 读 acc（3×32）
    let mut g0 = _mm512_loadu_ph(gate_acc.add(0 * ldc_acc));
    let mut g1 = _mm512_loadu_ph(gate_acc.add(1 * ldc_acc));
    let mut g2 = _mm512_loadu_ph(gate_acc.add(2 * ldc_acc));

    let mut u0 = _mm512_loadu_ph(up_acc.add(0 * ldc_acc));
    let mut u1 = _mm512_loadu_ph(up_acc.add(1 * ldc_acc));
    let mut u2 = _mm512_loadu_ph(up_acc.add(2 * ldc_acc));

    // kc 方向主循环
    for k in 0..kc {
        let bg = _mm512_loadu_ph(gate_panel.add(k * b_stride));
        let bu = _mm512_loadu_ph(up_panel  .add(k * b_stride));

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

    // 写回 acc
    _mm512_storeu_ph(gate_acc.add(0 * ldc_acc), g0);
    _mm512_storeu_ph(gate_acc.add(1 * ldc_acc), g1);
    _mm512_storeu_ph(gate_acc.add(2 * ldc_acc), g2);

    _mm512_storeu_ph(up_acc.add(0 * ldc_acc), u0);
    _mm512_storeu_ph(up_acc.add(1 * ldc_acc), u1);
    _mm512_storeu_ph(up_acc.add(2 * ldc_acc), u2);
}

/// 整个 K 累加完成后的收尾：
///   C_tile = SiLU(gate_acc) ⊙ up_acc
///
/// 约定：
/// - gate_acc / up_acc: 3×32，行距=32
/// - C_tile:            3×32，行距=ldc_out (= N_tile)
/// - MatMulParams.b_row_step_macro = ldc_out
#[target_feature(enable = "avx512fp16")]
pub unsafe fn silu_finish_3x32(
    gate_acc: *const f16,
    up_acc: *const f16,
    c: *mut f16,
    param: &MatMulParams,
) {
    debug_assert_eq!(param.a_row_step_micro, 3);
    debug_assert_eq!(param.b_row_step_micro, 32);

    let ldc_out = param.b_row_step_macro; // C 行距
    let ldc_acc = 32usize;

    // 读 acc
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

    // 写回 C（3×32，行距=ldc_out）
    _mm512_storeu_ph(c.add(0 * ldc_out), _mm512_mul_ph(sg0, u0));
    _mm512_storeu_ph(c.add(1 * ldc_out), _mm512_mul_ph(sg1, u1));
    _mm512_storeu_ph(c.add(2 * ldc_out), _mm512_mul_ph(sg2, u2));
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::arch::is_x86_feature_detected;
    use std::mem;

    use crate::kernel::generic::from_f32::FromF32; // 提供 f16::from_f32

    #[inline]
    fn f16_from_f32(x: f32) -> f16 {
        f16::from_f32(x)
    }

    /// 手写 f16 -> f32，只在测试里用于误差比较 / 打印
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
        let da = f32_from_f16(a);
        let db = f32_from_f16(b);
        (da - db).abs() <= tol
    }

    /// 标量参考：gate_acc/up_acc += A(3×Kc) × panel(Kc×32)
    fn reference_silu_update(
        a: &[f16],           // 3×Kc, lda = Kc
        gate_panel: &[f16],  // Kc×32
        up_panel: &[f16],    // Kc×32
        gate_acc: &mut [f16],// 3×32
        up_acc: &mut [f16],  // 3×32
        kc: usize,
    ) {
        let lda = kc;
        let ldc_acc = 32;

        for i in 0..3 {
            for j in 0..32 {
                let mut g = f32_from_f16(gate_acc[i * ldc_acc + j]);
                let mut u = f32_from_f16(up_acc[i * ldc_acc + j]);
                for kk in 0..kc {
                    let a_ = f32_from_f16(a[i * lda + kk]);
                    let bg = f32_from_f16(gate_panel[kk * 32 + j]);
                    let bu = f32_from_f16(up_panel  [kk * 32 + j]);
                    g += a_ * bg;
                    u += a_ * bu;
                }
                gate_acc[i * ldc_acc + j] = f16_from_f32(g);
                up_acc[i * ldc_acc + j]   = f16_from_f32(u);
            }
        }
    }

    /// 标量 SiLU(g) ⊙ up
    fn reference_silu_finish(
        gate_acc: &[f16], // 3×32, 行距=32
        up_acc: &[f16],   // 3×32
    ) -> Vec<f16> {
        let ldc_acc = 32;
        let mut out = vec![f16_from_f32(0.0); 3 * ldc_acc];

        for i in 0..3 {
            for j in 0..32 {
                let g = f32_from_f16(gate_acc[i * ldc_acc + j]);
                let u = f32_from_f16(up_acc[i * ldc_acc + j]);
                let sig = 1.0f32 / (1.0f32 + (-g).exp());
                let silu = g * sig;
                out[i * ldc_acc + j] = f16_from_f32(silu * u);
            }
        }
        out
    }

    #[test]
    fn test_silu_update_3x32_basic() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("Skipping test_silu_update_3x32_basic: avx512fp16 not detected");
            return;
        }

        const KC: usize = 16;
        const LDA: usize = KC;
        const LDC_ACC: usize = 32;

        // A: 3×KC
        let mut a = vec![f16_from_f32(0.0); 3 * LDA];
        for i in 0..3 {
            for k in 0..KC {
                let v = 0.01f32 * (i as f32) + 0.001f32 * (k as f32);
                a[i * LDA + k] = f16_from_f32(v);
            }
        }

        // gate_panel / up_panel: KC×32
        let mut gate_panel = vec![f16_from_f32(0.0); KC * 32];
        let mut up_panel   = vec![f16_from_f32(0.0); KC * 32];
        for k in 0..KC {
            for j in 0..32 {
                let vg = 0.02f32 * (k as f32) + 0.003f32 * (j as f32);
                let vu = 0.015f32 * (k as f32) - 0.002f32 * (j as f32);
                gate_panel[k * 32 + j] = f16_from_f32(vg);
                up_panel  [k * 32 + j] = f16_from_f32(vu);
            }
        }

        // acc 初始化：随便给点非零，方便看 accum 行为
        let mut gate_acc = vec![f16_from_f32(0.1); 3 * LDC_ACC];
        let mut up_acc   = vec![f16_from_f32(0.2); 3 * LDC_ACC];

        let mut gate_acc_ref = gate_acc.clone();
        let mut up_acc_ref   = up_acc.clone();

        // 标量参考
        reference_silu_update(
            &a,
            &gate_panel,
            &up_panel,
            &mut gate_acc_ref,
            &mut up_acc_ref,
            KC,
        );

        // AVX512 实现
        let param = MatMulParams {
            a_row_step_macro: LDA,          // lda = K
            b_row_step_macro: LDC_ACC,      // acc 行距 = 32
            column_step_macro: KC,          // kc
            a_row_step_micro: 3,           // MR
            b_row_step_micro: 32,          // NR
        };

        unsafe {
            silu_update_3x32(
                a.as_ptr(),
                gate_panel.as_ptr(),
                up_panel.as_ptr(),
                gate_acc.as_mut_ptr(),
                up_acc.as_mut_ptr(),
                &param,
            );
        }

        for idx in 0..gate_acc.len() {
            let g = gate_acc[idx];
            let g_ref = gate_acc_ref[idx];
            assert!(
                approx_eq(g, g_ref, 1e-1),
                "gate_acc mismatch at {}: got {:?} (f32={}), exp {:?} (f32={})",
                idx,
                g,
                f32_from_f16(g),
                g_ref,
                f32_from_f16(g_ref),
            );

            let u = up_acc[idx];
            let u_ref = up_acc_ref[idx];
            assert!(
                approx_eq(u, u_ref, 1e-1),
                "up_acc mismatch at {}: got {:?} (f32={}), exp {:?} (f32={})",
                idx,
                u,
                f32_from_f16(u),
                u_ref,
                f32_from_f16(u_ref),
            );
        }
    }

    #[test]
    fn test_silu_finish_3x32_basic() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("Skipping test_silu_finish_3x32_basic: avx512fp16 not detected");
            return;
        }

        const LDC_ACC: usize = 32;

        // 构造 gate_acc / up_acc: 3×32
        let mut gate_acc = vec![f16_from_f32(0.0); 3 * LDC_ACC];
        let mut up_acc   = vec![f16_from_f32(0.0); 3 * LDC_ACC];
        for i in 0..3 {
            for j in 0..32 {
                let g = 0.05f32 * (i as f32) + 0.01f32 * (j as f32);
                let u = 0.02f32 * (i as f32) - 0.015f32 * (j as f32);
                gate_acc[i * LDC_ACC + j] = f16_from_f32(g);
                up_acc  [i * LDC_ACC + j] = f16_from_f32(u);
            }
        }

        // 标量参考 SiLU(g) ⊙ up
        let c_ref = reference_silu_finish(&gate_acc, &up_acc);

        // AVX512 输出 C，行距我们就设成 32（一个 tile）
        let ldc_out = 32usize;
        let mut c = vec![f16_from_f32(0.0); 3 * ldc_out];

        let param = MatMulParams {
            a_row_step_macro: 3,        // MR（这里不重要）
            b_row_step_macro: ldc_out,  // ldc_out
            column_step_macro: 1,       // 不用
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        unsafe {
            silu_finish_3x32(
                gate_acc.as_ptr(),
                up_acc.as_ptr(),
                c.as_mut_ptr(),
                &param,
            );
        }

        for idx in 0..c.len() {
            let v = c[idx];
            let v_ref = c_ref[idx];
            assert!(
                approx_eq(v, v_ref, 2e-1),
                "C mismatch at {}: got {:?} (f32={}), exp {:?} (f32={})",
                idx,
                v,
                f32_from_f16(v),
                v_ref,
                f32_from_f16(v_ref),
            );
        }
    }
}