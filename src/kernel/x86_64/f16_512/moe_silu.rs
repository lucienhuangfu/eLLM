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