
use std::arch::x86_64::{
    _mm512_fmadd_ph, _mm512_loadu_ph, _mm512_set1_ph, _mm512_setzero_ph, _mm512_storeu_ph,
};
use std::f16;

use crate::init::matmul_params::MatMulParams;
///⚠️⚠️⚠️ 核心调整是 从最后累加变为了broadcast
///⚠️⚠️⚠️ 这里需要注意 我们把b微核调整成了32 
/// ⚠️⚠️⚠️还需要注意 使用的是loadu 如果有问题 可能是没对齐 可以改回load
/// matmul_block原版本 无累加k导致运算错误

/* 
 Micro-kernel (broadcast style) for FP16 AVX-512:
Computes a 3x32 tile of C in row-major:
C_tile[0..3, 0..32] += A_tile[0..3, 0..Kc]  *  B_panel[0..Kc, 0..32]

Layout assumptions:
A_tile: 3 rows by Kc columns, row-major, row stride = param.column (lda in elements)
B_panel: Kc rows by 32 columns, *packed/panelled*, row-major, row stride = 32 (contiguous)
C_tile: 3 rows by 32 columns, row-major, row stride = param.b_row (ldc in elements)
*/



#[inline(always)]
pub unsafe fn matmul_block(
    a: *const f16,        // base of A tile: A[0,0] of the 3xKc block
    b_panel: *const f16,  // base of packed B panel: Bp[0,0] of the Kc x 32 block (row-major, stride 32)
    c: *mut f16,          // base of C tile: C[0,0] of the 3x32 block (row-major, stride = param.b_row)
    param: &MatMulParams,
) {
    // ---- Invariants / sanity checks ----
    assert_eq!(param.a_row_step_micro, 3, "micro-kernel expects 3 rows of A/C");
    assert_eq!(param.b_row_step_micro, 32, "micro-kernel expects NR=32 for C columns");
    assert!(param.column_step_macro > 0, "Kc must be > 0");

    // ---- Strides (elements, not bytes) ----
    // lda: row stride for A (elements per full row of A in the parent matrix)
    // ldc: row stride for C (elements per full row of C in the parent matrix)
    let lda = param.column;
    let ldc = param.b_row;

    // Kc (the K panel length we will iterate over)
    let kc = param.column_step_macro;

    // ---- Row bases for A (3 rows) ----
    let a0 = a;             // A[0, 0]
    let a1 = a.add(lda);    // A[1, 0]
    let a2 = a.add(2 * lda);// A[2, 0]

    // ---- Initialize vector accumulators with current C (so we do +=) ----
    // Each accumulator holds 32 FP16 values for one row of the 3x32 tile.
    let mut c_row0 = _mm512_loadu_ph(c.add(0 * ldc)); // load C[0, 0..31]
    let mut c_row1 = _mm512_loadu_ph(c.add(1 * ldc)); // load C[1, 0..31]
    let mut c_row2 = _mm512_loadu_ph(c.add(2 * ldc)); // load C[2, 0..31]

    // ---- Main K loop: broadcast A scalars, FMA with B_panel row vectors ----
    for k in 0..kc {
        // Load one 32-lane FP16 vector: B_panel[k, 0..31]
        // Since B_panel is packed row-major with stride 32, offset = k * 32
        let bvec = _mm512_loadu_ph(b_panel.add(k * 32));

        // Broadcast the three A scalars A[0,k], A[1,k], A[2,k]
        let a0b = _mm512_set1_ph(*a0.add(k));
        let a1b = _mm512_set1_ph(*a1.add(k));
        let a2b = _mm512_set1_ph(*a2.add(k));

        // FMA: Crows += Arow_k * Brow_k
        c_row0 = _mm512_fmadd_ph(a0b, bvec, c_row0);
        c_row1 = _mm512_fmadd_ph(a1b, bvec, c_row1);
        c_row2 = _mm512_fmadd_ph(a2b, bvec, c_row2);
    }

    // ---- Store results back to C (row-major, ldc stride) ----
    _mm512_storeu_ph(c.add(0 * ldc), c_row0);
    _mm512_storeu_ph(c.add(1 * ldc), c_row1);
    _mm512_storeu_ph(c.add(2 * ldc), c_row2);
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::f16;
    use std::arch::is_x86_feature_detected;

    #[inline]
    fn f16v(v: f32) -> f16 {
        let h = half::f16::from_f32(v);          
        f16::from_bits(h.to_bits())         
    }
    #[inline]
    fn to_f32(x: f16) -> f32 {
        let h = half::f16::from_bits(x.to_bits());
        h.to_f32()                             
    }

    fn row_eq(a: &[f16], b: &[f16], tol: f32) -> bool {
        if a.len() != b.len() { return false; }
        for (x, y) in a.iter().zip(b.iter()) {
            if (to_f32(*x) - to_f32(*y)).abs() > tol { return false; }
        }
        true
    }

    #[test]
    fn test_broadcast_microkernel_3x32_k64_ones() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("Skipping: CPU lacks avx512fp16");
            return;
        }

        unsafe {
            let mr = 3usize;   
            let nr = 32usize;  
            let kc = 64usize;  

      
            let lda = kc;  
            let ldc = nr;    

            // 微核所需参数
            let param = MatMulParams {
                a_row: mr,               
                b_row: ldc,             
                column: lda,            
                a_row_step_macro: mr,    
                b_row_step_macro: nr,    
                column_step_macro: kc,   
                a_row_step_micro: mr,   
                b_row_step_micro: nr,    
            };

            let a: Vec<f16> = (0..mr*kc).map(|_| f16v(1.0)).collect();

            let b_panel: Vec<f16> = (0..kc*nr).map(|_| f16v(1.0)).collect();

            let mut c: Vec<f16> = (0..mr*ldc).map(|_| f16v(0.0)).collect();

            matmul_block(a.as_ptr(), b_panel.as_ptr(), c.as_mut_ptr(), &param);

            let expected: Vec<f16> = (0..mr*ldc).map(|_| f16v(kc as f32)).collect();

            assert!(row_eq(&c, &expected, 1e-3), "C != expected 64s");
        }
    }
}