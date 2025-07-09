// use super::asmsimd::*;
use std::arch::x86_64::{_mm512_fmadd_ph, _mm512_load_ph, _mm512_reduce_add_ph, _mm512_setzero_ph};
use std::f16;

use crate::init::matmul_params::MatMulParams;
// this block implementation has not been tested yet
#[inline(always)]
pub unsafe fn matmul_block(a: *const f16, b: *const f16, c: *mut f16, param: &MatMulParams) {
    // use the a[a_row_l, a_row_l + 3][column_l, column_r] and b[b_row_l, b_row_l + 2][column_l, column_r] to update c[a_row_l, a_row_l + 3][b_row_l, b_row_l + 2]
    assert_eq!(param.a_row_step_micro, 3);
    assert_eq!(param.b_row_step_micro, 2);
    assert_eq!(param.column_step_macro % 64, 0);

    // total 16 512-bit-wide ZMM registers
    // 1 ZMM register can store 32 half precision floats
    // 6 ZMM registers used for store values of a[a_row_l, a_row_l + 3][k, k+64]
    // 4 ZMM registers used for store values of b[b_row_l, b_row_l + 2][k, k+64]
    // 6 ZMM registers used for store result of 6 cells of C
    let mut t00 = _mm512_setzero_ph();
    let mut t01 = _mm512_setzero_ph();
    let mut t10 = _mm512_setzero_ph();
    let mut t11 = _mm512_setzero_ph();
    let mut t20 = _mm512_setzero_ph();
    let mut t21 = _mm512_setzero_ph();

    for k in (0..param.column_step_macro).step_by(64) {
        // 6 ZMM registers used for store values of a[a_row_l, a_row_l + 3][k, k+64]
        let a00 = _mm512_load_ph(a);
        let a01 = _mm512_load_ph(a.add(32));
        let a10 = _mm512_load_ph(a.add(param.column));
        let a11 = _mm512_load_ph(a.add(param.column + 32));
        let a20 = _mm512_load_ph(a.add(2 * param.column));
        let a21 = _mm512_load_ph(a.add(2 * param.column + 32));

        // 4 ZMM registers used for store values of b[b_row_l, b_row_l + 2][k, k+64]
        let b00 = _mm512_load_ph(b);
        let b01 = _mm512_load_ph(b.add(32));
        let b10 = _mm512_load_ph(b.add(param.column));
        let b11 = _mm512_load_ph(b.add(param.column + 32));

        t00 = _mm512_fmadd_ph(a00, b00, t00);
        t00 = _mm512_fmadd_ph(a01, b01, t00);
        t01 = _mm512_fmadd_ph(a00, b10, t01);
        t01 = _mm512_fmadd_ph(a01, b11, t01);
        t10 = _mm512_fmadd_ph(a10, b00, t10);
        t10 = _mm512_fmadd_ph(a11, b01, t10);
        t11 = _mm512_fmadd_ph(a10, b10, t11);
        t11 = _mm512_fmadd_ph(a11, b11, t11);
        t20 = _mm512_fmadd_ph(a20, b00, t20);
        t20 = _mm512_fmadd_ph(a21, b01, t20);
        t21 = _mm512_fmadd_ph(a20, b10, t21);
        t21 = _mm512_fmadd_ph(a21, b11, t21);
    }
    //store the result
    *c += _mm512_reduce_add_ph(t00);
    *c.offset((1) as isize) += _mm512_reduce_add_ph(t01);
    *c.offset((param.b_row) as isize) += _mm512_reduce_add_ph(t10);
    *c.offset((param.b_row + 1) as isize) += _mm512_reduce_add_ph(t11);
    *c.offset((2 * param.b_row) as isize) += _mm512_reduce_add_ph(t20);
    *c.offset((2 * param.b_row + 1) as isize) += _mm512_reduce_add_ph(t21);
}

#[cfg(test)]
mod tests {
    // use num_traits::real::Real;
    /*
    use super::*;
    // use std::f16;
    // Helper function to compare two f16 arrays with a tolerance
    fn compare_f16_arrays(arr1: &[f16], arr2: &[f16], tolerance: f16) -> bool {
        if arr1.len() != arr2.len() {
            return false;
        }
        for (a, b) in arr1.iter().zip(arr2.iter()) {
            let diff = f16::abs(*a - *b);
            if diff > tolerance {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_f16_general() {
        // a is     a 3 * 3 matrix, data type is f16, all elements are 1, stored in row-major order
        // b is     a 2 * 3 matrix, data type is f16, all elements are 1, stored in row-major order
        // expected a 3 * 2 matrix, data type is f16, all elements are 3, stored in row-major order
        // c is     a 3 * 2 matrix, data type is f16, all elements are 0, stored in row-major order
        let a_row = 3;
        let b_row = 2;
        let column = 64;
        let a_row_step_macro = 3;
        let b_row_step_macro = 2;
        let column_step_macro = 64;
        let a_row_step_micro = 3;
        let b_row_step_micro = 2;
        let param = MatMulParams{
            a_row,
            b_row,
            column,
            a_row_step_macro,
            b_row_step_macro,
            column_step_macro,
            a_row_step_micro,
            b_row_step_micro,
        };

        let a: Vec<f16> = vec![1.0; a_row * column].into_iter().map(|x| x).collect();
        let b: Vec<f16>  = vec![1.0; b_row * column].into_iter().map(|x| x).collect();
        let mut c: Vec<f16>  = vec![0.0; a_row * b_row].into_iter().map(|x| x).collect();
        let mut expected: Vec<f16>  = vec![64.0; a_row * b_row].into_iter().map(|x| x).collect();
        unsafe {
            matmul_block(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), &param);
        }
        // print the result
        for i in 0..a_row {
            for j in 0..b_row {
                print!("{:?} ", c[i * b_row + j]);
            }
            println!();
        }
        assert!(compare_f16_arrays(&c, &expected, 1e-3));

    }*/
}
