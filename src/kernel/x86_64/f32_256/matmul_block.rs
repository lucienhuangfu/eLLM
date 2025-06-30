use std::arch::x86_64::*;
// use super::super::super::x86_64::asmsimd::*;
use super::super::super::super::definition::matmul_params::MatMulParams;


// #[inline]
// pub unsafe fn _matmul_block(
//     a: *const f32,
//     b: *const f32,
//     c: *mut f32,
//     a_row_l: usize,
//     a_row_r: usize,
//     b_row_l: usize,
//     b_row_r: usize,
//     column_l: usize,
//     column_r: usize,
//     column: usize,
//     c_column: usize,
// ) {
//     //println!("This is the SIMD implementation for inner kernel in f32 256-bit wide AVX.");
//     for i in a_row_l..a_row_r {
//         for j in b_row_l..b_row_r {
//             for k in column_l..column_r {
//                 *c.offset((i * c_column + j) as isize) +=
//                     *a.offset((i * column + k) as isize) * *b.offset((j * column + k) as isize);
//             }
//         }
//     }
// }


#[inline]
pub unsafe fn _matmul_block(a: *const f32, b: *const f32, c: *mut f32, param: &MatMulParams) {
    assert_eq!(param.a_row_step_micro, 3);
    assert_eq!(param.b_row_step_micro, 2);
    // assert_eq!(param.column_step_macro % 8, 0); // 8 floats fit into one AVX2 register (__m256)

    // Use AVX2 registers, each __m256 register can hold 8 f32 values (256 bits)
    let mut t00 = _mm256_setzero_ps();
    let mut t01 = _mm256_setzero_ps();
    let mut t10 = _mm256_setzero_ps();
    let mut t11 = _mm256_setzero_ps();
    let mut t20 = _mm256_setzero_ps();
    let mut t21 = _mm256_setzero_ps();

    for k in (0..param.column_step_macro).step_by(8) {
        // Load 8 f32 values from matrix `a`
        let a00 = _mm256_loadu_ps(a.add(k));
        let a10 = _mm256_loadu_ps(a.add(k + param.column));
        let a20 = _mm256_loadu_ps(a.add(k + 2 * param.column));

        // Load 8 f32 values from matrix `b`
        let b00 = _mm256_loadu_ps(b.add(k));
        let b10 = _mm256_loadu_ps(b.add(k + param.column));

        // Compute the dot products and accumulate in the result registers
        t00 = _mm256_fmadd_ps(a00, b00, t00); // t00 = a00 * b00 + t00
        t01 = _mm256_fmadd_ps(a00, b10, t01); // t01 = a00 * b10 + t01
        t10 = _mm256_fmadd_ps(a10, b00, t10); // t10 = a10 * b00 + t10
        t11 = _mm256_fmadd_ps(a10, b10, t11); // t11 = a10 * b10 + t11
        t20 = _mm256_fmadd_ps(a20, b00, t20); // t20 = a20 * b00 + t20
        t21 = _mm256_fmadd_ps(a20, b10, t21); // t21 = a20 * b10 + t21
    }

    // Reduce each register and accumulate the final result in c
    *c.offset(0) += hsum256_ps_avx(t00); // Sum of first row, first column
    *c.offset(1) += hsum256_ps_avx(t01); // Sum of first row, second column
    *c.offset(param.b_row as isize) += hsum256_ps_avx(t10); // Sum of second row, first column
    *c.offset(param.b_row as isize + 1) += hsum256_ps_avx(t11); // Sum of second row, second column
    *c.offset(2 * param.b_row as isize) += hsum256_ps_avx(t20); // Sum of third row, first column
    *c.offset(2 * param.b_row as isize + 1) += hsum256_ps_avx(t21); // Sum of third row, second column
}

// Horizontal sum for AVX2 __m256
#[inline]
unsafe fn hsum256_ps_avx(v: __m256) -> f32 {
    // Use AVX2 shuffle and add instructions to reduce the 8 elements in the __m256 register
    let sum1 = _mm256_hadd_ps(v, v); // [a0 + a1, a2 + a3, a4 + a5, a6 + a7, a0 + a1, ...]
    let sum2 = _mm256_hadd_ps(sum1, sum1); // [a0 + a1 + a2 + a3, a4 + a5 + a6 + a7, ...]
    let sum3 = _mm256_extractf128_ps(sum2, 1); // Extract the upper 128 bits
    let result = _mm_add_ps(_mm256_castps256_ps128(sum2), sum3); // Add lower and upper 128 bits
    _mm_cvtss_f32(result) // Extract the result as f32
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::f16;
    fn compare_f32_arrays(arr1: &[f32], arr2: &[f32], tolerance: f32) -> bool {
        if arr1.len() != arr2.len() {
            return false;
        }
        
        for (a, b) in arr1.iter().zip(arr2.iter()) {
            let diff = (a - b).abs();
            if diff > tolerance {
                return false;
            }
        }
        
        true
    }

    #[test]
    fn test_f32_general() {
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

        let a : Vec<f32> = vec![1.0; a_row * column];
        let b: Vec<f32> = vec![1.0;  b_row * column];
        let mut c: Vec<f32> = vec![0.0; a_row * b_row];
        let mut expected: Vec<f32> = vec![64.0;  a_row * b_row];
        unsafe {
            _matmul_block(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), &param);
        }
        // print the result
        for i in 0..a_row {
            for j in 0..b_row {
                print!("{:?} ", c[i * b_row + j]);
            }
            println!();
        }
        // print!("c:{:?}",&c);
        assert!(compare_f32_arrays(&c, &expected, 1e-3));
        

    }
}  