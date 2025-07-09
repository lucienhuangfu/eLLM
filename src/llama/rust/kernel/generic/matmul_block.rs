use super::super::super::init::matmul_params::MatMulParams;
// use num_traits::Float;
use std::ops::{Add, Mul};

pub fn matmul_block<T>(a: *const T, b: *const T, c: *mut T, param: &MatMulParams)
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    for i in 0..param.a_row_step_micro {
        for j in 0..param.b_row_step_micro {
            for k in 0..param.column_step_macro {
                unsafe {
                    /*assert!(
                        i * column + k < param.a_row * param.column,
                        "a index out of bound"
                    );
                    assert!(
                        j * column + k < param.b_row * param.column,
                        "b index out of bound"
                    );
                    assert!(
                        i * c_column + j < param.a_row * param.b_row,
                        "c index out of bound"
                    );*/
                    let a_value = *a.offset((i * param.column + k) as isize);
                    let b_value = *b.offset((j * param.column + k) as isize);
                    let c_value = *c.offset((i * param.b_row + j) as isize);
                    *c.offset((i * param.b_row + j) as isize) = c_value + (a_value * b_value);
                }
            }
        }
    }
}

//test whether this organization can distribute the implementation of matmul_block to different platform
#[cfg(test)]
mod tests {
    use super::*;
    // use std::f16;
    /*
    // Helper function to compare two f16 arrays with a tolerance
    fn compare_f16_arrays(arr1: &[f16], arr2: &[f16], tolerance: f32) -> bool {
        if arr1.len() != arr2.len() {
            return false;
        }
        for (a, b) in arr1.iter().zip(arr2.iter()) {
            let diff = (f32::from(*a) - f32::from(*b)).abs();
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
        let column = 3;
        let param = MatMulParams{
            a_row,
            b_row,
            column,
            a_row_step_macro: 3,
            b_row_step_macro: 2,
            column_step_macro: 3,
            a_row_step_micro: 3,
            b_row_step_micro: 2,
        };

        let a = vec![1.0); a_row * column];
        let b = vec![1.0); b_row * column];
        let expected = vec![3.0); a_row * b_row];
        let mut c = vec![0.0); a_row * b_row];

        // prepare arguments for _matmul_block
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c.as_mut_ptr();

        // call _matmul_block
        matmul_block(a_ptr, b_ptr, c_ptr, &param);
        // print out c
        for i in 0..a_row {
            for j in 0..b_row {
                let index = i * b_row + j;
                println!("c[{}, {}] = {}", i, j, c[index]);
            }
        }
        // Compare c with the expected result using a tolerance
        let tolerance = 0.001;
        assert!(
            compare_f16_arrays(&c, &expected, tolerance),
            "Matrices are not equal"
        );
    } */

    //test for f32 avx2
    #[test]
    fn test_f32_general() {
        // a is     a 3 * 3 matrix, data type is f32, all elements are 1, stored in row-major order
        // b is     a 2 * 3 matrix, data type is f32, all elements are 1, stored in row-major order
        // expected a 3 * 2 matrix, data type is f32, all elements are 3, stored in row-major order
        // c is     a 3 * 2 matrix, data type is f32, all elements are 0, stored in row-major order
        let a_row = 3;
        let b_row = 2;
        let column = 3;
        let param = MatMulParams {
            a_row,
            b_row,
            column,
            a_row_step_macro: 3,
            b_row_step_macro: 2,
            column_step_macro: 3,
            a_row_step_micro: 3,
            b_row_step_micro: 2,
        };

        let a = vec![1.0f32; a_row * column];
        let b = vec![1.0f32; b_row * column];
        let expected = vec![3.0f32; a_row * b_row];
        let mut c = vec![0.0f32; a_row * b_row];

        // prepare arguments for _matmul_block
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c.as_mut_ptr();

        // call _matmul_block
        matmul_block(a_ptr, b_ptr, c_ptr, &param);
        // c should be equal to expected
        // Compare c with the expected result using a tolerance
        let tolerance = 0.001;
        assert!(
            c.iter()
                .zip(expected.iter())
                .all(|(a, b)| (a - b).abs() < tolerance),
            "Matrices are not equal"
        );
    }

    //test for f64 general
    #[test]
    fn test_f64_general() {
        // a is     a 3 * 3 matrix, data type is f64, all elements are 1, stored in row-major order
        // b is     a 2 * 3 matrix, data type is f64, all elements are 1, stored in row-major order
        // expected a 3 * 2 matrix, data type is f64, all elements are 3, stored in row-major order
        // c is     a 3 * 2 matrix, data type is f64, all elements are 0, stored in row-major order
        let a_row = 3;
        let b_row = 2;
        let column = 3;
        let param = MatMulParams {
            a_row,
            b_row,
            column,
            a_row_step_macro: 3,
            b_row_step_macro: 2,
            column_step_macro: 3,
            a_row_step_micro: 3,
            b_row_step_micro: 2,
        };

        let a = vec![1.0f64; a_row * column];
        let b = vec![1.0f64; b_row * column];
        let expected = vec![3.0f64; a_row * b_row];
        let mut c = vec![0.0f64; a_row * b_row];

        // prepare arguments for _matmul_block
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c.as_mut_ptr();

        // call _matmul_block
        matmul_block(a_ptr, b_ptr, c_ptr, &param);
        // c should be equal to expected
        // Compare c with the expected result using a tolerance
        let tolerance = 0.001;
        assert!(
            c.iter()
                .zip(expected.iter())
                .all(|(a, b)| (a - b).abs() < tolerance),
            "Matrices are not equal"
        );
    }
}
