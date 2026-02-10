use std::arch::x86_64::{
    _mm512_add_ph, _mm512_fmadd_ph, _mm512_loadu_ph, _mm512_mul_ph, _mm512_reduce_add_ph,
    _mm512_set1_ph, _mm512_setzero_ph, _mm512_storeu_ph,
};
use std::f16;

#[inline(always)]
fn sum_square(input_ptr: *const f16, length: usize) -> f16 {
    unsafe {
        let mut chunks_simd = _mm512_setzero_ph();
        for ptr in (0..length).step_by(32).map(|x| input_ptr.add(x)) {
            let x = _mm512_loadu_ph(ptr);
            chunks_simd = _mm512_fmadd_ph(x, x, chunks_simd);
        }
        _mm512_reduce_add_ph(chunks_simd)
    }
}

#[inline(always)]
fn add_sum_square(
    input_ptr1: *const f16,
    input_ptr2: *const f16,
    output_ptr: *mut f16,
    length: usize,
) -> f16 {
    unsafe {
        let mut chunks_simd = _mm512_setzero_ph();

        // 每 32 个元素做一次：z = x + y，写回 output，然后累加 z^2
        for x in (0..length).step_by(32) {
            let ptr1 = input_ptr1.add(x);
            let ptr2 = input_ptr2.add(x);
            let optr = output_ptr.add(x);

            let v1 = _mm512_loadu_ph(ptr1);
            let v2 = _mm512_loadu_ph(ptr2);
            let z = _mm512_add_ph(v1, v2);

            chunks_simd = _mm512_fmadd_ph(z, z, chunks_simd);
            _mm512_storeu_ph(optr, z);
        }

        _mm512_reduce_add_ph(chunks_simd)
    }
}

#[inline(always)]
pub fn norm(
    input_ptr: *const f16,
    output_ptr: *mut f16,
    length: usize,
    denominator: f16,
) {
    unsafe {
        let rrms_ = _mm512_set1_ph(denominator);
        for x in (0..length).step_by(32) {
            let vptr = input_ptr.add(x);
            let optr = output_ptr.add(x);

            let v = _mm512_loadu_ph(vptr);
            let result = _mm512_mul_ph(v, rrms_);
            _mm512_storeu_ph(optr, result);
        }
    }
}

#[inline(always)]
pub fn rms_norm(
    input_ptr: *const f16,
    output_ptr: *mut f16,
    length: usize,
    eps: f16,
) {
    let sum = sum_square(input_ptr, length);
    let square_root = f16::sqrt(sum / length as f16);
    let denominator = (square_root + eps).recip();
    norm(input_ptr, output_ptr, length, denominator);
}

pub fn add_rms_norm(
    input_ptr1: *const f16,
    input_ptr2: *const f16,
    output_ptr: *mut f16,
    length: usize,
    eps: f16,
) {
    let sum = add_sum_square(input_ptr1, input_ptr2, output_ptr, length);
    let square_root = f16::sqrt(sum / length as f16);
    let denominator = (square_root + eps).recip();
    norm(output_ptr, output_ptr, length, denominator);
}
#[cfg(test)]
mod tests {
    // use approx::assert_ulps_eq;
    use std::ptr;
    use std::slice;

    use super::*;
    use crate::kernel;
    use crate::memory::allocator::allocate_init;

    #[test]
    fn test_rms_norm() {
        let length = 64;
        // let v1: Vec<f16> = (0..length).into_iter().map(|x| x as f16).collect();
        // let weight: Vec<f16> = vec![1.0; length];
        // let mut output: Vec<f16> = vec![0.0; length];
        let v1 = allocate_init::<f16>(length, 0.0);
        for i in 0..length {
            unsafe {
                ptr::write(v1.wrapping_add(i), i as f16);
            }
        }
        let weight = allocate_init::<f16>(length, 1.0);
        let output = allocate_init::<f16>(length, 0.0);
        let output_slice = unsafe { slice::from_raw_parts(output, length) };

        rms_norm(v1, output, length,  1e-6);
        println!("{:?}", output);

        // let mut expected: Vec<f16> = vec![0.0; 64];
        let expected = allocate_init::<f16>(length, 0.0);
        let expected_slice = unsafe { slice::from_raw_parts(expected, length) };

        kernel::scalar::rms_norm::rms_norm(v1, expected, length, 1e-6);

        for j in 0..length {
            assert!(f16::abs(output_slice[j] - expected_slice[j]) < 1e-6);
        }
    }

    #[test]
    fn test_add_rms_norm() {
        let length = 64;
        // let v1: Vec<f16> = (0..length).into_iter().map(|x| x as f16).collect();
        // let v2: Vec<f16> = vec![1.0; length];
        // let weight: Vec<f16> = vec![1.0; length];
        // let mut output: Vec<f16> = vec![0.0; length];

        let v1 = allocate_init::<f16>(length, 0.0);
        for i in 0..length {
            unsafe {
                ptr::write(v1.wrapping_add(i), i as f16);
            }
        }
        let v2 = allocate_init::<f16>(length, 1.0);
        let weight = allocate_init::<f16>(length, 1.0);
        let output = allocate_init::<f16>(length, 0.0);
        let output_slice = unsafe { slice::from_raw_parts(output, length) };

        add_rms_norm(v1, v2, output, length,  1e-6);
        println!("{:?}", output);

        // let mut expected: Vec<f16> = vec![0.0; 64];
        let expected = allocate_init::<f16>(length, 0.0);
        let expected_slice = unsafe { slice::from_raw_parts(expected, length) };

        kernel::scalar::rms_norm::add_rms_norm(v1, v2, expected, length,  1e-6);

        for j in 0..length {
            assert!(f16::abs(output_slice[j] - expected_slice[j]) < 1e-6);
        }
    }
}

