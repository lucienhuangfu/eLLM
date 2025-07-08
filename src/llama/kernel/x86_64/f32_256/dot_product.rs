use std::arch::x86_64::*;
use std::ptr;
use super::utils::hsum256_ps_avx;

pub fn _dot_product(input_ptr1: *const f32, input_ptr2: *const f32, length: usize, output_ptr: *mut f32) {
    // println!("dot product 256");
    unsafe {
        let rem = length % 8;
        let length2 = length - rem;
        let mut chunks_sum = 0.0f32;
        if rem != length {
            let mut chunks_simd = _mm256_setzero_ps();
            for (ptr1, ptr2) in (0..length2).step_by(8).map(|x| (input_ptr1.add(x), input_ptr2.add(x))) {
                let x1 = _mm256_loadu_ps(ptr1);
                let x2 = _mm256_loadu_ps(ptr2);
                chunks_simd = _mm256_fmadd_ps(x1, x2, chunks_simd);
            }
            chunks_sum = hsum256_ps_avx(chunks_simd);
        }
        let mut remainder_sum = *output_ptr;
        if rem != 0 {
            remainder_sum = (length2..length).map(|x| {
                let x1 = *input_ptr1.add(x);
                let x2 = *input_ptr2.add(x);
                x1 * x2
            }).fold(remainder_sum, |acc, e| acc + e);
        }
        ptr::write(output_ptr, chunks_sum + remainder_sum);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let x1: Vec<f32> = (1..19).map(|x| x as f32).collect();
        let x2: Vec<f32> = (19..37).map(|x| x as f32).collect();
        let mut result = 1.0f32;
        _dot_product(x1.as_ptr(), x2.as_ptr(), x1.len(), &mut result);
        assert_eq!(result, 5188.0);
    }
}