use std::ptr;
use std::arch::x86_64::*;
use super::math::exp256;
use super::utils::hsum256_ps_avx;

pub fn _softmax(input_ptr: *const f32, output_ptr: *mut f32, length: usize) {
    unsafe {
        let rem = length % 8;
        let length2 = length - rem; 

        let mut chunks_sum = 0.0f32;
        if rem != length {
            let mut m_sum = _mm256_setzero_ps();
            for (ptr1, ptr2) in (0..length2).step_by(8).map(|x| (input_ptr.add(x), output_ptr.add(x))) {
                let x = _mm256_loadu_ps(ptr1);
                let y = exp256(x);
                // println!("{:?}", y);
                _mm256_storeu_ps(ptr2, y);
                m_sum = _mm256_add_ps(m_sum, y);
            }
            chunks_sum = hsum256_ps_avx(m_sum);
        }
        let mut remainder_sum = 0.0f32;
        if rem != 0 {
            for (ptr1, ptr2) in (length2..length).map(|count| (input_ptr.add(count), output_ptr.add(count))) {
                let expx = (*ptr1).exp();
                remainder_sum += expx;
                ptr::write(ptr2, expx);
            }
        }

        let sum = chunks_sum + remainder_sum;
        if rem != length {
            let sum1 = _mm256_set1_ps(sum);
            for ptr in (0..length2).step_by(8).map(|x| output_ptr.add(x)) {
                let mut x = _mm256_loadu_ps(ptr);
                // x = _mm256_rcp_ps(x);
                // x = _mm256_rcp_ps(_mm256_mul_ps(x, sum1));
                x = _mm256_div_ps(x, sum1);
                _mm256_storeu_ps(ptr, x);
            }
        }
        if rem != 0 {
            for ptr in (length2..length).map(|count| output_ptr.add(count)) {
                ptr::write(ptr, *ptr/sum);
            }
        }
    }
}

pub fn _scale_softmax(input_ptr: *const f32, output_ptr: *mut f32, length: usize, scale: f32) {
    unsafe {
        let rem = length % 8;
        let length2 = length - rem;

        let mut chunks_sum = 0.0f32;
        if rem != length {
            let mut m_sum = _mm256_setzero_ps();
            let scale_ = _mm256_set1_ps(scale);
            for (ptr1, ptr2) in (0..length2).step_by(8).map(|x| (input_ptr.add(x), output_ptr.add(x))) {
                let mut x = _mm256_loadu_ps(ptr1);
                x = _mm256_mul_ps(x, scale_);
                let y = exp256(x);
                // println!("{:?}", y);
                _mm256_storeu_ps(ptr2, y);
                m_sum = _mm256_add_ps(m_sum, y);
            }
            chunks_sum = hsum256_ps_avx(m_sum);
        }
        let mut remainder_sum = 0.0f32;
        if rem != 0 {
            for (ptr1, ptr2) in (length2..length).map(|count| (input_ptr.add(count), output_ptr.add(count))) {
                let expx = (*ptr1*scale).exp();
                remainder_sum += expx;
                ptr::write(ptr2, expx);
            }
        }

        let sum = chunks_sum + remainder_sum;
        if rem != length {
            let sum1 = _mm256_set1_ps(sum);
            for ptr in (0..length2).step_by(8).map(|x| output_ptr.add(x)) {
                let mut x = _mm256_loadu_ps(ptr);
                x = _mm256_div_ps(x, sum1);
                _mm256_storeu_ps(ptr, x);
            }
        }
        if rem != 0 {
            for ptr in (length2..length).map(|count| output_ptr.add(count)) {
                ptr::write(ptr, *ptr/sum);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{slice, ptr};

    use approx::assert_ulps_eq;
    use rand::Rng;

    use super::*;

    #[test]
    fn test_softmax() {
        let length = 35*35;
        let mut input_ptr = vec![0.0f32; length].as_mut_ptr();
        let mut output_ptr = vec![0.0f32; length].as_mut_ptr();
        let mut result_ptr = vec![0.0f32; length].as_mut_ptr();
        let mut rng = rand::thread_rng();

        unsafe {
            for i in 0..length {
                let ptr = input_ptr.add(i);
                ptr::write(ptr, rng.gen::<f32>());
            }
        
            for i in (0..length).step_by(35) {
                _softmax(input_ptr.add(i), output_ptr.add(i), 35);
            }

            for i in (0..length).step_by(35) {
                let input_ptr = input_ptr.add(i);
                let result_ptr = result_ptr.add(i);
                let mut sum: f32 = 0.0;
                for (ptr1, ptr2) in (0..35).map(|count| (input_ptr.add(count), result_ptr.add(count))) {
                    let expx = (*ptr1).exp();
                    sum += expx;
                    ptr::write(ptr2, expx);
                }
                for ptr in (0..35).map(|count| result_ptr.add(count)) {
                    ptr::write(ptr, *ptr/sum);
                }
            }
        }

        let output_slice = unsafe {slice::from_raw_parts_mut(output_ptr, length)};
        let result_slice = unsafe {slice::from_raw_parts_mut(result_ptr, length)};
        assert_ulps_eq!(output_slice, result_slice, max_ulps=4);
    }

    #[test]
    fn test_scale_softmax() {
        let length = 35*35;
        let mut input_ptr = vec![0.0f32; length].as_mut_ptr();
        let mut output_ptr = vec![0.0f32; length].as_mut_ptr();
        let mut result_ptr = vec![0.0f32; length].as_mut_ptr();
        let mut rng = rand::thread_rng();
        let scale = 0.5f32;

        unsafe {
            for i in 0..length {
                let ptr = input_ptr.add(i);
                ptr::write(ptr, rng.gen::<f32>());
            }
        
            for i in (0..length).step_by(35) {
                _scale_softmax(input_ptr.add(i), output_ptr.add(i), 35, scale);
            }

            for i in (0..length).step_by(35) {
                let input_ptr = input_ptr.add(i);
                let result_ptr = result_ptr.add(i);
                let mut sum: f32 = 0.0;
                for (ptr1, ptr2) in (0..35).map(|count| (input_ptr.add(count), result_ptr.add(count))) {
                    let expx = (*ptr1*scale).exp();
                    sum += expx;
                    ptr::write(ptr2, expx);
                }
                for ptr in (0..35).map(|count| result_ptr.add(count)) {
                    ptr::write(ptr, *ptr/sum);
                }
            }
        }

        let output_slice = unsafe {slice::from_raw_parts_mut(output_ptr, length)};
        let result_slice = unsafe {slice::from_raw_parts_mut(result_ptr, length)};
        assert_ulps_eq!(output_slice, result_slice, max_ulps=4);
    }
}