use std::arch::x86_64::*;
use std::ptr;

pub fn _add(input_ptr1: *const f32, input_ptr2: *const f32, output_ptr: *mut f32, length: usize) {
    println!("add avx 256");
    unsafe {
        let rem = length % 8;
        let length2 = length - rem; 
        if rem != length {
            for (ptr1, ptr2, pout) in (0..length2).step_by(8).map(|x| (input_ptr1.add(x), input_ptr2.add(x), output_ptr.add(x))) {
                let x1 = _mm256_loadu_ps(ptr1);
                let x2 = _mm256_loadu_ps(ptr2);
                let y = _mm256_add_ps(x1, x2);
                _mm256_storeu_ps(pout, y);
            }
        }
        if rem != 0 {
            for (input1, input2, output) in (length2..length).map(|x| (input_ptr1.add(x), input_ptr2.add(x), output_ptr.add(x))) {
                let y = *input1 + *input2;
                ptr::write(output, y);
            } 
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;

    use super::*;

    #[test]
    fn test_add() {
        let v1 = [1.0; 19];
        let v2 = [2.0; 19];
        let mut v3 = [0.0; 19];
        let result = [3.0; 19];
        _add(v1.as_ptr(), v2.as_ptr(), v3.as_mut_ptr(), v1.len());
        assert_ulps_eq!(v3[..], result[..], max_ulps=4);
    }
}