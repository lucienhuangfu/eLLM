use std::ptr;
use std::ops::{Add, Mul};

// calculate the dot product of two vectors beginninng at input_ptr1 and input_ptr2, whose length is `length`, and store the result in `output_ptr`
pub fn dot_product<T>(input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T, length: usize)
where T: Copy + Add<Output = T> + Mul<Output = T> 
{
    unsafe {
        let sum = (0..length).map(|x| {
            let x1 = *input_ptr1.add(x);
            let x2 = *input_ptr2.add(x);
            x1 * x2
        }).fold(*output_ptr, |acc, e| acc + e);
        ptr::write(output_ptr, sum);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_dot_product() {
        let v1 = [1.0; 19];
        let v2 = [2.0; 19];
        let mut v3 = [0.0; 1];
        let result = [38.0];
        dot_product(v1.as_ptr(), v2.as_ptr(), v3.as_mut_ptr(), v1.len());
        assert_eq!(v3[..], result[..]);
    }
}