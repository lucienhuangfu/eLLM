use std::ptr;
use std::ops::{Add, Mul};
use super::dot_product::dot_product;


pub fn mat_vec_mul<T>(a_ptr: *const T, b_ptr: *const T, c_ptr: *mut T, col_size: usize, row_size: usize) 
where T: Copy + Add<Output = T> + Mul<Output = T> 
{
    unsafe {
        for i in 0..row_size {
            let offset = i*col_size;
            dot_product(a_ptr, b_ptr.add(offset), c_ptr.add(i), col_size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_mat_vec_mul() {
        let col_size = 64;
        let row_size = 32;

        let v1 = vec![1.0; col_size];

        let v2 = vec![1.0; row_size*col_size];
        let mut v3 = vec![0.0; row_size];
        let result = vec![64.0;row_size];
        mat_vec_mul(v1.as_ptr(), v2.as_ptr(), v3.as_mut_ptr(), col_size, row_size);
        assert_eq!(v3[..], result[..]);
    }
}