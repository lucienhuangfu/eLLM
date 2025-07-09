use std::ptr;
use std::ops::{Add, Mul, AddAssign};
pub fn colmul<T>(ptr1: *const T, ptr2: *const T, ptr3: *mut T, row_size: usize, col_size: usize) 
where
    T: Copy + Add<Output = T> + Mul<Output = T> + AddAssign<T> +  From<f32>,
{
    // 列和矩阵相乘
    unsafe {
        for (ptr1, ptr2) in (0..row_size).map(|x| (ptr1.add(1), ptr2.add(col_size))) {
            for j in (0..col_size) {
                *ptr3.add(j) += *ptr1 * *ptr2.add(j);    
            } 
        } 
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;
    use super::*;

    #[test]
    fn test_mul() {
        let head_size = 128;
        let head_num  = 64;
        let batch_size = 16;
        let sequence_length = 256;
        let hidden_size = 8192;
        
        let shapes = vec![batch_size, hidden_size];
        // let length = shapes.iter().product();
        let data1: Vec<f32> = vec![1.0; sequence_length];
        let data2: Vec<f32> = vec![1.0; sequence_length * head_size];
        let mut output: Vec<f32> = vec![0.0; head_size];
        let result: Vec<f32> = vec![sequence_length as f32; head_size];
        colmul(data1.as_ptr(), data2.as_ptr(), output.as_mut_ptr(), sequence_length, head_size);
        assert_ulps_eq!(output[..], result[..], max_ulps=4);
    }
}