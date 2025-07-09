use std::ops::{Add, Sub, Mul, Div};
use crate::kernel::generic::{neg_infinity::NegInfinity, exp::Exp};



/// Computes the dot product of two vectors.
/// 
/// # Arguments
/// 
/// * `a` - Pointer to the first vector.
/// * `b` - Pointer to the second vector.
/// * `length` - Length of the vectors.
/// 
/// # Returns
/// 
/// The dot product of the two vectors.
pub fn dot_product<T>(a: *const T, b: *const T, length: usize) -> T
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>  ,
{
    let mut result = T::default();
    (0..length).for_each(|i| {
        unsafe {
            result = result + *a.add(i) * *b.add(i);
        }
    });
    result
}

/// Multiplies a scalar with a vector and accumulates the result into another vector.
/// 
/// # Arguments
/// 
/// * `scalar1` - The scalar to multiply with the output vector.
/// * `o` - Pointer to the output vector.
/// * `scalar2` - The scalar to multiply with the input vector.
/// * `v` - Pointer to the input vector.
/// * `length` - Length of the vectors.
unsafe fn scalar_vector_mul_and_acc<T>(
    scalar1: T, o: *mut T, scalar2: T, v: *const T, length: usize
) where
    T: Copy + Add<Output = T> + Mul<Output = T> ,
{
    (0..length).for_each(|i| {
        *o.add(i) = scalar1* (*o.add(i))  + scalar2 * *v.add(i);
    });
}

/// Performs the flash attention mechanism.
/// 
/// # Arguments
/// 
/// * `q` - Pointer to the query vector.
/// * `K` - Pointer to the key matrix.
/// * `V` - Pointer to the value matrix.
/// * `o` - Pointer to the output vector.
/// * `length` - Length of the query vector.
/// * `stride` - Stride of the key and value matrices.
/// * `position` - Number of positions to process.
pub fn flash_attention<T>(
    q: *const T, K: *const T, V: *const T, o: *mut T, inverse_sqrt_head: T, length: usize, stride: usize, position: usize
)
where T: Copy + Default + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + PartialOrd + NegInfinity + Exp,
{
    let mut m_i_1 = T::neg_infinity();
    let mut d_i_1 = T::default();
    let mut offset = 0;
    for i in 0..=position {
        unsafe {
            //  dot product
            let x_i = dot_product(q, K.add(offset), length);
            let scale_x_i =  x_i * inverse_sqrt_head;
            let m_i = if  scale_x_i > m_i_1 { scale_x_i } else { m_i_1 };

            let update_exp = d_i_1 * (m_i_1 - m_i).exp();
            let add_exp = (x_i - m_i).exp();
            // let update_exp = d_i_1 * (m_i_1 - m_i);
            // let add_exp = (x_i - m_i);

            let d_i = update_exp + add_exp;
            let u = update_exp / d_i;
            let a = add_exp / d_i;
            scalar_vector_mul_and_acc(u, o, a, V.add(offset), length);
            m_i_1 = m_i;
            d_i_1 = d_i;
        }
        offset += stride;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0];
        let b: Vec<f32>  = vec![4.0, 5.0, 6.0];
        let result = dot_product(a.as_ptr(), b.as_ptr(), 3);
        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_scalar_vector_mul_and_acc() {
        let scalar1: f32 = 1.0;
        let scalar2: f32 = 2.0;
        let mut o = vec![1.0;128];
        let v = vec![2.0;128];
        unsafe {
            scalar_vector_mul_and_acc(scalar1, o.as_mut_ptr(), scalar2, v.as_ptr(), v.len());
        }
        let expected = vec![5.0;128];
        for i in 0..v.len() {
            assert!((o[i] - expected[i]).abs() < 1e-6);
        }
    }

    
    #[test]
    fn test_flash_attention() {
        let length: usize = 128;
        let row_size = 8;

        let q: Vec<f32> = vec![1.0;length];
        let K: Vec<f32> = vec![1.0;row_size*length];
        let V: Vec<f32> = vec![1.0;row_size*length];
        let mut o: Vec<f32> = vec![1.0;length];

  

        flash_attention(
            q.as_ptr(),
            K.as_ptr(),
            V.as_ptr(),
            o.as_mut_ptr(),
            1.0,
            length,
            length,
            row_size - 1,
        );
        println!("Result: {:?}", o);
    
        let expected = vec![1.0;length]; // Adjust based on actual expected output
        for i in 0..length {
            assert!((o[i] - expected[i]).abs() < 1e-6);
        } 
    } 
}
