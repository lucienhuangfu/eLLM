use std::ptr;
use std::ops::{Add, Mul, Div, Neg};
// use num_traits::Float;
use super::sigmoid::{self, Sigmoid};

/* 
#[inline]
fn sigmoid<T>(x: T) -> T 
where 
 T: Add<Output = T> + Mul<Output = T> + Div<Output = T> + Copy  + Neg<Output = T>,
{
    T::from(1.0) / (T::from(1.0) + T::exp(-x))
    // T::from(1.0)
}*/

pub fn silu<T>(input_ptr: *const T, output_ptr: *mut T, length: usize) 
where 
    T: Default+ Copy + Add<Output = T> + Mul<Output = T> + Div<Output = T> + Neg<Output = T> + Sigmoid<T>,
{
    (0..length).for_each(|x| {
        unsafe {
            let v = *input_ptr.add(x);
            *output_ptr.add(x) = v * v.sigmoid();
        }
    });
}

pub fn silu_multiply<T>(input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T, length: usize) 
where 
    T: Default+ Copy + Add<Output = T> + Mul<Output = T> + Div<Output = T>  + Neg<Output = T> + Sigmoid<T>,
{
    (0..length).for_each(|x| {
        unsafe {
            let v1 = input_ptr1.add(x).read();
            let sm = v1 * v1.sigmoid();
            let result = sm * input_ptr2.add(x).read();
            output_ptr.add(x).write(result);
        }
    });
}


#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;
    use super::*;

    #[test]
    fn test_silu() {
        let v1: [f32; 19] = [2.1671206951141357,
        1.4490455389022827,
        -2.002431631088257,
        0.5662149786949158,
        0.3909946382045746,
        0.9437483549118042,
        -0.37030690908432007,
        0.7542704939842224,
        0.5875813961029053,
        1.6026240587234497,
        2.2485475540161133,
        -0.6622593402862549,
        -0.0015666020335629582,
        -0.5069465041160583,
        -0.37254711985588074,
        0.4420417249202728,
        -0.9305257201194763,
        0.5145581364631653,
        0.6260590553283691];       
        let mut output = [0.0f32; 19];
        silu(v1.as_ptr(), output.as_mut_ptr(), v1.len());
        let result = [
        1.9444659948349,
        1.1735117435455322,
        -0.23818494379520416,
        0.36118248105049133,
        0.23323695361614227,
        0.6793630719184875,
        -0.15125809609889984,
        0.5129857659339905,
        0.3777032196521759,
        1.3339999914169312,
        2.033867835998535,
        -0.22532200813293457,
        -0.0007826874498277903,
        -0.1905660629272461,
        -0.15197153389453888,
        0.269090861082077,
        -0.2631694972515106,
        0.32204875349998474,
        0.4079371392726898];

        assert_ulps_eq!(output[..], result, max_ulps=4);      
    }
}