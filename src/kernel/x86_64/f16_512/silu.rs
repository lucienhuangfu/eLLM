use std::arch::x86_64::{_mm512_load_ph, _mm512_mul_ph, _mm512_store_ph};
use std::f16;

use super::activation::sigmoid512;
#[inline(always)]
pub fn silu(input_ptr: *const f16, output_ptr: *mut f16, length: usize) {
    unsafe {
        for (ptr1, ptr2) in (0..length)
            .step_by(32)
            .map(|x| (input_ptr.add(x), output_ptr.add(x)))
        {
            let x = _mm512_load_ph(ptr1);
            _mm512_store_ph(ptr2, _mm512_mul_ph(x, sigmoid512(x)));
        }
    }
}
#[inline(always)]
pub fn silu_multiply(
    input_ptr1: *const f16,
    input_ptr2: *const f16,
    output_ptr: *mut f16,
    length: usize,
) {
    unsafe {
        for (ptr1, ptr2, optr) in (0..length)
            .step_by(32)
            .map(|x| (input_ptr1.add(x), input_ptr2.add(x), output_ptr.add(x)))
        {
            let v1 = _mm512_load_ph(ptr1);
            let silu_value = _mm512_mul_ph(v1, sigmoid512(v1));
            let v2 = _mm512_load_ph(ptr2);
            let result = _mm512_mul_ph(v2, silu_value);
            _mm512_store_ph(optr, result);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::allocator::allocate_init;
    use approx::assert_ulps_eq;
    use std::ptr;
    use std::slice;

    #[test]
    fn test_silu() {
        let v1: Vec<f16> = vec![
            2.1671206951141357,
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
            0.6260590553283691,
            2.1671206951141357,
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
            // -0.5069465041160583,
            // -0.37254711985588074,
            // 0.4420417249202728,
            // -0.9305257201194763,
            // 0.5145581364631653,
            // 0.6260590553283691,
        ];

        println!("len{}", v1.len());
        let v1_ptr = allocate_init::<f16>(v1.len(), 0.0);
        for i in 0..v1.len() {
            unsafe {
                ptr::write(v1_ptr.wrapping_add(i), v1[i]);
            }
        }

        // let mut output: Vec<f16> = vec![0.0; v1.len()];
        let output = allocate_init::<f16>(v1.len(), 0.0);
        let output_slice = unsafe { slice::from_raw_parts(output, v1.len()) };
        silu(v1_ptr, output, v1.len());
        // println!("{:?}", output);
        let expected: Vec<f16> = vec![
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
            0.4079371392726898,
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
        ];

        // println!("expected len {}", expected.len());
        for i in 0..v1.len() {
            println!("{} {}", output_slice[i] as f32, expected[i] as f32);
            // assert!((output[i] - expected[i]).abs() < 1e-6);
        }

        // assert_ulps_eq!(output[..], result, max_ulps=4);
    }
}
