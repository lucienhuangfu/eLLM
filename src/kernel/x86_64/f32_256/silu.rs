use std::ptr;
use std::arch::x86_64::*;
use super::math::sigmoid256;

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn _silu(input_ptr: *const f32, output_ptr: *mut f32, length: usize) {
    unsafe {
        let rem = length % 8;
        let length2 = length - rem;

        if rem != length {
            for (ptr1, ptr2) in (0..length2).step_by(8).map(|x| (input_ptr.add(x), output_ptr.add(x))) {
                let x = _mm256_loadu_ps(ptr1);
                _mm256_storeu_ps(ptr2, _mm256_mul_ps(x, sigmoid256(x)));
            }
        }
        if rem != 0 {
            for (ptr1, ptr2) in (length2..length).map(|count| (input_ptr.add(count), output_ptr.add(count))) {
                let x = *ptr1;
                ptr::write(ptr2, x * sigmoid(x));
            }
        }
    }
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
        _silu(v1.as_ptr(), output.as_mut_ptr(), v1.len());
        let result = [1.9444659948349,
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