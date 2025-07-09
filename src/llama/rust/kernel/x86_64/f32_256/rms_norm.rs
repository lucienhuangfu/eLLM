use std::arch::x86_64::*;
use std::ptr;
use super::utils::hsum256_ps_avx;

fn rms(input_ptr: *const f32, length: usize) -> f32 {
    unsafe {
        let rem = length % 8;
        let length2 = length - rem;
        let mut chunks_sum = 0.0f32;
        if rem != length {
            let mut chunks_simd = _mm256_setzero_ps();
            for ptr in (0..length2).step_by(8).map(|x| input_ptr.add(x)) {
                let x = _mm256_loadu_ps(ptr);
                chunks_simd = _mm256_fmadd_ps(x, x, chunks_simd);
            }
            chunks_sum = hsum256_ps_avx(chunks_simd);
        }
        let mut remainder_sum = 0.0f32;
        if rem != 0 {
            remainder_sum = (length2..length).map(|x| *input_ptr.add(x)).map(|x| x*x).reduce(|acc, e| acc + e).unwrap();
        }
        ((chunks_sum + remainder_sum) / (length as f32)).sqrt()
    }
}

pub fn _rms_norm(input_ptr: *const f32, weight: *const f32, output_ptr: *mut f32, length: usize, eps: f32) {
    unsafe {
        let rem = length % 8;
        let length2 = length - rem;
        let rrms = 1.0 / (rms(input_ptr, length) + eps);
        if rem != length {
            let rrms_ = _mm256_set1_ps(rrms);
            for (vptr, gptr, optr) in (0..length2).step_by(8).map(|x| (input_ptr.add(x), weight.add(x), output_ptr.add(x))) {
                let mut x = _mm256_loadu_ps(vptr);
                let y = _mm256_loadu_ps(gptr);
                x = _mm256_mul_ps(x, y);
                x = _mm256_mul_ps(x, rrms_);
                _mm256_storeu_ps(optr, x);
            }
        }
        if rem != 0 {
            for(vptr, gptr, optr) in (length2..length).map(|x| (input_ptr.add(x), weight.add(x), output_ptr.add(x))) {
                ptr::write(optr, *vptr * rrms * *gptr);
            }
        }
    }
}

pub fn _add_rms_norm(input_ptr1: *const f32, input_ptr2: *const f32, weight: *const f32, output_ptr: *mut f32, length: usize, eps: f32) {
    unsafe {
        for (input1, input2, output) in (0..length).map(|x| (input_ptr1.add(x), input_ptr2.add(x), output_ptr.add(x))) {
            let y = *input1 + *input2;
            ptr::write(output, y);
        }
        _rms_norm(output_ptr, weight, output_ptr, length, eps);
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;

    use super::*;

    #[test]
    fn test_rms_norm() {
        let v1: Vec<f32> = (1..19).map(|x| x as f32).collect();
        let g = [1.0f32; 18];
        let mut output = [0.0f32; 18];
        _rms_norm(v1.as_ptr(), g.as_ptr(), output.as_mut_ptr(), v1.len(), 1e-6);
        let result = [0.09238425642251968,
        0.18476851284503937,
        0.27715277671813965,
        0.36953702569007874,
        0.4619212746620178,
        0.5543055534362793,
        0.646689772605896,
        0.7390740513801575,
        0.831458330154419,
        0.9238425493240356,
        1.0162267684936523,
        1.1086111068725586,
        1.2009953260421753,
        1.293379545211792,
        1.3857638835906982,
        1.478148102760315,
        1.5705323219299316,
        1.662916660308838];

        assert_ulps_eq!(output[..], result, max_ulps=4);      
    }
    
    #[test]
    fn test_add_rms_norm() {
        let v1: Vec<f32> = (0..18).map(|x| x as f32).collect();
        let v2 = [1.0f32; 18];
        let weight = [1.0f32; 18];
        let mut output = [0.0f32; 18];
        _add_rms_norm(v1.as_ptr(), v2.as_ptr(), weight.as_ptr(), output.as_mut_ptr(), v1.len(), 1e-6);
        let result = [0.09238425642251968,
        0.18476851284503937,
        0.27715277671813965,
        0.36953702569007874,
        0.4619212746620178,
        0.5543055534362793,
        0.646689772605896,
        0.7390740513801575,
        0.831458330154419,
        0.9238425493240356,
        1.0162267684936523,
        1.1086111068725586,
        1.2009953260421753,
        1.293379545211792,
        1.3857638835906982,
        1.478148102760315,
        1.5705323219299316,
        1.662916660308838];

        assert_ulps_eq!(output[..], result, max_ulps=4);
    }
}