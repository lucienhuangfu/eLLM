use std::ptr;
use super::super::super::asmsimd::*;
use std::f16;

#[inline(always)]
unsafe fn fast_var(input_ptr: *const f16, length: usize, mean: f16) -> f16 {
    let rem = length % 32;
    let length2 = length - rem;    

    let mut chunks_sum = f16::ZERO;
    if rem != length {
        unsafe {
            let mean_ = _mm512_set1_ph(mean);
            let mut sum1 = _mm512_setzero_ph();
            for ptr in (0..length2).step_by(32).map(|x| input_ptr.add(x)) {
                let mut y = _mm512_loadu_ph(ptr);
                y = _mm512_sub_ph(y, mean_);
                sum1 = _mm512_fmadd_ph(y, y, sum1);
            }
            chunks_sum = _mm512_reduce_add_ph(sum1);
        }
    }
    unsafe {
        let mut remainder_sum = f16::ZERO;
        if rem != 0 {
            remainder_sum = (length2..length).map(|x| {
                let mut y = *input_ptr.add(x);
                y = y.ss_sub(mean);
                y = y.ss_mul(y);
                y
            }).reduce(|acc, e| acc.ss_add(e)).unwrap();
        }
        (chunks_sum + remainder_sum).ss_div_by_rcp(length as f32))
    }
}

#[inline]
unsafe fn fast_mean(input_ptr: *const f16, length: usize) -> f16 {
    let rem = length % 32;
    let length2 = length - rem; 
    let mut chunks_sum = f16::ZERO;
    if rem != length {
        unsafe {
            let mut sum1 = _mm512_setzero_ph();
            for ptr in (0..length2).step_by(32).map(|x| input_ptr.add(x)) {
                let x = _mm512_loadu_ph(ptr);
                sum1 = _mm512_add_ph(sum1, x);
            }
            chunks_sum = _mm512_reduce_add_ph(sum1);
        }
    }
    unsafe {
        let mut remainder_sum = f16::ZERO;
        if rem != 0 {
            remainder_sum = (length2..length).map(|x| *input_ptr.add(x)).reduce(|acc, e| acc+e).unwrap();
        }
        (chunks_sum + remainder_sum).ss_div_by_rcp(length as f32))
    }
}

pub fn _layer_norm(input_ptr: *const f16, output_ptr: *mut f16, length: usize, eps: f16, gamma: f16, beta: f16) {
    unsafe {
        let mean = fast_mean(input_ptr, length);
        let var = fast_var(input_ptr, length, mean);

        let rem = length % 32;
        let length2 = length - rem;
        if rem != length {
            let mean_ = _mm512_set1_ph(mean);
            let var_ = _mm512_set1_ph(var);
            let eps_ = _mm512_set1_ph(eps);
            let gamma_ = _mm512_set1_ph(gamma);
            let beta_ = _mm512_set1_ph(beta);
            let tmp = _mm512_sqrt_ph(_mm512_add_ph(var_, eps_));
            for (ptr1, ptr2) in (0..length2).step_by(32).map(|x| (input_ptr.add(x), output_ptr.add(x))) {
                let x = _mm512_loadu_ph(ptr1);
                let mut y = _mm512_sub_ph(x, mean_);
                y = _mm512_divbyrcp_ph(y, tmp);
                y = _mm512_fmadd_ph(y, gamma_, beta_);
                _mm512_storeu_ph(ptr2, y);
            }
        }
        let tmp = (var.to_f32() + eps.to_f32()).sqrt());
        for (ptr1, ptr2) in (length2..length).map(|x| (input_ptr.add(x), output_ptr.add(x))) {
            let mut y = *ptr1;
            y=(y.ss_sub(mean)).ss_div_by_rcp(tmp);
            y=y.ss_mul(gamma).ss_add(beta);
            ptr::write(ptr2, y);
        }
    }
}

pub fn _add_layer_norm(input_ptr1: *const f16, input_ptr2: *const f16, output_ptr: *mut f16, length: usize, eps: f16, gamma: f16, beta: f16) {
    unsafe {
        let rem = length % 32;
        let length2 = length - rem; 
        if rem != length {
            for (ptr1, ptr2, pout) in (0..length2).step_by(32).map(|x| (input_ptr1.add(x), input_ptr2.add(x), output_ptr.add(x))) {
                let x1 = _mm512_loadu_ph(ptr1);
                let x2 = _mm512_loadu_ph(ptr2);
                let y = _mm512_add_ph(x1, x2);
                _mm512_storeu_ph(pout, y);
            }
        }
        if rem != 0 {
            for (input1, input2, output) in (length2..length).map(|x| (input_ptr1.add(x), input_ptr2.add(x), output_ptr.add(x))) {
                let x1 = *input1;
                let x2 = *input2;
                let y = x1.ss_add(x2);
                ptr::write(output, y);
            } 
        }
        _layer_norm(output_ptr, output_ptr, length, eps, gamma, beta);
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;
    use std::f16;
    use super::{fast_mean, fast_var, _layer_norm, _add_layer_norm};

    #[test]
    fn test_mean_var() {
        let a: Vec<f16> = (1..37).map(|x| x as f32)).collect();
        let mean = unsafe {fast_mean(a.as_ptr(), a.len())};
        // assert_ulps_eq!(mean, 9.5f32 , max_ulps=4);
        let var = unsafe {fast_var(a.as_ptr(), a.len(), mean)};
        // assert_ulps_eq!(var, 26.91666603088379f32, max_ulps=4);
        println!("{:?}", mean);
        println!("{:?}", var);
    }

    #[test]
    fn test_layer_norm() {
        let a: Vec<f16> = (1..37).map(|x| x as f32)).collect();
        let mut b =vec![f16::ZERO;a.len()];
        _layer_norm(a.as_ptr(), b.as_mut_ptr(), a.len(), 1.0e-5f32), f16::ONE, f16::ZERO);
        println!("{:?}", b);
        // let c: Vec<f32> = vec![-1.6383557319641113,
        // -1.4456080198287964,
        // -1.2528603076934814,
        // -1.060112476348877,
        // -0.8673648238182068,
        // -0.6746170520782471,
        // -0.48186933994293213,
        // -0.2891216278076172,
        // -0.09637391567230225,
        // 0.0963737964630127,
        // 0.28912150859832764,
        // 0.48186933994293213,
        // 0.6746169328689575,
        // 0.867364764213562,
        // 1.0601123571395874,
        // 1.252860188484192,
        // 1.4456080198287964,
        // 1.6383556127548218];
        // assert_ulps_eq!(&b[..], &c[..], max_ulps=4);
    }

    #[test]
    fn test_add_layernorm() {
        let v1: Vec<f16> = (1..37).map(|x| x as f32)).collect();
        let v2: Vec<f16> = (38..74).map(|x| x as f32)).collect();
        let mut v3 = vec![f16::ZERO; v1.len()];
        _add_layer_norm(v1.as_ptr(), v2.as_ptr(), v3.as_mut_ptr(), v1.len(), 1.0e-5f32), f16::ONE, f16::ZERO);
        println!("{:?}", v3);
        // let result: [f32; 18] = [-1.63835608959198,
        // -1.445608377456665,
        // -1.2528605461120605,
        // -1.060112714767456,
        // -0.8673651218414307,
        // -0.6746172904968262,
        // -0.4818694591522217,
        // -0.2891216278076172,
        // -0.0963737964630127,
        // 0.0963737964630127,
        // 0.2891216278076172,
        // 0.4818694591522217,
        // 0.6746170520782471,
        // 0.8673651218414307,
        // 1.060112714767456,
        // 1.2528603076934814,
        // 1.445608377456665,
        // 1.6383559703826904];       
        // assert_ulps_eq!(v3[..], result, max_ulps=4);
    }
}