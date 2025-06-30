use std::{arch::x86_64::*, ptr, slice};
use super::utils::hsum256_ps_avx;

#[inline]
unsafe fn fast_var(input_ptr: *const f32, length: usize, mean: f32) -> f32 {
    let rem = length % 8;
    let length2 = length - rem;    

    let mut chunks_sum = 0.0f32;
    if rem != length {
        unsafe {
            let mean_ = _mm256_set1_ps(mean);
            let mut sum1 = _mm256_setzero_ps();
            for ptr in (0..length2).step_by(8).map(|x| input_ptr.add(x)) {
                let mut y = _mm256_loadu_ps(ptr);
                y = _mm256_sub_ps(y, mean_);
                sum1 = _mm256_fmadd_ps(y, y, sum1);
            }
            chunks_sum = hsum256_ps_avx(sum1);
        }
    }
    unsafe {
        let mut remainder_sum = 0.0f32;
        if rem != 0 {
            remainder_sum = (length2..length).map(|x| {
                let mut y = *input_ptr.add(x);
                y = y - mean;
                y = y*y;
                y
            }).reduce(|acc, e| acc+e).unwrap();
        }
        (chunks_sum + remainder_sum) / (length as f32)
    }
}

#[inline]
unsafe fn fast_add_var(input_ptr1: *const f32, input_ptr2: *const f32, length: usize, mean: f32) -> f32 {
    let rem = length % 8;
    let length2 = length - rem; 

    let mut chunks_sum = 0.0f32;
    if rem != length {
        unsafe {
            let mean_ = _mm256_set1_ps(mean);
            let mut sum1 = _mm256_setzero_ps();
            for (ptr1, ptr2) in (0..length2).step_by(8).map(|x| (input_ptr1.add(x), input_ptr2.add(x))) {
                let x1 = _mm256_loadu_ps(ptr1);
                let x2 = _mm256_loadu_ps(ptr2);
                let mut y = _mm256_add_ps(x1, x2);
                y = _mm256_sub_ps(y, mean_);
                sum1 = _mm256_fmadd_ps(y, y, sum1);
            }
            chunks_sum = hsum256_ps_avx(sum1);
        }
    }
    unsafe {
        let mut remainder_sum = 0.0f32;
        if rem != 0 {
            remainder_sum = (length2..length).map(|x| {
                let y1 = *input_ptr1.add(x);
                let y2 = *input_ptr2.add(x);
                let mut y = y1 + y2;
                y = y - mean;
                y = y*y;
                y
            }).reduce(|acc, e| acc+e).unwrap();
        }
        (chunks_sum + remainder_sum) / (length as f32)
    }
}

#[inline]
unsafe fn fast_mean(input_ptr: *const f32, length: usize) -> f32 {
    let rem = length % 8;
    let length2 = length - rem; 
    let mut chunks_sum = 0.0f32;
    if rem != length {
        unsafe {
            let mut sum1 = _mm256_setzero_ps();
            for ptr in (0..length2).step_by(8).map(|x| input_ptr.add(x)) {
                let x = _mm256_loadu_ps(ptr);
                sum1 = _mm256_add_ps(sum1, x);
            }
            chunks_sum = hsum256_ps_avx(sum1);
        }
    }
    unsafe {
        let mut remainder_sum = 0.0f32;
        if rem != 0 {
            remainder_sum = (length2..length).map(|x| *input_ptr.add(x)).reduce(|acc, e| acc+e).unwrap();
        }
        (chunks_sum + remainder_sum) / (length as f32)
    }
}

pub fn _layer_norm(input_ptr: *const f32, output_ptr: *mut f32, length: usize, eps: f32, gamma: f32, beta: f32) {
    unsafe {
        let mean = fast_mean(input_ptr, length);
        let var = fast_var(input_ptr, length, mean);

        let rem = length % 8;
        let length2 = length - rem; 
        if rem != length {
            let mean_ = _mm256_set1_ps(mean);
            let var_ = _mm256_set1_ps(var);
            let eps_ = _mm256_set1_ps(eps);
            let gamma_ = _mm256_set1_ps(gamma);
            let beta_ = _mm256_set1_ps(beta);
            let tmp = _mm256_sqrt_ps(_mm256_add_ps(var_, eps_));
            for (ptr1, ptr2) in (0..length2).step_by(8).map(|x| (input_ptr.add(x), output_ptr.add(x))) {
                let x = _mm256_loadu_ps(ptr1);
                let mut y = _mm256_sub_ps(x, mean_);
                y = _mm256_div_ps(y, tmp);
                y = _mm256_fmadd_ps(y, gamma_, beta_);
                _mm256_storeu_ps(ptr2, y);
            }
        }
        if rem != 0 {
            let tmp = (var + eps).sqrt();
            for (ptr1, ptr2) in (length2..length).map(|x| (input_ptr.add(x), output_ptr.add(x))) {
                let mut y = *ptr1;
                y=(y-mean)/tmp;
                y=y*gamma+beta;
                ptr::write(ptr2, y);
            } 
        }
    }
}

pub fn _add_layer_norm(input_ptr1: *const f32, input_ptr2: *const f32, output_ptr: *mut f32, length: usize, eps: f32, gamma: f32, beta: f32) {
    unsafe {
        for (input1, input2, output) in (0..length).map(|x| (input_ptr1.add(x), input_ptr2.add(x), output_ptr.add(x))) {
            let y = *input1 + *input2;
            ptr::write(output, y);
        }
        _layer_norm(output_ptr, output_ptr, length, eps, gamma, beta);
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;

    use super::{fast_mean, fast_var, _layer_norm, fast_add_var, _add_layer_norm};

    #[test]
    fn test_mean_var() {
        let a: Vec<f32> = (1..19).map(|x| x as f32).collect();
        let mean = unsafe {fast_mean(a.as_ptr(), a.len())};
        assert_ulps_eq!(mean, 9.5f32 , max_ulps=4);
        let var = unsafe {fast_var(a.as_ptr(), a.len(), mean)};
        assert_ulps_eq!(var, 26.91666603088379f32, max_ulps=4);
    }

    #[test]
    fn test_layer_norm() {
        let a: Vec<f32> = (1..19).map(|x| x as f32).collect();
        let mut b =vec![0.0f32;a.len()];
        _layer_norm(a.as_ptr(), b.as_mut_ptr(), a.len(), 1.0e-5f32, 1.0, 0.0);
        let c: Vec<f32> = vec![-1.6383557319641113,
        -1.4456080198287964,
        -1.2528603076934814,
        -1.060112476348877,
        -0.8673648238182068,
        -0.6746170520782471,
        -0.48186933994293213,
        -0.2891216278076172,
        -0.09637391567230225,
        0.0963737964630127,
        0.28912150859832764,
        0.48186933994293213,
        0.6746169328689575,
        0.867364764213562,
        1.0601123571395874,
        1.252860188484192,
        1.4456080198287964,
        1.6383556127548218];
        assert_ulps_eq!(&b[..], &c[..], max_ulps=4);
    }

    #[test]
    fn test_add_var() {
        let v1: Vec<f32> = (1..19).map(|x| x as f32).collect();
        let v2: Vec<f32> = (19..37).map(|x| x as f32).collect();
        let mean1 = unsafe {fast_mean(v1.as_ptr(), v1.len())};
        let mean2 = unsafe {fast_mean(v2.as_ptr(), v2.len())};
        let mean = mean1 + mean2;
        let var = unsafe {fast_add_var(v1.as_ptr(), v2.as_ptr(), v1.len(), mean)};
        assert_ulps_eq!(var, 107.66666412353516, max_ulps=4);
    }

    #[test]
    fn test_add_layer_norm() {
        let v1: Vec<f32> = (1..19).map(|x| x as f32).collect();
        let v2: Vec<f32> = (19..37).map(|x| x as f32).collect();
        let mut v3 = vec![0.0f32; v1.len()];
        _add_layer_norm(v1.as_ptr(), v2.as_ptr(), v3.as_mut_ptr(), v1.len(), 1.0e-5f32, 1.0, 0.0);
        let result: [f32; 18] = [-1.63835608959198,
        -1.445608377456665,
        -1.2528605461120605,
        -1.060112714767456,
        -0.8673651218414307,
        -0.6746172904968262,
        -0.4818694591522217,
        -0.2891216278076172,
        -0.0963737964630127,
        0.0963737964630127,
        0.2891216278076172,
        0.4818694591522217,
        0.6746170520782471,
        0.8673651218414307,
        1.060112714767456,
        1.2528603076934814,
        1.445608377456665,
        1.6383559703826904];       
        assert_ulps_eq!(v3[..], result, max_ulps=4);
    }
}