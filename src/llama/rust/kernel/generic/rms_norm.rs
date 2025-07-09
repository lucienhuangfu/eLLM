// use crate::kernel::generic::from_usize::FromUsize;
use crate::kernel::generic::sqrt::Sqrt;
use std::ops::{Add, Div, Mul};
use std::ptr;

pub fn rms_norm<T>(input_ptr: *const T, output_ptr: *mut T, length: usize, weight: *const T, eps: T)
where
    T: Sqrt,
{
    unsafe {
        let variance = {
            let sum = (0..length)
                .map(|x| *input_ptr.add(x))
                .map(|x| x * x)
                .reduce(|acc, e| acc + e)
                .unwrap();
            sum / T::from_usize(length)
        };
        let rrms = T::from_usize(1) / (variance.sqrt() + eps);
        for (vptr, gptr, optr) in
            (0..length).map(|x| (input_ptr.add(x), weight.add(x), output_ptr.add(x)))
        {
            ptr::write(optr, *vptr * rrms * *gptr);
        }
    }
}

pub fn add_rms_norm<T>(
    input_ptr1: *const T,
    input_ptr2: *const T,
    output_ptr: *mut T,
    length: usize,
    weight: *const T,
    eps: T,
) where
    T: Sqrt,
{
    unsafe {
        for (input1, input2, output) in
            (0..length).map(|x| (input_ptr1.add(x), input_ptr2.add(x), output_ptr.add(x)))
        {
            let y = *input1 + *input2;
            ptr::write(output, y);
        }
        rms_norm(output_ptr, output_ptr, length, weight, eps);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn test_rms_norm() {
        let v1: Vec<f32> = (1..19).map(|x| x as f32).collect();
        let weight = [1.0f32; 18];
        let mut output = [0.0f32; 18];
        rms_norm(
            v1.as_ptr(),
            output.as_mut_ptr(),
            v1.len(),
            weight.as_ptr(),
            1e-6,
        );
        let result = [
            0.09238425642251968,
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
            1.662916660308838,
        ];

        assert_ulps_eq!(output[..], result, max_ulps = 4);
    }

    #[test]
    fn test_add_rms_norm() {
        let v1: Vec<f32> = (0..18).map(|x| x as f32).collect();
        let v2 = [1.0f32; 18];
        let weight = [1.0f32; 18];
        let mut output = [0.0f32; 18];
        add_rms_norm(
            v1.as_ptr(),
            v2.as_ptr(),
            output.as_mut_ptr(),
            v1.len(),
            weight.as_ptr(),
            1e-6,
        );
        let result = [
            0.09238425642251968,
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
            1.662916660308838,
        ];

        assert_ulps_eq!(output[..], result, max_ulps = 4);
    }
}
