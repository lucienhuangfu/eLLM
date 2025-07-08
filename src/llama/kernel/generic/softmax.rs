use std::ptr;
use num_traits::Float;
use std::ops::AddAssign;

pub fn scale_softmax<T: Float + AddAssign>(input_ptr: *const T, output_ptr: *mut T, length: usize, scale: T) {
    unsafe {
        let mut sum = T::zero();
        //let mut sum = T::from(0.0).unwrap();
        for (ptr1, ptr2) in (0..length).map(|x| (input_ptr.add(x), output_ptr.add(x))) {
            let x = (*ptr1 * scale).exp();
            sum += x;
            ptr::write(ptr2, x);
        }
        for ptr in (0..length).map(|x| output_ptr.add(x)) {
            *ptr = *ptr / sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;
    use super::*;

    #[test]
    fn test_scale_softmax() {
        let v1: Vec<f32> = (1..19).map(|x| x as f32).collect();
        let mut output = vec![0.0f32; v1.len()];
        scale_softmax(v1.as_ptr(), output.as_mut_ptr(), v1.len(), 0.65);
        let result: [f32; 18] = [7.5933926382276695e-06,
        1.4545462363457773e-05,
        2.7862415663548745e-05,
        5.337157563189976e-05,
        0.00010223548451904207,
        0.00019583618268370628,
        0.0003751322510652244,
        0.0007185811409726739,
        0.0013764717150479555,
        0.0026366880629211664,
        0.005050681531429291,
        0.009674787521362305,
        0.01853245310485363,
        0.035499654710292816,
        0.06800107657909393,
        0.13025879859924316,
        0.24951595067977905,
        0.4779582619667053];
        assert_ulps_eq!(output[..], result, max_ulps=4);
    }
}