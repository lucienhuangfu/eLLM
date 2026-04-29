use std::ops::{Add, Div, Mul, Neg};

use crate::common::num_traits::Sigmoid;

pub fn sigmoid<T>(input_ptr: *const T, output_ptr: *mut T, length: usize)
where
    T: Default
        + Copy
        + Add<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>
        + Sigmoid,
{
    (0..length).for_each(|index| unsafe {
        let value = *input_ptr.add(index);
        *output_ptr.add(index) = value.sigmoid();
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn test_sigmoid() {
        let input = [-2.0f32, 0.0, 2.0];
        let mut output = [0.0f32; 3];

        sigmoid(input.as_ptr(), output.as_mut_ptr(), input.len());

        assert_ulps_eq!(output[0], 1.0 / (1.0 + (2.0f32).exp()), max_ulps = 4);
        assert_ulps_eq!(output[1], 0.5, max_ulps = 4);
        assert_ulps_eq!(output[2], 1.0 / (1.0 + (-2.0f32).exp()), max_ulps = 4);
    }
}
