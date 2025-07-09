use std::ops::{Add, Mul, Div};
use std::f16;

pub trait Sigmoid<T> {
    fn sigmoid(self) -> T;
}

impl<T> Sigmoid<T> for T
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Div<Output = T> + Default,
{
    default fn sigmoid(self) -> T {
        // let one: T = 1.0.into();
        // let exp_neg_self: T = (-self.into()).exp().into();
        // one / (one + exp_neg_self)
        T::default()
    }
}

impl Sigmoid<f16> for f16 {
    fn sigmoid(self) -> Self {
        1.0 / (1.0 + (-self).exp())
    }
}


impl Sigmoid<f32> for f32 {
    fn sigmoid(self) -> Self {
        1.0 / (1.0 + (-self).exp())
    }
}

impl Sigmoid<f64> for f64 {
    fn sigmoid(self) -> Self {
        1.0 / (1.0 + (-self).exp())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_f32() {


        let x: f32 = 0.0;
        assert!((x.sigmoid() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_f64() {
        let x: f64 = 0.0;
        assert!((x.sigmoid() - 0.5).abs() < 1e-6);
    }


}
