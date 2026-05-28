use std::f16;

pub trait Sigmoid {
    fn sigmoid(self) -> Self;
}

impl Sigmoid for f16 {
    fn sigmoid(self) -> Self {
        1.0 / (1.0 + (-self).exp())
    }
}

impl Sigmoid for f32 {
    fn sigmoid(self) -> Self {
        1.0 / (1.0 + (-self).exp())
    }
}

impl Sigmoid for f64 {
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
