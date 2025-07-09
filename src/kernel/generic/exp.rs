use std::f16;

pub trait Exp {
    fn exp(self) -> Self;
}

impl Exp for f16 {
    fn exp(self) -> Self {
        f16::exp(self)
    }
}


impl Exp for f32 {
    fn exp(self) -> Self {
        f32::exp(self)
    }
}

impl Exp for f64 {
    fn exp(self) -> Self {
        f64::exp(self)
    }
}