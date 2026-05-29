use super::from_number::FromNumber;
use std::f16;
use std::ops::{Add, Div, Mul};

pub trait Sqrt:
    Copy + Add<Output = Self> + Mul<Output = Self> + Div<Output = Self> + PartialOrd + FromNumber
{
    fn sqrt(self) -> Self;
}

impl Sqrt for f16 {
    fn sqrt(self) -> Self {
        f16::sqrt(self)
    }
}

impl Sqrt for f32 {
    fn sqrt(self) -> Self {
        self.sqrt()
    }
}

impl Sqrt for f64 {
    fn sqrt(self) -> Self {
        self.sqrt()
    }
}
