use std::ops::{Add, Div, Mul};
use crate::kernel::generic::from_usize::FromUsize;

pub trait Powf:
    Copy + Add<Output = Self> + Mul<Output = Self> + Div<Output = Self> + PartialOrd + FromUsize
{
    fn powf(self) -> Self;  
}

impl Powf for f16 {
    fn powf(self) -> Self {
        self.powf(num_traits::Float::from_f64(2.0).unwrap()) // Example: raising to the power of 2
        // .powf()
    }
}

impl Powf for f32 {
    fn powf(self) -> Self {
        self.powf()
    }
}

impl Powf for f64 {
    fn powf(self) -> Self {
        self.powf()
    }
}