use std::ops::{Add, Div, Mul};
use crate::kernel::generic::from_usize::FromUsize;

pub trait Powf:
    Copy + Add<Output = Self> + Mul<Output = Self> + Div<Output = Self> + PartialOrd + FromUsize
{
    fn powf(self, power: Self) -> Self;  // 添加 power 参数
}

impl Powf for f16 {
    fn powf(self, power: Self) -> Self {
        self.powf(power) // 使用 power 参数
    }
}

impl Powf for f32 {
    fn powf(self, power: Self) -> Self {
        self.powf(power)
    }
}

impl Powf for f64 {
    fn powf(self, power: Self) -> Self {
        self.powf(power)
    }
}