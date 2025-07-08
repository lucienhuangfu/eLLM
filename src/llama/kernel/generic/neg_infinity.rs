use std::f16;

pub trait NegInfinity {
    fn neg_infinity() -> Self;
}

impl NegInfinity for f16 {
    fn neg_infinity() -> Self {
        f16::NEG_INFINITY
    }
}

impl NegInfinity for f32 {
    fn neg_infinity() -> Self {
        f32::NEG_INFINITY
    }
}

impl NegInfinity for f64 {
    fn neg_infinity() -> Self {
        f64::NEG_INFINITY
    }
}