pub trait FromUsize {
    fn from_usize(n: usize) -> Self;
}

impl FromUsize for f16 {
    fn from_usize(n: usize) -> Self {
        n as f16
    }
}

impl FromUsize for f32 {
    fn from_usize(n: usize) -> Self {
        n as f32
    }
}

impl FromUsize for f64 {
    fn from_usize(n: usize) -> Self {
        n as f64
    }
}