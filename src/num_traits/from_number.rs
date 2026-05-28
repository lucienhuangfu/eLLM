pub trait FromNumber {
    fn from_f32(n: f32) -> Self;
    fn from_usize(n: usize) -> Self;
}

impl FromNumber for f16 {
    fn from_f32(n: f32) -> Self {
        n as f16
    }

    fn from_usize(n: usize) -> Self {
        n as f16
    }
}

impl FromNumber for f32 {
    fn from_f32(n: f32) -> Self {
        n
    }

    fn from_usize(n: usize) -> Self {
        n as f32
    }
}

impl FromNumber for f64 {
    fn from_f32(n: f32) -> Self {
        n as f64
    }

    fn from_usize(n: usize) -> Self {
        n as f64
    }
}
