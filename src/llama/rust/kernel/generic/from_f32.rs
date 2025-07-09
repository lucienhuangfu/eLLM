pub trait FromF32 {
    fn from_f32(n: f32) -> Self;
}

impl FromF32 for f16 {
    fn from_f32(n: f32) -> Self {
        n as f16
    }
}

impl FromF32 for f32 {
    fn from_f32(n: f32) -> Self {
        n as f32
    }
}

impl FromF32 for f64 {
    fn from_f32(n: f32) -> Self {
        n as f64
    }
}