use crate::num_traits::FromNumber;

#[inline]
fn inv_freqs(dim: usize, theta: f32) -> Vec<f32> {
    debug_assert!(dim % 2 == 0, "RoPE head_dim must be even");
    (0..dim)
        .step_by(2)
        .map(|i| {
            let p = i as f32 / dim as f32;
            theta.powf(p).recip()
        })
        .collect()
}

/// Precompute RoPE cache as interleaved [cos, sin, cos, sin, ...] for each position.
/// Output length = max_sequence_length * dim.
pub fn precompute_freqs_cis(dim: usize, max_sequence_length: usize, theta: f32) -> Vec<f32> {
    let inv = inv_freqs(dim, theta);
    let mut out = Vec::with_capacity(max_sequence_length * dim);

    for pos in 0..max_sequence_length {
        let t = pos as f32;
        for &inv_f in &inv {
            let angle = t * inv_f;
            out.push(angle.cos());
            out.push(angle.sin());
        }
    }
    out
}

/// Same as `precompute_freqs_cis`, but returns a generic numeric type (f16/f32/f64).
pub fn precompute_freqs_cis_t<T: FromNumber>(
    dim: usize,
    max_sequence_length: usize,
    theta: f32,
) -> Vec<T> {
    precompute_freqs_cis(dim, max_sequence_length, theta)
        .into_iter()
        .map(T::from_f32)
        .collect()
}

#[cfg(test)]
mod test {
    // use std::f16;
    use super::*;
    #[test]
    fn test_freqs() {
        let dim = 64;
        let max_len = 512;
        let freqs = precompute_freqs_cis(dim, max_len, 10000.0);
        assert_eq!(freqs.len(), dim * max_len);
        // position 0 should be cos=1, sin=0 for all pairs
        for i in (0..dim).step_by(2) {
            assert!((freqs[i] - 1.0).abs() < 1e-6);
            assert!(freqs[i + 1].abs() < 1e-6);
        }
    }

}
