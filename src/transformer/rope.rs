use crate::common::num_traits::FromNumber;

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotaryEmbedding {
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub max_sequence_length: usize,
    pub theta: f32,
    pub attention_scaling: f32,
}

impl RotaryEmbedding {
    pub fn new(head_dim: usize, rotary_dim: usize, max_sequence_length: usize, theta: f32) -> Self {
        Self {
            head_dim,
            rotary_dim: rotary_dim.min(head_dim),
            max_sequence_length,
            theta,
            attention_scaling: 1.0,
        }
    }

    pub fn forward<T: FromNumber>(&self) -> Vec<T> {
        debug_assert!(self.head_dim % 2 == 0, "RoPE head_dim must be even");
        debug_assert!(
            self.rotary_dim <= self.head_dim,
            "rotary_dim must not exceed head_dim"
        );
        debug_assert!(self.rotary_dim % 2 == 0, "RoPE rotary_dim must be even");

        let rotary_pairs = self.rotary_dim / 2;
        let inv = inv_freqs(self.rotary_dim, self.theta);
        let mut out = Vec::with_capacity(self.max_sequence_length * self.head_dim);

        for pos in 0..self.max_sequence_length {
            let t = pos as f32;

            for &inv_f in &inv {
                let angle = t * inv_f;
                out.push(T::from_f32(angle.cos() * self.attention_scaling));
                out.push(T::from_f32(angle.sin() * self.attention_scaling));
            }

            for _ in rotary_pairs..(self.head_dim / 2) {
                out.push(T::from_f32(1.0));
                out.push(T::from_f32(0.0));
            }
        }

        out
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_freqs() {
        let rope = RotaryEmbedding::new(64, 64, 512, 10000.0);
        let freqs = rope.forward::<f32>();
        let dim = 64;
        let max_len = 512;
        assert_eq!(freqs.len(), dim * max_len);
        // position 0 should be cos=1, sin=0 for all pairs
        for i in (0..dim).step_by(2) {
            assert!((freqs[i] - 1.0).abs() < 1e-6);
            assert!(freqs[i + 1].abs() < 1e-6);
        }
    }

    #[test]
    fn test_partial_rotary_embedding_identity_tail() {
        let rope = RotaryEmbedding::new(8, 4, 2, 10_000.0);
        let freqs = rope.forward::<f32>();

        assert_eq!(freqs.len(), 16);

        assert!((freqs[0] - 1.0).abs() < 1e-6);
        assert!(freqs[1].abs() < 1e-6);
        assert!((freqs[2] - 1.0).abs() < 1e-6);
        assert!(freqs[3].abs() < 1e-6);

        assert!((freqs[4] - 1.0).abs() < 1e-6);
        assert!(freqs[5].abs() < 1e-6);
        assert!((freqs[6] - 1.0).abs() < 1e-6);
        assert!(freqs[7].abs() < 1e-6);

        let pos1 = 8;
        assert!((freqs[pos1] - 1.0f32.cos()).abs() < 1e-6);
        assert!((freqs[pos1 + 1] - 1.0f32.sin()).abs() < 1e-6);
        assert!((freqs[pos1 + 2] - 0.01f32.cos()).abs() < 1e-6);
        assert!((freqs[pos1 + 3] - 0.01f32.sin()).abs() < 1e-6);
        assert!((freqs[pos1 + 4] - 1.0).abs() < 1e-6);
        assert!((freqs[pos1 + 5] - 0.0).abs() < 1e-6);
        assert!((freqs[pos1 + 6] - 1.0).abs() < 1e-6);
        assert!((freqs[pos1 + 7] - 0.0).abs() < 1e-6);
    }
}
