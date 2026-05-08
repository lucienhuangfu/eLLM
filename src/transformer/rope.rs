use std::collections::HashMap;

use crate::common::num_traits::FromNumber;
use serde_json::Value;

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
struct YarnScaling {
    factor: f32,
    original_max_position_embeddings: usize,
    attention_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
}

fn parse_rope_scaling(
    rope_scaling: Option<&HashMap<String, Value>>,
    default_original_max_position_embeddings: usize,
) -> Option<YarnScaling> {
    let rope_scaling = rope_scaling?;

    let rope_type = rope_scaling
        .get("rope_type")
        .or_else(|| rope_scaling.get("type"))?
        .as_str()?
        .to_ascii_lowercase();

    if rope_type != "yarn" {
        return None;
    }

    let factor = rope_scaling
        .get("factor")
        .and_then(value_to_f32)
        .filter(|factor| *factor > 0.0)
        .unwrap_or(1.0);

    let original_max_position_embeddings = rope_scaling
        .get("original_max_position_embeddings")
        .and_then(value_to_usize)
        .unwrap_or(default_original_max_position_embeddings);

    let beta_fast = rope_scaling
        .get("beta_fast")
        .and_then(value_to_f32)
        .unwrap_or(32.0);

    let beta_slow = rope_scaling
        .get("beta_slow")
        .and_then(value_to_f32)
        .unwrap_or(1.0);

    let attention_factor = rope_scaling
        .get("attention_factor")
        .or_else(|| rope_scaling.get("attn_factor"))
        .and_then(value_to_f32)
        .unwrap_or_else(|| 0.1 * factor.ln() + 1.0);

    Some(YarnScaling {
        factor,
        original_max_position_embeddings,
        attention_factor,
        beta_fast,
        beta_slow,
    })
}

fn value_to_f32(value: &Value) -> Option<f32> {
    match value {
        Value::Number(number) => number.as_f64().map(|v| v as f32),
        Value::String(text) => text.parse::<f32>().ok(),
        _ => None,
    }
}

fn value_to_usize(value: &Value) -> Option<usize> {
    match value {
        Value::Number(number) => number.as_u64().map(|v| v as usize),
        Value::String(text) => text.parse::<usize>().ok(),
        _ => None,
    }
}

fn apply_yarn_scaling(
    inv_freqs: &mut [f32],
    rotary_dim: usize,
    theta: f32,
    yarn: &YarnScaling,
) {
    if yarn.factor <= 1.0 {
        return;
    }

    let rotary_pairs = rotary_dim / 2;
    if rotary_pairs == 0 {
        return;
    }

    let low_rot = yarn.beta_fast.max(1.0).round() as usize;
    let high_rot = yarn.beta_slow.max(1.0).round() as usize;
    let (low, high) = yarn_find_correction_range(
        low_rot,
        high_rot,
        rotary_dim,
        theta,
        yarn.original_max_position_embeddings,
    );
    let ramp = yarn_linear_ramp_mask(low as f32, high as f32, rotary_pairs);
    let inv_freq_extrapolation = inv_freqs.to_vec();
    let inv_freq_interpolation: Vec<f32> = inv_freq_extrapolation
        .iter()
        .map(|freq| *freq / yarn.factor)
        .collect();

    for i in 0..rotary_pairs {
        let inv_freq_mask = 1.0 - ramp[i];
        inv_freqs[i] = inv_freq_interpolation[i] * (1.0 - inv_freq_mask)
            + inv_freq_extrapolation[i] * inv_freq_mask;
    }
}

fn yarn_find_correction_dim(
    num_rotations: usize,
    dim: usize,
    base: f32,
    max_position_embeddings: usize,
) -> f32 {
    (dim as f32
        * ((max_position_embeddings as f32) / (num_rotations as f32 * 2.0 * core::f32::consts::PI))
            .ln())
        / (2.0 * base.ln())
}

fn yarn_find_correction_range(
    low_rot: usize,
    high_rot: usize,
    dim: usize,
    base: f32,
    max_position_embeddings: usize,
) -> (usize, usize) {
    let low = yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings).floor();
    let high = yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings).ceil();
    let low = low.max(0.0) as usize;
    let high = high.min((dim.saturating_sub(1)) as f32) as usize;
    (low, high)
}

fn yarn_linear_ramp_mask(low: f32, high: f32, dim: usize) -> Vec<f32> {
    let high = if (low - high).abs() < f32::EPSILON {
        high + 0.001
    } else {
        high
    };

    (0..dim)
        .map(|i| {
            let linear = (i as f32 - low) / (high - low);
            linear.clamp(0.0, 1.0)
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
    yarn_scaling: Option<YarnScaling>,
}

impl RotaryEmbedding {
    pub fn new(
        head_dim: usize,
        rotary_dim: usize,
        max_sequence_length: usize,
        theta: f32,
        rope_scaling: Option<HashMap<String, Value>>,
    ) -> Self {
        let yarn = parse_rope_scaling(rope_scaling.as_ref(), max_sequence_length);
        let attention_scaling = yarn.map(|y| y.attention_factor).unwrap_or(1.0);
        Self {
            head_dim,
            rotary_dim: rotary_dim.min(head_dim),
            max_sequence_length,
            theta,
            attention_scaling,
            yarn_scaling: yarn,
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
        let mut inv = inv_freqs(self.rotary_dim, self.theta);
        if let Some(yarn) = &self.yarn_scaling {
            apply_yarn_scaling(
                &mut inv,
                self.rotary_dim,
                self.theta,
                &yarn,
            );
        }
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
    use serde_json::json;

    #[test]
    fn test_freqs() {
        let rope = RotaryEmbedding::new(64, 64, 512, 10000.0, None);
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
        let rope = RotaryEmbedding::new(8, 4, 2, 10_000.0, None);
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

    #[test]
    fn test_yarn_scaling_uses_attention_factor() {
        let mut rope_scaling = HashMap::new();
        rope_scaling.insert("rope_type".to_string(), json!("yarn"));
        rope_scaling.insert("factor".to_string(), json!(4.0));
        rope_scaling.insert(
            "original_max_position_embeddings".to_string(),
            json!(2usize),
        );
        rope_scaling.insert("attention_factor".to_string(), json!(1.25));

        let rope = RotaryEmbedding::new(8, 8, 16, 10_000.0, Some(rope_scaling));
        assert!((rope.attention_scaling - 1.25).abs() < 1e-6);

        let freqs = rope.forward::<f32>();
        let base = RotaryEmbedding::new(8, 8, 16, 10_000.0, None).forward::<f32>();

        assert_eq!(freqs.len(), base.len());
        assert!((freqs[0] - 1.25).abs() < 1e-6);
        assert!(freqs
            .iter()
            .zip(base.iter())
            .any(|(lhs, rhs)| (lhs - rhs).abs() > 1e-6));
    }
}
