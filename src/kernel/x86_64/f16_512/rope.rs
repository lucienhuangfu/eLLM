// === kernel/x86_64/f16_512/rope.rs ===
#![allow(non_snake_case)]

use std::arch::x86_64::{_mm512_fmul_pch, _mm512_loadu_ph, _mm512_storeu_ph};
use std::f16;

/// AVX-512 in-place rotate_half_rope: replaces the scalar Vec-allocating version.
///
/// Input head layout: [x0..x_{half-1}, x_{half}..x_{len-1}]  (two halves, not interleaved)
/// Rope layout:       [cos0, sin0, cos1, sin1, ...]          (interleaved complex pairs)
///
/// For each i: (head[i], head[i+half]) *= complex (rope[2i], rope[2i+1])
///
/// Works in-place in groups of 32 f16 (16 complex pairs) using _mm512_fmul_pch.
/// No heap allocation, no scalar loop.
#[target_feature(enable = "avx512fp16")]
pub unsafe fn rotate_half_rope_avx512(head: *mut f16, rope: *const f16, length: usize) {
    debug_assert_eq!(length % 32, 0, "length must be multiple of 32");
    let half = length / 2; // e.g. 128 → 64

    // Process 16 complex pairs per iteration (32 f16 = one AVX-512 vector).
    let mut interleaved = [0.0_f16; 32];
    for offset in (0..half).step_by(16) {
        // Load 16 elements from first half  → lower 16 of the 32-element vector
        // Load 16 elements from second half → upper 16
        // Interleave into (head[i], head[i+half]) pairs for complex_mul
        let lo = head.add(offset);
        let hi = head.add(half + offset);
        let rp = rope.add(2 * offset);

        for j in 0..16 {
            interleaved[2 * j] = *lo.add(j);
            interleaved[2 * j + 1] = *hi.add(j);
        }

        let x = _mm512_loadu_ph(interleaved.as_ptr());
        let y = _mm512_loadu_ph(rp);
        let z = _mm512_fmul_pch(x, y);

        // Store result back to interleaved, then scatter to lo/hi halves
        _mm512_storeu_ph(interleaved.as_mut_ptr(), z);
        for j in 0..16 {
            *lo.add(j) = interleaved[2 * j];
            *hi.add(j) = interleaved[2 * j + 1];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::arch::is_x86_feature_detected;

    fn scalar_rotate_half_rope_reference(head: &[f16], rope: &[f16]) -> Vec<f16> {
        let length = head.len();
        let half = length / 2;
        let mut result = vec![0.0_f16; length];
        for i in 0..half {
            let x1 = head[i] as f32;
            let x2 = head[i + half] as f32;
            let cos = rope[2 * i] as f32;
            let sin = rope[2 * i + 1] as f32;
            result[i] = (x1 * cos - x2 * sin) as f16;
            result[i + half] = (x2 * cos + x1 * sin) as f16;
        }
        result
    }

    #[test]
    fn test_rotate_half_rope_avx512_basic() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }

        let length: usize = 128;
        let half = length / 2;

        let mut head = vec![0.0_f16; length];
        let mut rope = vec![0.0_f16; length];

        for i in 0..length {
            head[i] = (0.01 * i as f32 + 0.5) as f16;
        }
        for i in 0..half {
            let angle = 0.1 * i as f32;
            rope[2 * i] = angle.cos() as f16;
            rope[2 * i + 1] = angle.sin() as f16;
        }

        let expected = scalar_rotate_half_rope_reference(&head, &rope);

        unsafe {
            rotate_half_rope_avx512(head.as_mut_ptr(), rope.as_ptr(), length);
        }

        for i in 0..length {
            let got = head[i] as f32;
            let exp = expected[i] as f32;
            assert!((got - exp).abs() < 0.1, "mismatch at {}: got={}, exp={}", i, got, exp);
        }
    }

    #[test]
    fn test_rotate_half_rope_avx512_identity_rope() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }

        let length: usize = 128;
        let half = length / 2;

        let mut head = vec![0.0_f16; length];
        let mut rope = vec![0.0_f16; length];

        for i in 0..length {
            head[i] = (0.01 * i as f32) as f16;
        }
        // Identity rope: cos=1, sin=0
        for i in 0..half {
            rope[2 * i] = 1.0_f16;
            rope[2 * i + 1] = 0.0_f16;
        }

        let original = head.clone();
        unsafe {
            rotate_half_rope_avx512(head.as_mut_ptr(), rope.as_ptr(), length);
        }

        for i in 0..length {
            let got = head[i] as f32;
            let exp = original[i] as f32;
            assert!((got - exp).abs() < 0.05, "identity mismatch at {}: got={}, exp={}", i, got, exp);
        }
    }
}
