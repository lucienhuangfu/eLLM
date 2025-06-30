// Common functions for vectors

use std::arch::x86_64::*;

#[inline]
pub unsafe fn hsum_ps_sse3(v: __m128) -> f32 {
    let mut shuf = _mm_movehdup_ps(v);
    let mut sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    _mm_cvtss_f32(sums)
}

// horizontal sum
#[inline]
pub unsafe fn hsum256_ps_avx(v: __m256) -> f32 {
    let mut vlow  = _mm256_castps256_ps128(v);
    let vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    hsum_ps_sse3(vlow)
}

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;

    use super::*;

    #[test]
    fn test_hsum() {
        unsafe {
            let v = _mm256_set_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
            let sum = hsum256_ps_avx(v);
            assert_ulps_eq!(36.0f32, sum, max_ulps=4);
        }
    }
}