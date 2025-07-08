use std::arch::x86_64::{
    __m512h, _mm512_add_epi16, _mm512_add_ph, _mm512_castsi512_ph, _mm512_cvttph_epi16,
    _mm512_div_ph, _mm512_fmadd_ph, _mm512_max_ph, _mm512_min_ph, _mm512_mul_ph, _mm512_rcp_ph,
    _mm512_roundscale_round_ph, _mm512_set1_epi16, _mm512_set1_ph, _mm512_slli_epi16,
    _mm512_sub_ph, _MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT,
};
// _mm512_divbyrcp_ph
// use super::asmsimd::*;

use std::f16;

#[inline]
fn sigmoid(x: f16) -> f16 {
    1.0 / (1.0 + f16::exp(-x))
}

#[inline]
pub unsafe fn exp512(x: __m512h) -> __m512h {
    let one = _mm512_set1_ph(1.0);
    let max = _mm512_set1_ph(88.0); // maximum value for exp to avoid overflow
    let min = _mm512_set1_ph(-88.0); // minimum value for exp to avoid underflow

    let x = _mm512_max_ph(_mm512_min_ph(x, max), min); // clamp x to the range [-88, 88]

    let x = _mm512_mul_ph(x, _mm512_set1_ph(1.4426950408889634)); // x * log2(e)
    let fx = _mm512_roundscale_round_ph::<{ _MM_FROUND_TO_NEAREST_INT }, { _MM_FROUND_NO_EXC }>(x);

    let tmp = _mm512_sub_ph(x, fx);
    let tmp2 = _mm512_mul_ph(tmp, _mm512_set1_ph(0.6931471805599453)); // tmp * ln(2)

    let mut result = _mm512_add_ph(one, tmp2);
    result = _mm512_fmadd_ph(tmp2, _mm512_set1_ph(0.5), result);
    result = _mm512_fmadd_ph(tmp2, _mm512_set1_ph(0.16666666666666666), result);
    result = _mm512_fmadd_ph(tmp2, _mm512_set1_ph(0.041666666666666664), result);
    result = _mm512_fmadd_ph(tmp2, _mm512_set1_ph(0.008333333333333333), result);
    result = _mm512_fmadd_ph(tmp2, _mm512_set1_ph(0.001388888888888889), result);

    let pow2n = _mm512_castsi512_ph(_mm512_slli_epi16(
        _mm512_add_epi16(_mm512_cvttph_epi16(fx), _mm512_set1_epi16(15)),
        10,
    ));
    _mm512_mul_ph(result, pow2n)
}

#[inline]
pub unsafe fn tanh512(x: __m512h) -> __m512h {
    let negone = _mm512_set1_ph(-1.0);
    let posexp = exp512(x);
    let negexp = exp512(_mm512_mul_ph(x, negone));
    let fenzi = _mm512_sub_ph(posexp, negexp);
    let fenmu = _mm512_add_ph(posexp, negexp);
    // _mm512_divbyrcp_ph(fenzi, fenmu)
    _mm512_div_ph(fenzi, fenmu)
}

#[inline]
pub unsafe fn sigmoid512(x: __m512h) -> __m512h {
    let neg = _mm512_set1_ph(-1.0);
    let pos = _mm512_set1_ph(1.0);
    // _mm512_divbyrcp_ph(pos, _mm512_add_ph(pos, exp512(_mm512_mul_ph(neg, x))))

    let b = _mm512_add_ph(pos, exp512(_mm512_mul_ph(neg, x)));
    let c = _mm512_rcp_ph(b);
    _mm512_mul_ph(c, pos)
}

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;

    use super::*;
    use std::arch::x86_64::{_mm512_loadu_ph, _mm512_storeu_ph};
    /*
    #[test]
    fn test_exp() {
        let v: Vec<f16> = vec![2.0f16; 32];

        unsafe {
            let a = _mm512_loadu_ph(v.as_ptr());
            let o = exp512(a);

            let mut res: Vec<f16> = vec![0.0; 32];

            _mm512_storeu_ph(res.as_mut_ptr(), o);

            let expected: Vec<f16> = vec![2.0f16; 32].into_iter().map(|x| x.exp()).collect();
            for j in 0..32 {
                println!(
                    "Input: {}, Expected: {}, Result: {}",
                    v[j] as f32, expected[j] as f32, res[j] as f32
                );
                assert_ulps_eq!(res[j] as f32, expected[j] as f32, max_ulps = 4);
            }
        }
    } */
}
