use std::arch::x86_64::{
    __m512h, __m512i, _mm512_abs_ph, _mm512_add_epi16, _mm512_add_ph, _mm512_and_si512,
    _mm512_andnot_si512, _mm512_castph_si512, _mm512_castsi512_ph, _mm512_cvttph_epi16,
    _mm512_div_ph, _mm512_fmadd_ph, _mm512_fnmadd_ph, _mm512_max_epi16, _mm512_max_ph,
    _mm512_min_ph, _mm512_mul_ph, _mm512_or_si512, _mm512_roundscale_round_ph, _mm512_set1_epi16,
    _mm512_set1_ph, _mm512_setzero_si512, _mm512_slli_epi16, _mm512_sub_ph, _MM_FROUND_NO_EXC,
    _MM_FROUND_TO_NEAREST_INT,
};

use std::f16;

#[inline]
pub unsafe fn exp512(x: __m512h) -> __m512h {
    let mut x = x;
    // f16 max is ~65504, ln(65504) ~= 11.09.
    // Clamp input to avoid overflow in exp result or intermediate 2^n calculation.
    let exp_hi = _mm512_set1_ph(11.088);
    let exp_lo = _mm512_set1_ph(-17.0); // exp(-17) is small enough to be considered 0 in f16

    let cephes_LOG2EF = _mm512_set1_ph(1.44269504088896341);

    // Split ln(2) into hi and lo for higher precision range reduction
    // ln2_hi = 0.693115234375 (exactly representable in f16 as 0x398b)
    let ln2_hi = _mm512_set1_ph(0.69311523);
    let ln2_lo = _mm512_set1_ph(0.000031946);

    let cephes_exp_p0 = _mm512_set1_ph(1.9875691500E-4);
    let cephes_exp_p1 = _mm512_set1_ph(1.3981999507E-3);
    let cephes_exp_p2 = _mm512_set1_ph(8.3334519073E-3);
    let cephes_exp_p3 = _mm512_set1_ph(4.1665795894E-2);
    let cephes_exp_p4 = _mm512_set1_ph(1.6666665459E-1);
    let cephes_exp_p5 = _mm512_set1_ph(5.0000001201E-1);
    let one = _mm512_set1_ph(1.0);

    x = _mm512_min_ph(x, exp_hi);
    x = _mm512_max_ph(x, exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    let mut fx = _mm512_mul_ph(x, cephes_LOG2EF);
    fx = _mm512_roundscale_round_ph::<_MM_FROUND_TO_NEAREST_INT, _MM_FROUND_NO_EXC>(fx);

    // Range reduction with higher precision: x = x - fx * ln2_hi - fx * ln2_lo
    // Uses Fused Negative Multiply Add: -(a*b) + c
    x = _mm512_fnmadd_ph(fx, ln2_hi, x);
    x = _mm512_fnmadd_ph(fx, ln2_lo, x);

    let z = _mm512_mul_ph(x, x);

    let mut y = cephes_exp_p0;
    y = _mm512_fmadd_ph(y, x, cephes_exp_p1);
    y = _mm512_fmadd_ph(y, x, cephes_exp_p2);
    y = _mm512_fmadd_ph(y, x, cephes_exp_p3);
    y = _mm512_fmadd_ph(y, x, cephes_exp_p4);
    y = _mm512_fmadd_ph(y, x, cephes_exp_p5);
    y = _mm512_fmadd_ph(y, z, x);
    y = _mm512_add_ph(y, one);

    /* build 2^n */
    let mut imm0 = _mm512_cvttph_epi16(fx);
    // f16 exponent bias is 15
    imm0 = _mm512_add_epi16(imm0, _mm512_set1_epi16(0xf));
    imm0 = _mm512_max_epi16(imm0, _mm512_setzero_si512());
    imm0 = _mm512_slli_epi16(imm0, 10);
    let pow2n = _mm512_castsi512_ph(imm0);

    y = _mm512_mul_ph(y, pow2n);
    y
}

#[inline]
pub unsafe fn tanh512(x: __m512h) -> __m512h {
    // tanh(x) = sign(x) * (1 - exp(-2|x|)) / (1 + exp(-2|x|))
    // This form is numerically stable and avoids overflow for large |x|.

    let one = _mm512_set1_ph(1.0);
    let minus_two = _mm512_set1_ph(-2.0);
    let sign_mask = _mm512_set1_epi16(-32768); // 0x8000

    // Calculate |x|
    let x_abs = _mm512_abs_ph(x);

    // Calculate exp(-2|x|)
    let exp_arg = _mm512_mul_ph(x_abs, minus_two);
    let e = exp512(exp_arg);

    // (1 - e) / (1 + e)
    let num = _mm512_sub_ph(one, e);
    let den = _mm512_add_ph(one, e);
    let res_abs = _mm512_div_ph(num, den);

    // Restore sign: result = copysign(res_abs, x)
    let x_bits = _mm512_castph_si512(x);
    let res_bits = _mm512_castph_si512(res_abs);

    let sign = _mm512_and_si512(x_bits, sign_mask);
    let val = _mm512_andnot_si512(sign_mask, res_bits);
    let result = _mm512_or_si512(val, sign);

    _mm512_castsi512_ph(result)
}

#[inline]
pub unsafe fn sigmoid512(x: __m512h) -> __m512h {
    let one = _mm512_set1_ph(1.0);
    let neg_one = _mm512_set1_ph(-1.0);

    // sigmoid(x) = 1 / (1 + exp(-x))
    // If x is large positive, exp(-x) -> 0, res -> 1.
    // If x is large negative, exp(-x) -> inf. 1+inf = inf. 1/inf = 0.
    // exp512 clamps input, so max exp output is ~65504.
    // 1 + 65504 overflows f16 to INF. 1/INF is 0. This behavior is correct.

    let exp_neg_x = exp512(_mm512_mul_ph(x, neg_one));
    let den = _mm512_add_ph(one, exp_neg_x);
    _mm512_div_ph(one, den)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::allocator::allocate_init;
    use std::arch::x86_64::{_mm512_load_ph, _mm512_store_ph};
    use std::slice;

    const TOLERANCE: f16 = 1e-3;

    #[test]
    fn test_exp() {
        let length = 32;
        // Test range from -10 to 10
        let input_vals: Vec<f16> = (0..length)
            .map(|i| ((i as f16 - 16.0) * 0.5))
            .collect();
        let mut v = allocate_init::<f16>(length, 0.0);
        unsafe {
            std::ptr::copy_nonoverlapping(input_vals.as_ptr(), v, length);

            let a = _mm512_load_ph(v);
            let o = exp512(a);

            let mut res = allocate_init::<f16>(length, 0.0);
            _mm512_store_ph(res, o);
            let res_slice = slice::from_raw_parts(res, length);

            for j in 0..length {
                let expected = input_vals[j].exp();
                let actual = res_slice[j];
                println!(
                    "Exp: x={}, actual={}, expected={}",
                    input_vals[j], actual, expected
                );

                if expected.is_infinite() {
                    assert!(actual.is_infinite() || actual > 65000.0);
                } else {
                    assert!(
                        (actual - expected).abs() < TOLERANCE * expected.abs().max(1.0f16),
                        "Mismatch at index {}: got {}, expected {}",
                        j,
                        actual,
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn test_tanh() {
        let length = 32;
        let input_vals: Vec<f16> = (0..length)
            .map(|i| ((i as f16 - 16.0) * 0.5))
            .collect();
        let mut v = allocate_init::<f16>(length, 0.0);

        unsafe {
            std::ptr::copy_nonoverlapping(input_vals.as_ptr(), v, length);
            let a = _mm512_load_ph(v);
            let o = tanh512(a);

            let mut res = allocate_init::<f16>(length, 0.0);
            _mm512_store_ph(res, o);
            let res_slice = slice::from_raw_parts(res, length);

            for j in 0..length {
                let expected = input_vals[j].tanh();
                let actual = res_slice[j];
                println!(
                    "Tanh: x={}, actual={}, expected={}",
                    input_vals[j], actual, expected
                );
                assert!(
                    (actual - expected).abs() < TOLERANCE,
                    "Mismatch at index {}: got {}, expected {}",
                    j,
                    actual,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_sigmoid() {
        let length = 32;
        let input_vals: Vec<f16> = (0..length)
            .map(|i| ((i as f16 - 16.0) * 0.5))
            .collect();
        let mut v = allocate_init::<f16>(length, 0.0);

        unsafe {
            std::ptr::copy_nonoverlapping(input_vals.as_ptr(), v, length);
            let a = _mm512_load_ph(v);
            let o = sigmoid512(a);

            let mut res = allocate_init::<f16>(length, 0.0);
            _mm512_store_ph(res, o);
            let res_slice = slice::from_raw_parts(res, length);

            for j in 0..length {
                let x = input_vals[j];
                let expected = 1.0 / (1.0 + (-x).exp());
                let actual = res_slice[j];
                println!("Sigmoid: x={}, actual={}, expected={}", x, actual, expected);
                assert!(
                    (actual - expected).abs() < TOLERANCE,
                    "Mismatch at index {}: got {}, expected {}",
                    j,
                    actual,
                    expected
                );
            }
        }
    }
}
