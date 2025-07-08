use std::arch::x86_64::{__m512h, _MM_FROUND_TO_NEAREST_INT, _MM_FROUND_NO_EXC, 
    _mm512_add_epi16, _mm512_set1_epi16, _mm512_slli_epi16,
     _mm512_set1_ph, _mm512_min_ph, _mm512_max_ph, _mm512_mul_ph, 
    _mm512_roundscale_round_ph, _mm512_sub_ph, _mm512_fmadd_ph, _mm512_add_ph,
    _mm512_cvttph_epi16, _mm512_castsi512_ph, _mm512_div_ph, _mm512_rcp_ph
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
    let mut x = x;
    let exp_hi = _mm512_set1_ph(88.3762626647949);
    let exp_lo = _mm512_set1_ph(-88.3762626647949);

    let cephes_LOG2EF = _mm512_set1_ph(1.44269504088896341);
    let inv_LOG2EF    = _mm512_set1_ph(0.693147180559945);

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
    let mut z = _mm512_mul_ph(fx, inv_LOG2EF);
    x = _mm512_sub_ph(x, z);
    z = _mm512_mul_ph(x,x);

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
    imm0 = _mm512_add_epi16(imm0, _mm512_set1_epi16(0xf));
    imm0 = _mm512_slli_epi16(imm0, 10);
    let pow2n  = _mm512_castsi512_ph(imm0);
    y = _mm512_mul_ph(y, pow2n);
    y
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
    use std::arch::x86_64::{_mm512_load_ph, _mm512_store_ph};
    use std::slice;
    // use approx::assert_ulps_eq;
    // use std::ptr;

    use super::*;
    use crate::memory::allocator::allocate_init;

#[test]
fn test_exp() {
    // let scalar1 = 1.0;
    // let scalar2 = 2.0;
    // let v: Vec<f16> = vec![2.0f16;32];

    let length = 32;
    let mut v = allocate_init::<f16>(length, 2.0);

    unsafe {
        let a = _mm512_load_ph(v);
        let o = exp512(a);
  

        // let mut res: Vec<f16> = vec![0.0; 32];
        let mut res = allocate_init::<f16>(length, 0.0);
        let res_slice = slice::from_raw_parts(res, length);

        
        _mm512_store_ph(res, o);
        
        let mut expected: Vec<f16> = vec![2.0f16; 32].into_iter().map(|x| x.exp()).collect();
        for j in 0..32 {
            println!("{} {}", res_slice[j] as f32, expected[j] as f32);
            // assert!(f16::abs(res[j] - exp[j]) < 1e-6f16);
            // assert_ulps_eq!(res[j] as f32, 2.0f32.exp(), max_ulps = 5);
       
        }
    }
}
}
