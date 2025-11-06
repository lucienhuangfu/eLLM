use std::arch::x86_64::*;

/* 
#[inline(always)]
unsafe fn exp256_ps(x: __m256) -> __m256 {
    let max_x = _mm256_set1_ps(88.3762626647949);
    let min_x = _mm256_set1_ps(-88.3762626647949);
    let clamped = _mm256_min_ps(_mm256_max_ps(x, min_x), max_x);
    let log2e = _mm256_set1_ps(1.4426950408889634);
    let ln2 = _mm256_set1_ps(0.6931471805599453);
    let fx = _mm256_mul_ps(clamped, log2e);
    let fx_floor = _mm256_floor_ps(fx);
    let r = _mm256_sub_ps(clamped, _mm256_mul_ps(fx_floor, ln2));
    let r2 = _mm256_mul_ps(r, r);
    let r3 = _mm256_mul_ps(r2, r);
    let r4 = _mm256_mul_ps(r2, r2);
    let r5 = _mm256_mul_ps(r4, r);
    let mut poly = _mm256_set1_ps(1.0);
    poly = _mm256_add_ps(poly, r);
    poly = _mm256_add_ps(poly, _mm256_mul_ps(r2, _mm256_set1_ps(0.5)));
    poly = _mm256_add_ps(poly, _mm256_mul_ps(r3, _mm256_set1_ps(0.16666667)));
    poly = _mm256_add_ps(poly, _mm256_mul_ps(r4, _mm256_set1_ps(0.04166667)));
    poly = _mm256_add_ps(poly, _mm256_mul_ps(r5, _mm256_set1_ps(0.008333333)));
    let n = _mm256_cvttps_epi32(fx_floor);
    let exponent = _mm256_add_epi32(n, _mm256_set1_epi32(127));
    let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(exponent, 23));
    _mm256_mul_ps(poly, pow2n)
}*/


#[inline]
pub unsafe fn exp256(x: __m256) -> __m256 {
    let mut x = x;
    let exp_hi = _mm256_set1_ps(88.3762626647949f32);
    let exp_lo = _mm256_set1_ps(-88.3762626647949f32);

    let cephes_LOG2EF = _mm256_set1_ps(1.44269504088896341f32);
    let inv_LOG2EF    = _mm256_set1_ps(0.693147180559945f32);

    let cephes_exp_p0 = _mm256_set1_ps(1.9875691500E-4);
    let cephes_exp_p1 = _mm256_set1_ps(1.3981999507E-3);
    let cephes_exp_p2 = _mm256_set1_ps(8.3334519073E-3);
    let cephes_exp_p3 = _mm256_set1_ps(4.1665795894E-2);
    let cephes_exp_p4 = _mm256_set1_ps(1.6666665459E-1);
    let cephes_exp_p5 = _mm256_set1_ps(5.0000001201E-1);
    let one = _mm256_set1_ps(1.0f32);

    x = _mm256_min_ps(x, exp_hi);
    x = _mm256_max_ps(x, exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    let mut fx = _mm256_mul_ps(x, cephes_LOG2EF);
    fx = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
    let mut z = _mm256_mul_ps(fx, inv_LOG2EF);
    x = _mm256_sub_ps(x, z);
    z = _mm256_mul_ps(x,x);

    let mut y = cephes_exp_p0;
    y = _mm256_fmadd_ps(y, x, cephes_exp_p1);
    y = _mm256_fmadd_ps(y, x, cephes_exp_p2);
    y = _mm256_fmadd_ps(y, x, cephes_exp_p3);
    y = _mm256_fmadd_ps(y, x, cephes_exp_p4);
    y = _mm256_fmadd_ps(y, x, cephes_exp_p5);
    y = _mm256_fmadd_ps(y, z, x);
    y = _mm256_add_ps(y, one);

    /* build 2^n */
    let mut imm0 = _mm256_cvttps_epi32(fx);
    imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
    imm0 = _mm256_slli_epi32(imm0, 23);
    let pow2n  = _mm256_castsi256_ps(imm0);
    y = _mm256_mul_ps(y, pow2n);
    y
}

#[inline]
pub unsafe fn tanh256(x: __m256) -> __m256 {
    let negone = _mm256_set1_ps(-1.0f32);
    let posexp = exp256(x);
    let negexp = exp256(_mm256_mul_ps(x, negone));
    let fenzi = _mm256_sub_ps(posexp, negexp);
    let fenmu = _mm256_add_ps(posexp, negexp);
    _mm256_div_ps(fenzi, fenmu)
}

#[inline]
pub unsafe fn sigmoid256(x: __m256) -> __m256 {
    let neg = _mm256_set1_ps(-1.0);
    let pos = _mm256_set1_ps(1.0);
    _mm256_div_ps(pos, _mm256_add_ps(pos, exp256(_mm256_mul_ps(neg, x))))
}