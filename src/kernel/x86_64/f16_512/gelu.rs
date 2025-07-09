use std::ptr;
use super::super::super::asmsimd::*;
use std::f16;
use super::math::tanh512;
const SQRT_2_DIV_PI: f16 = f16::from_f32_const(0.7978845608028654);

#[inline(always)]
fn _single_gelu(x: f16) -> f16 {
    let y = (x.ss_add(f16::from_f32_const(0.044715f32).ss_mul(x).ss_mul(x).ss_mul(x))).ss_mul(SQRT_2_DIV_PI);
    let y2 = f16::from_f32_const(0.5).ss_mul(x).ss_mul(f16::ONE.ss_add(y.to_f32().tanh())));
    y2
}

pub fn _gelu(input_ptr: *const f16, output_ptr: *mut f16, length: usize) {
    unsafe {
        let rem = length % 32;
        let length2 = length - rem;

        if rem != length {
            let var1 = _mm512_set1_ph(f16::from_f32_const(0.044715f32));
            let var2 = _mm512_set1_ph(SQRT_2_DIV_PI);
            let var10 = _mm512_set1_ph(f16::from_f32_const(0.5f32));
            let var_one = _mm512_set1_ph(f16::ONE);
            for (ptr1, ptr2) in (0..length2).step_by(32).map(|x| (input_ptr.add(x), output_ptr.add(x))) {
                let x = _mm512_loadu_ph(ptr1);
                let mut y = _mm512_mul_ph(x, x);
                y = _mm512_mul_ph(y, x);
                y = _mm512_fmadd_ph(y, var1, x);
                y = _mm512_mul_ph(y, var2);
                y = tanh512(y);
                y = _mm512_add_ph(y, var_one);
                y = _mm512_mul_ph(y, x);
                y = _mm512_mul_ph(y, var10);
                _mm512_storeu_ph(ptr2, y);
            }
        }
        if rem != 0 {
            for (ptr1, ptr2) in (length2..length).map(|count| (input_ptr.add(count), output_ptr.add(count))) {
                ptr::write(ptr2, _single_gelu(*ptr1));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;
    use super::*;

    #[test]
    fn test_gelu() {
        let v1: [f32; 36] = [0.3136247992515564,
        -2.6968541145324707,
        0.7449121475219727,
        -0.3058519959449768,
        -1.034066915512085,
        -0.985573410987854,
        -0.5345404744148254,
        -1.3619849681854248,
        0.3012881577014923,
        0.8911539912223816,
        -1.2453598976135254,
        -0.3054046630859375,
        0.5982641577720642,
        -0.21695956587791443,
        -0.0798346996307373,
        -0.7486835718154907,
        0.6165731549263,
        -1.0666285753250122,0.3136247992515564,
        -2.6968541145324707,
        0.7449121475219727,
        -0.3058519959449768,
        -1.034066915512085,
        -0.985573410987854,
        -0.5345404744148254,
        -1.3619849681854248,
        0.3012881577014923,
        0.8911539912223816,
        -1.2453598976135254,
        -0.3054046630859375,
        0.5982641577720642,
        -0.21695956587791443,
        -0.0798346996307373,
        -0.7486835718154907,
        0.6165731549263,
        -1.0666285753250122];
        let v1: Vec<f16> = v1.into_iter().map(|x| x)).collect();
        let mut output = [f16::ZERO; 36];
        // let result: [f32; 18] = [0.1954157054424286,
        // -0.008965473622083664,0.5748836994171143,
        // -0.11618322879076004,
        // -0.15584588050842285,
        // -0.15997932851314545,
        // -0.15850451588630676,
        // -0.11818332970142365,
        // 0.18631485104560852,
        // 0.7249078154563904,
        // -0.13285332918167114,
        // -0.11606530100107193,
        // 0.4338094890117645,
        // -0.08984798192977905,
        // -0.03737737238407135,
        // -0.1700376570224762,
        // 0.4508279263973236,
        // -0.15277788043022156];
        
        _gelu(v1.as_ptr(), output.as_mut_ptr(), v1.len());
        // assert_ulps_eq!(output[..], result[..], max_ulps=4)
        println!("{:?}", output);
    }
}