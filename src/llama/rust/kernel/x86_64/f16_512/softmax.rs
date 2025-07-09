use std::arch::x86_64::*;
use std::ptr;
use std::f16;


use super::super::super::asmsimd::*;
use super::math::exp512;

type Reg = __m512h;

// pub fn _softmax(input_ptr: *const f16, output_ptr: *mut f16, length: usize) {
//     unsafe {
        

//         let mut m_sum = _mm512_set1_ph(f16::ZERO);
//         let mut n_sum = _mm512_set1_ph(f16::MIN);

//         for i in (0..length).step_by(32) {
//             let x_i = _mm512_loadu_ph(input_ptr.add(i));
//             let (m_i, n_i) = ext_exp_avx512(x_i);
//             let n_max = _mm512_max_ph(n_i, n_sum);
//             let exp1 = _mm512_sub_ph(n_i, n_max);
//             let exp2 = _mm512_sub_ph(n_sum, n_max);
//             m_sum = _mm512_add_ph(_mm512_mul_ph(m_i, calculate_exp2_512(exp1)), _mm512_mul_ph(m_sum, calculate_exp2_512(exp2)));
//             n_sum = n_max;
//         }
        
//         let (m_sum, n_sum) = sumh_m512(m_sum, n_sum);


//         let lambda_sum = _mm512_rcp_ph(m_sum);
    

//         for i in (0..length).step_by(32) {
//             let x_i = _mm512_loadu_ph(input_ptr.add(i));
//             let (m_i, n_i) = ext_exp_avx512(x_i);
            
//             let exp = _mm512_sub_ph(n_i, n_sum);
//             let y_i = _mm512_mul_ph(_mm512_mul_ph(m_i, lambda_sum), calculate_exp2_512(exp));
//             _mm512_storeu_ph(output_ptr.add(i), y_i);
//         }
//     }
// }

#[inline]
unsafe fn ext_exp_avx512(x_i: Reg) -> (Reg, Reg) {
    let log2e = _mm512_set1_ph(f16::LOG2_E);
    let loge2 = _mm512_set1_ph(f16::LN_2);
    let temp:Reg = _mm512_mul_ph(x_i, log2e);//x * log2e
    //let n:Reg = _mm512_roundscale_ps(temp, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);//n = round to nearest integer
    let n:Reg = _mm512_roundscale_round_ph::<_MM_FROUND_TO_NEAREST_INT, _MM_FROUND_NO_EXC>(temp);
    let temp1:Reg = _mm512_mul_ph(n, loge2);//n * loge2
    let t:Reg = _mm512_sub_ph(x_i, temp1);//t = x - n * loge2
    (exp512(t),n)
}

#[inline]
unsafe fn calculate_exp2_512(n:Reg) -> Reg {
    let mut n = _mm512_cvttph_epi16(n);
    //fixed bias for f16 is 15
    n = _mm512_add_epi16(n, _mm512_set1_epi16(0x0f));
    //shift by 10 to get the exponent
    n = _mm512_slli_epi16(n, 10);
    let ans = _mm512_castsi512_ph(n);
    _mm512_max_ph(ans, _mm512_setzero_ph())
}

#[inline]
unsafe fn sumh_m512(m_i:Reg, n_i:Reg) -> (Reg,Reg) {
    let n_max = hmax(n_i);
    let exp = _mm512_sub_ph(n_i, n_max);
    let m_i = _mm512_mul_ph(m_i, calculate_exp2_512(exp));
    (hsum(m_i),n_max)
}

#[inline]
unsafe fn hmax(n_i:Reg) -> Reg {
    _mm512_set1_ph(_mm512_reduce_max_ph(n_i))
}

#[inline]
unsafe fn hsum(v:Reg) -> Reg {
    
    let mut sum = _mm512_reduce_add_ph(v);
    
    _mm512_set1_ph(sum)
}

#[inline]
pub fn _scale_softmax(input_ptr: *const f16, output_ptr: *mut f16, length: usize, scale: f16) {
    unsafe {
        let rem = length % 32;
        let length2 = length - rem;

        let mut chunks_sum = f16::ZERO;
        if rem != length {
            let mut m_sum = _mm512_setzero_ph();
            let scale_ = _mm512_set1_ph(scale);
            for (ptr1, ptr2) in (0..length2).step_by(32).map(|x| (input_ptr.add(x), output_ptr.add(x))) {
                let mut x = _mm512_loadu_ph(ptr1);
                x = _mm512_mul_ph(x, scale_);
                let y = exp512(x);
                // println!("{:?}", y);
                _mm512_storeu_ph(ptr2, y);
               m_sum = _mm512_add_ph(m_sum, y);
            }
           chunks_sum = _mm512_reduce_add_ph(m_sum);
        }
        let mut remainder_sum = 0.0f32;
        if rem != 0 {
            for (ptr1, ptr2) in (length2..length).map(|count| (input_ptr.add(count), output_ptr.add(count))) {
                let expx = ((*ptr1).to_f32()*scale.to_f32()).exp();
                remainder_sum += expx;
                ptr::write(ptr2, expx));
            }
        }
        let remainder_sum = remainder_sum);
        let sum = chunks_sum.ss_add(remainder_sum);
        if rem != length {
            let sum1 = _mm512_set1_ph(sum);
            for ptr in (0..length2).step_by(32).map(|x| output_ptr.add(x)) {
                let mut x = _mm512_loadu_ph(ptr);
                x = _mm512_divbyrcp_ph(x, sum1);
                _mm512_storeu_ph(ptr, x);
            }
        }
        if rem != 0 {
            for ptr in (length2..length).map(|count| output_ptr.add(count)) {
                let x1 = *ptr;
                ptr::write(ptr, x1.ss_div_by_rcp(sum));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;
    use super::*;

    // #[test]
    // fn test_softmax() {
    //     let v1: Vec<f32> = (1..17).map(|x| x as f32).collect();
    //     let exp_sum: f32 = v1.iter().map(|x| x.exp()).sum();
    //     let result: Vec<f32> = v1.iter().map(|x| x.exp()/exp_sum).collect();

    //     let v1:Vec<f16> = v1.into_iter().map(|x| x)).collect();
    //     let mut output = vec![f16::ZERO; v1.len()];
    //     _softmax(v1.as_ptr(), output.as_mut_ptr(), v1.len());

    //     //transforming the output to f32
    //     let output: Vec<f32> = output.iter().map(|x| x.to_f32()).collect();
    //     println!("{:?}", output);
    //     println!("{:?}", result);
    // }

    #[test]
    fn test_calculate_exp2_512() {
        let v1: Vec<f32> = (-17..-1).map(|x| x as f32).collect();
        let v1:Vec<f16> = v1.into_iter().map(|x| x)).collect();
        unsafe {
            let x_i = _mm512_loadu_ph(v1.as_ptr());
            //let n = _mm512_set1_ph(-7.0));
            let result = calculate_exp2_512(x_i);
            let mut output = vec![f16::ZERO; v1.len()];
            _mm512_storeu_ph(output.as_mut_ptr(), result);
            println!("{:?}", output);
        }
    }

    // #[test]
    // fn test_scale_softmax() {
    //     let v1: [f32; 36] = [-1.8426496982574463,
    //     0.23383729159832,
    //     -0.7135310173034668,
    //     0.03737562149763107,
    //     0.7525914907455444,
    //     0.5600153207778931,
    //     -0.4442578852176666,
    //     -0.6083138585090637,
    //     -1.7974090576171875,
    //     0.5650796890258789,
    //     -1.218799352645874,
    //     0.5769850015640259,
    //     0.1274106502532959,
    //     -0.05540940538048744,
    //     -0.06808994710445404,
    //     -0.8286862373352051,
    //     -1.5906175374984741,
    //     -2.566009283065796,
    //     0.06307223439216614,
    //     0.4780084490776062,
    //     -0.6066019535064697,
    //     -0.02494196966290474,
    //     2.193176507949829,
    //     0.7341309189796448,
    //     -1.7555780410766602,
    //     1.1725033521652222,
    //     -1.8690853118896484,
    //     1.59326171875,
    //     -0.6819493174552917,
    //     -0.30916815996170044,
    //     -0.23978428542613983,
    //     0.7639755606651306,
    //     0.8059216737747192,
    //     0.8706153035163879,
    //     -0.31227850914001465,
    //     1.0633316040039062];
    //     let v1:Vec<f16> = v1.into_iter().map(|x| x)).collect();
    //     let mut output = vec![f16::ZERO; v1.len()];
    //     _scale_softmax(v1.as_ptr(), output.as_mut_ptr(), v1.len(), f16::from_f32_const(0.65));
    //     println!("{:?}", output)
        // let result: [f32; 36] = [0.0030522907618433237,
        // 0.024346286430954933,
        // 0.009440519846975803,
        // 0.020003709942102432,
        // 0.04090014100074768,
        // 0.0337357260286808,
        // 0.012357757426798344,
        // 0.010487963445484638,
        // 0.00319354934617877,
        // 0.03390700742602348,
        // 0.005695877596735954,
        // 0.03431309387087822,
        // 0.021888310089707375,
        // 0.018231168389320374,
        // 0.01800144463777542,
        // 0.008413653820753098,
        // 0.003927191719412804,
        // 0.0014807264087721705,
        // 0.020524395629763603,
        // 0.031079566106200218,
        // 0.01050593238323927,
        // 0.01879517361521721,
        // 0.17272807657718658,
        // 0.04015202447772026,
        // 0.003329972270876169,
        // 0.06224295496940613,
        // 0.002972659422084689,
        // 0.09480325877666473,
        // 0.00974342506378889,
        // 0.014145179651677608,
        // 0.015161477029323578,
        // 0.04136841744184494,
        // 0.04314056411385536,
        // 0.046023737639188766,
        // 0.014101251028478146,
        // 0.055805567651987076];
        // assert_ulps_eq!(output[..], result, max_ulps=4);
    // }
}