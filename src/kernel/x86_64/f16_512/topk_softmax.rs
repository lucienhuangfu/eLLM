use std::arch::x86_64::*;
use std::ptr;
use std::f16;
use super::activation::exp512;
use crate::kernel::generic::merge_topk::merge_topk_lists;

pub fn topk_softmax(
    // [thread_num, topk_size]
    input_indices_ptr: *const usize,
    // [thread_num, topk_size]
    input_values_ptr: *const f16,
    // [thread_num]
    sums_ptr: *const f16,

    // max_positions_ptr: *mut usize,
    // [topk_size]
    output_indices_ptr: *mut usize,
    // [topk_size]
    output_values_ptr: *mut f16,
    // [1]
    output_token_ptr: *mut usize,
    thread_num: usize,
    topk_size: usize,
) {
    unsafe {
        let merged_count = merge_topk_lists(
            input_indices_ptr,
            input_values_ptr,
            // max_positions_ptr,
            output_indices_ptr,
            output_values_ptr,
            // [1]
            output_token_ptr,
            thread_num,
            topk_size,
        );
        
        // Get max value directly from first element (merge sort results are ordered)
        let max_val = *output_values_ptr.add(0)
    
        // Calculate adjusted total sum (subtract max for numerical stability)
        let mut total_sum = f16::ZERO;
        for i in 0..thread_num {
            total_sum += (*sums_ptr.add(i))*(*input_values_ptr.add(i*topk_size) - max_val).exp();
        }
        
        // Use SIMD for softmax computation directly on output
        let values_vec = _mm512_loadu_ph(output_values_ptr);
        let max_vec = _mm512_set1_ph(max_val);
        
        // Apply softmax: subtract max and exp
        let shifted_vec = _mm512_sub_ph(values_vec, max_vec);
        let exp_vec = exp512(shifted_vec);
        
        // Use the adjusted total sum for normalization
        let sum_vec = _mm512_set1_ph(total_sum);
        
        // Normalize and store directly
        let normalized_vec = _mm512_div_ph(exp_vec, sum_vec);
        _mm512_storeu_ph(output_values_ptr, normalized_vec);
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;
    use super::*;

    #[test]
    fn test_topk_softmax_integration() {
        unsafe {
            let thread_num = 2;
            let topk_size = 3;
            
            // Input data
            let input_values = vec![
                f16::from_f32(2.0), f16::from_f32(1.0), f16::from_f32(0.5), // thread 0
                f16::from_f32(1.5), f16::from_f32(1.2), f16::from_f32(0.3), // thread 1
            ];
            let input_indices = vec![10, 30, 50, 20, 40, 60];
            let sums = vec![f16::ZERO; thread_num]; // not used in current implementation
            
            let mut max_positions = vec![0usize; thread_num];
            let mut output_values = vec![f16::ZERO; topk_size];
            let mut output_indices = vec![0usize; topk_size];
            
            topk_softmax(
                input_indices.as_ptr(),
                input_values.as_ptr(),
                sums.as_ptr(),
                max_positions.as_mut_ptr(),
                output_indices.as_mut_ptr(),
                output_values.as_mut_ptr(),
                thread_num,
                topk_size,
            );
            
            // Check that indices are correctly ordered
            assert_eq!(output_indices[0], 10); // highest value 2.0
            assert_eq!(output_indices[1], 20); // second highest 1.5
            assert_eq!(output_indices[2], 40); // third highest 1.2
            
            // Check that softmax values sum to approximately 1.0
            let sum: f32 = output_values.iter().map(|&x| x.to_f32()).sum();
            assert_ulps_eq!(sum, 1.0, max_ulps = 10);
            
            // Check that values are in descending order after softmax
            assert!(output_values[0] >= output_values[1]);
            assert!(output_values[1] >= output_values[2]);
        }
    }



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