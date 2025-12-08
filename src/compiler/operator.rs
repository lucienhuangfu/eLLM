//use std::arch::x86_64::_MM_EXCEPT_DENORM;
// use num_traits::{Float, FromPrimitive};
use crate::kernel::generic::sigmoid::Sigmoid;
use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{exp::Exp, neg_infinity::NegInfinity};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use super::map::experts_softmax_norm::ExpertsSoftmaxNorm;
use super::map::left_vector::LiftVector;
use super::map::lookup_rms_map::LookupRMSMap;
use super::map::rms_map::RMSMap;
use super::map::topk_softmax::TopKSoftmax;
// Add missing imports for zip map operations
use super::mul::matmul::Matmul;
use super::mul::matmul3::Matmul3;
use super::mul::matmul_add::MatmulAdd;
use super::mul::matmul_silu_mul_matmul::MatmulSilu;
use super::mul::matmul_topk::MatmulTopK;
use super::zip_map::add_rms_zip::AddRMSZipMap;
use super::zip_map::add_zip::AddZipMap;
use super::zip_map::complex_zip::ComplexZipMap;
use super::zip_map::silu_mul_zip::SiluMulZipMap;
// use super::map::softmax_map::SoftmaxMap;
// use super::reduce::argmax_reduce::ArgmaxReduce;
use super::mul::attention::Attention;
use super::mul::experts_matmul_mul::ExpertsMatmulMul;
use super::mul::experts_matmul_silu_mul_matmul::ExpertsMatmulSilu;
use super::mul::experts_merge_add::ExpertsMergeAdd;
// use crate::init::matmul_params::MatmulParams;
// use crate::init::send_sync_ptr::{ConstPtr, MutPtr};

#[derive(Clone)]
pub enum Operator<T> {
    AddRMSZipMap(AddRMSZipMap<T>),
    AddZipMap(AddZipMap<T>),

    Attention(Attention<T>),
    ComplexZipMap(ComplexZipMap<T>),
    ExpertsMatmulMul(ExpertsMatmulMul<T>),
    ExpertsMatmulSiluMulMatmul(ExpertsMatmulSilu<T>),
    ExpertsMergeAdd(ExpertsMergeAdd<T>),
    ExpertsSoftmaxNorm(ExpertsSoftmaxNorm<T>),
    LiftVector(LiftVector<T>),
    LookupRMSMap(LookupRMSMap<T>),
    Matmul(Matmul<T>),
    // Matmul3(Matmul3<T>),
    MatmulAdd(MatmulAdd<T>),
    MatmulSiluMulMatmul(MatmulSilu<T>),
    MatmulTopK(MatmulTopK<T>),
    RMSMap(RMSMap<T>),
    SiluMulZipMap(SiluMulZipMap<T>),
    // SoftmaxMap(SoftmaxMap<T>),
    TopKSoftmax(TopKSoftmax<T>),
    // ArgmaxReduce(ArgmaxReduce<T>),
}

impl<T> Operator<T>
where
    T: Copy
        + Default
        + Sub<Output = T>
        + Neg<Output = T>
        + Exp
        + NegInfinity
        + Sigmoid<T>
        + Sqrt
        + AddAssign,
{
    pub fn run(&self, batch_size: usize, decode_size: usize, cpu_num: usize, thread_id: usize) {
        match self {
            /*
            Self::AddRMSZipMap(operator) => {
                operator.run(batch_size, decode_size, cpu_num, thread_id);
            }
            Self::AddZipMap(operator) => {
                operator.run(batch_size, decode_size, cpu_num, thread_id);
            }
                        Self::ComplexZipMap(operator) => {
                operator.run(batch_size, decode_size, cpu_num, thread_id);
            }

                        Self::MatmulSiluMulMatmul(operator) => {
                operator.run(batch_size, decode_size, cpu_num, thread_id);
            }

                        Self::SiluMulZipMap(operator) => {
                operator.run(batch_size, decode_size, cpu_num, thread_id);
            }



            Self::ArgmaxReduce(operator) => {
                operator.run(batch_size, thread_id);
            },
            Self::SoftmaxMap(operator) => {
                operator.run(batch_size, thread_id);
            }
             */
            Self::Attention(operator) => {
                operator.run(batch_size, decode_size, cpu_num, thread_id);
            }

            Self::ExpertsMatmulMul(operator) => {
                operator.run(batch_size, decode_size, cpu_num, thread_id);
            }
            Self::ExpertsMatmulSiluMulMatmul(operator) => {
                operator.run(batch_size, decode_size, cpu_num, thread_id);
            }
            Self::ExpertsMergeAdd(operator) => {
                operator.run(batch_size, decode_size, cpu_num, thread_id);
            }
            Self::ExpertsSoftmaxNorm(operator) => {
                operator.run(batch_size, decode_size, cpu_num, thread_id);
            }
            Self::LiftVector(operator) => {
                operator.run(batch_size, decode_size, cpu_num, thread_id);
            }
            Self::LookupRMSMap(operator) => {
                operator.run(batch_size, decode_size, cpu_num, thread_id);
            }
            Self::Matmul(operator) => {
                operator.run(batch_size, decode_size, cpu_num, thread_id);
            }
            /*
            Self::Matmul3(operator) => {
                operator.run(
                    batch_size,
                    decode_size,
                    cpu_num,
                    thread_id,
                );
            } */
            Self::MatmulAdd(operator) => {
                operator.run(batch_size, decode_size, cpu_num, thread_id);
            }

            Self::MatmulTopK(operator) => {
                operator.run(batch_size, decode_size, cpu_num, thread_id);
            }
            Self::RMSMap(operator) => {
                operator.run(batch_size, decode_size, cpu_num, thread_id);
            }

            Self::TopKSoftmax(operator) => {
                operator.run(batch_size, decode_size, cpu_num, thread_id);
            }

            _ => panic!(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::init::record::{Phase, TokenRecord, UserRecord};
    use approx::assert_ulps_eq;
    use rand::seq;
    // use crate::ptensor::tensor_utils::{get_aligned_strides, get_broadcast_shape, get_strides};
    // use std::sync::{Arc, Barrier};
    // use std::thread;

    #[test]
    fn test_add_zip() {
        // let sequence_chunk_size = 1;
        let batch_size = 10;
        let head_num = 3;
        let head_size = 6;

        let shapes = vec![batch_size, head_num, head_size];
        let length = shapes.iter().product();

        let input_data1: Vec<f32> = (0..=17).cycle().take(length).map(|x| x as f32).collect();
        let input_data2: Vec<f32> = vec![1.0; length];
        let results: Vec<f32> = (1..=18).cycle().take(length).map(|x| x as f32).collect();
        let mut output_data: Vec<f32> = vec![0.0; length];

        let thread_num: usize = num_cpus::get();
        let operator = Operator::AddZipMap(AddZipMap::new(
            input_data1.as_ptr(),
            input_data2.as_ptr(),
            output_data.as_mut_ptr(),
            // batch_size,
            head_num,
            head_size,
        ));

        // let position_index = 0;
        for i in 0..thread_num {
            operator.run(batch_size, 1, thread_num, i);
        }

        assert_ulps_eq!(output_data[0..180], results[0..180], max_ulps = 4);
        println!("{:?}", output_data);
    }

    #[test]
    fn test_rms() {
        // let sequence_chunk_size = 1;
        let batch_size = 10;
        let hidden_size = 18;
        // let position_index = 0;
        let cpu_num = num_cpus::get();

        let shapes = vec![batch_size, hidden_size];
        let length = shapes.iter().product();
        let input_data: Vec<f32> = (1..=hidden_size)
            .cycle()
            .take(length)
            .map(|x| x as f32)
            .collect();
        let weight = [1.0f32; 18];
        let eps = 1e-6;
        let mut output_data: Vec<f32> = vec![0.0; length];

        let operator = Operator::RMSMap(RMSMap::new(
            input_data.as_ptr(),
            output_data.as_mut_ptr(),
            // batch_size,
            hidden_size,
            // weight.as_ptr(),
            eps,
            false,
        ));

        let result = [
            0.09238425642251968,
            0.18476851284503937,
            0.27715277671813965,
            0.36953702569007874,
            0.4619212746620178,
            0.5543055534362793,
            0.646689772605896,
            0.7390740513801575,
            0.831458330154419,
            0.9238425493240356,
            1.0162267684936523,
            1.1086111068725586,
            1.2009953260421753,
            1.293379545211792,
            1.3857638835906982,
            1.478148102760315,
            1.5705323219299316,
            1.662916660308838,
        ];
        let thread_num: usize = cpu_num;
        for i in 0..thread_num {
            operator.run(batch_size, 1, cpu_num, i);
        }
        assert_ulps_eq!(output_data[18..36], result, max_ulps = 4);
        println!("{:?}", output_data);
    }

    #[test]
    fn test_complexmul() {
        let sequence_length = 10;
        // let sequence_chunk_size = 8;
        let batch_size = 80;
        let head_num = 10;
        let head_size = 34;

        let shape1 = vec![batch_size, head_num, head_size];
        let shape2 = vec![sequence_length, head_size];

        let length1: usize = shape1.iter().product();
        let length2: usize = shape2.iter().product();
        let length = length1;
        let input_data1: Vec<f32> = (1..=head_size)
            .cycle()
            .take(length1)
            .map(|x| x as f32)
            .collect();
        let input_data2: Vec<f32> = (1..=head_size)
            .cycle()
            .take(length2)
            .map(|x| x as f32)
            .collect();
        let mut output_data: Vec<f32> = vec![0.0; length];

        let expected: Vec<f32> = vec![
            -3.0, 4.0, -7.0, 24.0, -11.0, 60.0, -15.0, 112.0, -19.0, 180.0, -23.0, 264.0, -27.0,
            364.0, -31.0, 480.0, -35.0, 612.0, -39.0, 760.0, -43.0, 924.0, -47.0, 1104.0, -51.0,
            1300.0, -55.0, 1512.0, -59.0, 1740.0, -63.0, 1984.0, -67.0, 2244.0,
        ];

        let thread_num: usize = num_cpus::get();
        let mut operator = Operator::ComplexZipMap(ComplexZipMap::<f32>::new(
            input_data1.as_ptr(),
            input_data2.as_ptr(),
            output_data.as_mut_ptr(),
            // sequence_chunk_size,
            //batch_size,
            head_num,
            head_size,
            false,
        ));

        for i in 0..thread_num {
            operator.run(batch_size, 1, thread_num, i);
        }

        assert_eq!(output_data[3434..3468], expected);
    }

    #[test]
    fn test_silu() {
        // let sequence_chunk_size = 8;
        let batch_size = 80;
        // let hidden_size = 19;
        let head_num = 1;
        let head_size = 19;

        let shapes = vec![batch_size, head_num, head_size];

        let length = shapes.iter().product();
        let input_data1: Vec<f32> = vec![
            2.1671206951141357,
            1.4490455389022827,
            -2.002431631088257,
            0.5662149786949158,
            0.3909946382045746,
            0.9437483549118042,
            -0.37030690908432007,
            0.7542704939842224,
            0.5875813961029053,
            1.6026240587234497,
            2.2485475540161133,
            -0.6622593402862549,
            -0.0015666020335629582,
            -0.5069465041160583,
            -0.37254711985588074,
            0.4420417249202728,
            -0.9305257201194763,
            0.5145581364631653,
            0.6260590553283691,
        ]
        .repeat(batch_size);
        // let input_data2: [f32; 190] = [1.0; 190];

        let mut input_data2: Vec<f32> = vec![1.0; length];
        let mut output_data: Vec<f32> = vec![0.0; length];

        let thread_num: usize = num_cpus::get();
        let mut operator = Operator::SiluMulZipMap(SiluMulZipMap::new(
            input_data1.as_ptr(),
            input_data2.as_ptr(),
            output_data.as_mut_ptr(),
            // batch_size,
            head_num,
            head_size,
        ));

        for i in 0..thread_num {
            operator.run(batch_size, 1, thread_num, i);
        }
        let result = vec![
            1.9444659948349,
            1.1735117435455322,
            -0.23818494379520416,
            0.36118248105049133,
            0.23323695361614227,
            0.6793630719184875,
            -0.15125809609889984,
            0.5129857659339905,
            0.3777032196521759,
            1.3339999914169312,
            2.033867835998535,
            -0.22532200813293457,
            -0.0007826874498277903,
            -0.1905660629272461,
            -0.15197153389453888,
            0.269090861082077,
            -0.2631694972515106,
            0.32204875349998474,
            0.4079371392726898,
        ]
        .repeat(batch_size);
        assert_ulps_eq!(output_data[..], result, max_ulps = 4);
    }

    #[test]
    fn test_experts_softmax_norm() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            println!("AVX2 not supported, skipping test.");
            return;
        }

        // let sequence_chunk_size = 1;
        let batch_size = 2;
        let num_experts = 16;
        let num_topk = 4;
        let num_tokens = batch_size;

        let input_data1: Vec<f32> = vec![
            0.5, -1.0, 2.5, 3.0, 7.5, 6.5, -2.0, 10.0, 4.0, 8.0, 1.0, 9.5, -3.5, 5.5, 11.0, -0.25,
        ];
        let input_data2: Vec<f32> = vec![
            -0.5, 0.25, 3.75, -2.0, 6.0, 1.75, -4.25, 2.5, 0.0, 5.25, -1.25, 4.0, 3.0, -3.5, 7.5,
            2.25,
        ];
        let mut input_data = Vec::new();
        input_data.extend_from_slice(&input_data1);
        input_data.extend_from_slice(&input_data2);

        let mut experts_indicator = vec![false; num_experts];
        let mut indice_ptr = vec![false; num_experts * num_tokens];
        let mut weight_ptr = vec![0.0f32; num_experts * num_tokens];
        let mut topk_indices_ptr = vec![0usize; num_topk * num_tokens];

        let operator = Operator::ExpertsSoftmaxNorm(ExpertsSoftmaxNorm::<f32>::new(
            input_data.as_ptr(),
            experts_indicator.as_mut_ptr(),
            indice_ptr.as_mut_ptr(),
            weight_ptr.as_mut_ptr(),
            topk_indices_ptr.as_mut_ptr(),
            // sequence_chunk_size,
            batch_size,
            num_experts,
            num_topk,
            false,
        ));

        let thread_num = 1;
        let thread_id = 0;
        operator.run(batch_size, 1, thread_num, thread_id);

        // Verification for token 0
        let mut expected1: Vec<(usize, f32)> = input_data1.iter().copied().enumerate().collect();
        expected1.sort_by(|a, b| b.1.total_cmp(&a.1));
        let max_val1 = input_data1
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let denom1: f32 = input_data1.iter().map(|v| (v - max_val1).exp()).sum();

        for i in 0..num_topk {
            let (idx, val) = expected1[i];
            let prob = (val - max_val1).exp() / denom1;
            assert!(experts_indicator[idx]);
            let offset = idx * num_tokens + 0;
            assert!(indice_ptr[offset]);
            assert_ulps_eq!(weight_ptr[offset], prob, max_ulps = 4);
        }

        // Verification for token 1
        let mut expected2: Vec<(usize, f32)> = input_data2.iter().copied().enumerate().collect();
        expected2.sort_by(|a, b| b.1.total_cmp(&a.1));
        let max_val2 = input_data2
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let denom2: f32 = input_data2.iter().map(|v| (v - max_val2).exp()).sum();

        for i in 0..num_topk {
            let (idx, val) = expected2[i];
            let prob = (val - max_val2).exp() / denom2;
            assert!(experts_indicator[idx]);
            let offset = idx * num_tokens + 1;
            assert!(indice_ptr[offset]);
            assert_ulps_eq!(weight_ptr[offset], prob, max_ulps = 4);
        }
    }

    #[test]
    fn test_topk_softmax() {
        let batch_size = 2;
        let topk_size = 8;
        let thread_num = 4;
        let sequence_length = 2;
        let eos_id = 100;
        // let position_begin = 0;
        // let position_interval = 1;

        let total_candidates_per_item = topk_size * thread_num;
        let input_len = batch_size * total_candidates_per_item;

        let mut input_values = Vec::<f32>::with_capacity(input_len);
        let mut input_indices = Vec::<usize>::with_capacity(input_len);
        let mut token_records = Vec::with_capacity(batch_size);
        let mut user_records = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            token_records.push(TokenRecord {
                token_id: 0,
                batch_index: i,
                position_index: 0,
            });
            user_records.push(UserRecord {
                sequence_index: i,
                kv_index: 0,
                phase: Phase::Decode,
            });
            for j in 0..total_candidates_per_item {
                input_values.push(5.0 - (j as f32 * 0.1) - (i as f32));
                input_indices.push(i * 1000 + j);
            }
        }

        let sums = vec![0.0f32; batch_size];
        let mut output_values = vec![0.0f32; batch_size * topk_size];
        let mut output_indices = vec![0usize; batch_size * topk_size];
        let mut output_sequences = vec![0usize; batch_size * sequence_length];

        let operator = Operator::TopKSoftmax(TopKSoftmax::<f32>::new(
            input_indices.as_ptr(),
            input_values.as_ptr(),
            sums.as_ptr(),
            token_records.as_ptr(),
            user_records.as_mut_ptr(),
            output_indices.as_mut_ptr(),
            output_values.as_mut_ptr(),
            output_sequences.as_mut_ptr(),
            batch_size,
            topk_size,
            eos_id,
        ));

        for i in 0..thread_num {
            operator.run(batch_size, batch_size, thread_num, i);
        }

        for i in 0..batch_size {
            let item_input_values =
                &input_values[i * total_candidates_per_item..(i + 1) * total_candidates_per_item];
            let item_input_indices =
                &input_indices[i * total_candidates_per_item..(i + 1) * total_candidates_per_item];

            let mut paired: Vec<_> = item_input_values
                .iter()
                .copied()
                .zip(item_input_indices.iter().copied())
                .collect();
            paired.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            let topk = &paired[..topk_size];
            let max_val = topk[0].0;
            let denom: f32 = topk.iter().map(|(v, _)| (v - max_val).exp()).sum();

            let expected_probs: Vec<f32> = topk
                .iter()
                .map(|(v, _)| (v - max_val).exp() / denom)
                .collect();
            let expected_indices: Vec<usize> = topk.iter().map(|(_, idx)| *idx).collect();

            let output_vals_slice = &output_values[i * topk_size..(i + 1) * topk_size];
            let output_idx_slice = &output_indices[i * topk_size..(i + 1) * topk_size];

            assert_ulps_eq!(output_vals_slice, expected_probs.as_slice(), max_ulps = 4);
            assert_eq!(output_idx_slice, expected_indices.as_slice());
            assert_eq!(output_sequences[batch_size + i], expected_indices[0]);
        }
    }

    /*
    #[test]
    fn test_lookup_rms_map() {
        let sequence_chunk_size = 1;
        let batch_size = 10;
        let hidden_size = 18;
        let vocab_size = 10;
        let cpu_num = num_cpus::get();
        let sequence_length = 16;
        let position = 0;

        let shapes = vec![sequence_chunk_size, batch_size, hidden_size];
        // let strides = vec![hidden_size, 1];
        let length = shapes.iter().product();

        let input_data: Vec<f32> = (1..=hidden_size)
            .cycle()
            .take(length)
            .map(|x| x as f32)
            .collect();
        let mut sequences = vec![1; sequence_length];
        let word_embedding: Vec<f32> = (1..=18)
            .cycle()
            .take(vocab_size * hidden_size)
            .map(|x| x as f32)
            .collect();

        let weight = vec![1.0f32; hidden_size];
        let eps = 1e-6;
        let mut output_data: Vec<f32> = vec![0.0; length];

        let mut Operator = Operator::LookupRMSMap(LookupRMSMap::new(
            sequences.as_mut_ptr(),
            output_data.as_mut_ptr(),
            batch_size,
            hidden_size,
            word_embedding.as_ptr(),
            weight.as_ptr(),
            eps,
        ));
        let result = [
            0.09238425642251968,
            0.18476851284503937,
            0.27715277671813965,
            0.36953702569007874,
            0.4619212746620178,
            0.5543055534362793,
            0.646689772605896,
            0.7390740513801575,
            0.831458330154419,
            0.9238425493240356,
            1.0162267684936523,
            1.1086111068725586,
            1.2009953260421753,
            1.293379545211792,
            1.3857638835906982,
            1.478148102760315,
            1.5705323219299316,
            1.662916660308838,
        ];

        let thread_num: usize = cpu_num;
        for i in 0..thread_num {
            Operator.run(position, sequence_chunk_size, batch_size, thread_num, i);
        }

        assert_ulps_eq!(output_data[18..36], result, max_ulps = 4);
        println!("{:?}", output_data);
    }

    #[test]
    fn test_softmax_map() {
        let batch_size = 10;
        let head_num = 10;
        let sequence_size = 18;
        let shapes = vec![batch_size, head_num, sequence_size];
        let strides = get_strides(&shapes);
        let length = shapes.iter().product();
        let position_index = 18;
        let cpu_num = num_cpus::get();

        let input_data: Vec<f32> = (1..=18).cycle().take(1800).map(|x| x as f32).collect();
        let mut output_data: Vec<f32> = vec![0.0; length];

        let chunks = chunk_map(
            shapes,
            strides,
            input_data.as_ptr(),
            output_data.as_mut_ptr(),
        );
        let mut operator = Operator::SoftmaxMap(SoftmaxMap::new(head_num, cpu_num));
        let result = vec![
            0.0012586231,
            0.0017267587,
            0.0023690138,
            0.0032501512,
            0.0044590216,
            0.006117522,
            0.00839289,
            0.011514563,
            0.015797319,
            0.02167302,
            0.02973414,
            0.040793534,
            0.0559664,
            0.0767827,
            0.105341434,
            0.14452243,
            0.19827652,
            0.27202395,
        ];
        operator.set_map_chunk(chunks);
        let thread_num: usize = cpu_num;
        for i in 0..thread_num {
            operator.run(batch_size, position_index, i);
        }
        assert_ulps_eq!(output_data[0..18], result, max_ulps = 4);
    }

    #[test]
    fn test_Matmul_batch_size_1() {
        let max_batch_size = 8;
        let hidden_size = 16;

        let shape1 = vec![max_batch_size, hidden_size];
        let size1 = shape1.iter().product();
        let data1: Vec<f32> = vec![1.0; size1];

        let shape2 = vec![hidden_size, hidden_size];
        let size2 = shape2.iter().product();
        let data2: Vec<f32> = vec![1.0; size2];

        let output_shape = vec![max_batch_size, hidden_size];
        let size3 = output_shape.iter().product();
        let mut data3: Vec<f32> = vec![0.0; size3];

        let mut result = vec![0.0 as f32; size3];

        for i in 0..hidden_size {
            result[i] = hidden_size as f32;
        }

        let params = MatmulParams {
            a_row: max_batch_size,
            b_row: hidden_size,
            column: hidden_size,
            a_row_step_macro: 1,
            b_row_step_macro: 1,
            column_step_macro: hidden_size,
            a_row_step_micro: 1,
            b_row_step_micro: 1,
        };

        let chunks = chunk_Matmul(data1.as_ptr(), data2.as_ptr(), data3.as_mut_ptr(), &params);

        let thread_num: usize = num_cpus::get();
        let barrier = Barrier::new(thread_num);
        let barrier_arc = Arc::new(barrier);
        let mut operator = Operator::Matmul(Matmul::<f32>::new(
            max_batch_size,
            hidden_size,
            hidden_size,
            1,
            1,
            hidden_size,
            1,
            1,
            1,
            thread_num,
            barrier_arc,
        ));
        operator.set_zipmap_chunk(chunks);

        let position_index: usize = 1;
        for i in 0..thread_num {
            operator.run(1, position_index, i);
        }

        assert_ulps_eq!(data3[..], result[..], max_ulps = 4);
    }

    #[test]
    fn test_Matmul_batch_size_1_sequence() {
        let max_batch_size = 8;
        let hidden_size = 16;
        let sequence_length = 16;
        let position_index = 4;

        let shape1 = vec![max_batch_size, hidden_size];
        let size1 = shape1.iter().product();
        let data1: Vec<f32> = vec![1.0; size1];

        let shape2 = vec![hidden_size, hidden_size];
        let size2 = shape2.iter().product();
        let data2: Vec<f32> = vec![1.0; size2];

        let output_shape = vec![sequence_length, max_batch_size, hidden_size];
        let size3 = output_shape.iter().product();
        let mut data3: Vec<f32> = vec![0.0; size3];

        let mut result = vec![0.0 as f32; size3];
        let offset = position_index * max_batch_size * hidden_size;

        for i in 0..hidden_size {
            result[i + offset] = hidden_size as f32;
        }

        let params = MatmulParams {
            a_row: max_batch_size,
            b_row: hidden_size,
            column: hidden_size,
            a_row_step_macro: 1,
            b_row_step_macro: 1,
            column_step_macro: hidden_size,
            a_row_step_micro: 1,
            b_row_step_micro: 1,
        };

        let chunks = chunk_Matmul(data1.as_ptr(), data2.as_ptr(), data3.as_mut_ptr(), &params);

        let thread_num: usize = num_cpus::get();
        let barrier = Barrier::new(thread_num);
        let barrier_arc = Arc::new(barrier);
        let mut operator = Operator::Matmul(Matmul::<f32>::new(
            max_batch_size,
            hidden_size,
            hidden_size,
            1,
            1,
            hidden_size,
            1,
            1,
            sequence_length,
            thread_num,
            barrier_arc,
        ));
        operator.set_zipmap_chunk(chunks);

        for i in 0..thread_num {
            operator.run(1, position_index, i);
        }

        assert_ulps_eq!(data3[..], result[..], max_ulps = 4);
    }




        #[test]
    fn test_attention_mul() {
        let head_size = 128;
        let head_num = 64;
        let batch_size = 16;
        let sequence_length = 256;

        let shape1 = vec![batch_size, head_num, head_size];
        let size1 = shape1.iter().product();
        let data1 = vec![1.0; size1];
        let strides1 = get_strides(&shape1);

        let shape2 = vec![sequence_length, batch_size, head_num, head_size];
        let strides2 = get_strides(&shape2);
        let size2 = shape2.iter().product();
        let data2 = vec![1.0; size2];

        let _shape2 = vec![batch_size, head_num, sequence_length, head_size];
        let _strides2 = vec![strides2[1], strides2[2], strides2[0], strides2[3]];

        let shape3 = vec![batch_size, sequence_length, head_num, head_size];
        let size3 = shape3.iter().product();
        let data3 = vec![1.0; size3];

        let shape4 = vec![batch_size, head_num, head_size];
        let size4 = shape4.iter().product();
        let mut data4 = vec![0.0; size4];
        let strides4 = get_strides(&shape4);

        let result = vec![1.0; size4];

        let tasks = chunk_attention(
            data1.as_ptr(),
            shape1,
            strides1,
            data2.as_ptr(),
            _shape2,
            _strides2.clone(),
            data3.as_ptr(),
            data4.as_mut_ptr(),
            shape4,
            strides4,
        );

        let thread_num: usize = num_cpus::get();
        let mut operator = Operator::AttentionMul(AttentionMul::<f32>::new(
            head_size,
            head_num,
            _strides2[2],
            1.0,
            thread_num,
        ));
        operator.set_attention_chunk(tasks);

        for i in 0..thread_num {
            operator.run(batch_size, sequence_length, i);
        }

        assert_ulps_eq!(data4[..], result[..], max_ulps = 4);
    }*/
}
