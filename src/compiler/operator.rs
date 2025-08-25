//use std::arch::x86_64::_MM_EXCEPT_DENORM;
// use num_traits::{Float, FromPrimitive};
use std::ops::{Add, Sub, Div, Mul, AddAssign, Neg };
use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{neg_infinity::NegInfinity, exp::Exp};
use crate::kernel::generic::sigmoid::Sigmoid;

use crate::init::matmul_params::MatMulParams;
use crate::init::send_sync_ptr::{ConstPtr, MutPtr};
use super::map::lookup_rms_map::LookupRMSMap;
use super::map::rms_map::RMSMap;
// use super::map::softmax_map::SoftmaxMap;
use super::reduce::argmax_reduce::ArgmaxReduce;
use super::mul::mat_mul::MatMul;
use super::zip_map::add_rms_zip::AddRMSZipMap;
use super::zip_map::add_zip::AddZipMap;
use super::zip_map::complex_zip::ComplexZipMap;
use super::zip_map::silu_mul_zip::SiluZipMap;
// use super::mul::vec_mul::VecMul;
// use super::mul::col_mul::ColMul;
use super::mul::attention_mul::AttentionMul;

#[derive(Clone)]
pub enum Operator<T> {
    AddRMSZipMap(AddRMSZipMap<T>),
    AddZipMap(AddZipMap<T>),
    ArgmaxReduce(ArgmaxReduce<T>),
    AttentionMul(AttentionMul<T>),
    ComplexZip(ComplexZipMap<T>),
    LookupRMSMap(LookupRMSMap<T>),
    RMSMap(RMSMap<T>),
    SiluMulZipMap(SiluZipMap<T>),
    // SoftmaxMap(SoftmaxMap<T>),
    // VecMul(VecMul<T>),
    // ColMul(ColMul<T>),
    MatMul(MatMul<T>),
}

impl<T> Operator<T>
where T:
    Copy 
    + Default 
    + Sub<Output = T>
    + Neg<Output = T>
    + Exp
    + NegInfinity
    + Sigmoid<T>
    + Sqrt
{
    pub fn run(&self, batch_size: usize, position_index: usize, thread_id: usize) {
        match self {
            Self::AddRMSZipMap(operator) => {
                operator.run(batch_size, thread_id);
            }
            Self::AddZipMap(operator) => {
                operator.run(batch_size, position_index, thread_id);
            }
            Self::ArgmaxReduce(operator) => {
                operator.run(batch_size, position_index, thread_id);
            }, 
            Self::AttentionMul(operator) => {
                operator.run(batch_size, position_index, thread_id);
            }
            Self::ComplexZip(operator) => {
                operator.run(batch_size, position_index, thread_id);
            }
            Self::LookupRMSMap(operator) => {
                operator.run(batch_size, position_index, thread_id);
            }
            Self::RMSMap(operator) => {
                operator.run(batch_size, position_index, thread_id);
            }
            Self::SiluMulZipMap(operator) => {
                operator.run(batch_size, thread_id);
            }
              /*
            Self::SoftmaxMap(operator) => {
                operator.run(batch_size, position_index, thread_id);
            }
          
            Self::VecMul(operator) => {
                operator.run(batch_size, position_index, thread_id);
            },
            Self::ColMul(operator) => {
                operator.run(batch_size, position_index, thread_id);
            },
             */
            Self::MatMul(operator) => {
                operator.run(batch_size, position_index, thread_id);
            }

            _ => panic!(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::compiler::mul::chunk_attention::chunk_attention;
    // use crate::ptensor::chunk_colmul::chunk_colmul;
    use crate::compiler::map::chunk_map::chunk_map;
    use crate::compiler::mul::chunk_matmul::chunk_matmul;
    // use crate::ptensor::chunk_vecmul::chunk_vecmul;
    use crate::compiler::zip_map::chunk_zipmap::chunk_zipmap;
    use crate::ptensor::tensor_utils::{get_aligned_strides, get_broadcast_shape, get_strides};
    use approx::assert_ulps_eq;
    // use nom::sequence;
    use std::sync::{Arc, Barrier};
    use std::thread;

    /* 
    #[test]
    fn test_add_zip() {
        let shapes = vec![10, 18];
        let input_strides1 = vec![18, 1];
        let input_strides2 = vec![18, 1];
        let output_strides = vec![18, 1];
        let length = shapes.iter().product();
        let batch_size = 10;
        let position_size = 0;

        let input_data1: Vec<f32> = (0..=17).cycle().take(180).map(|x| x as f32).collect();
        let input_data2: Vec<f32> = vec![1.0; length];
        let results: Vec<f32> = (1..=18).cycle().take(180).map(|x| x as f32).collect();
        let mut output_data: Vec<f32> = vec![0.0; length];

        let chunks = chunk_zipmap(
            shapes,
            input_data1.as_ptr(),
            input_strides1,
            input_data2.as_ptr(),
            input_strides2,
            output_data.as_mut_ptr(),
            output_strides,
        );

        let thread_num: usize = num_cpus::get();
        let mut operator = Operator::AddZipMap(AddZipMap::new(18, thread_num));
        operator.set_zipmap_chunk(chunks);

        for i in 0..thread_num {
            operator.run(batch_size, position_size, i);
        }

        assert_ulps_eq!(output_data[0..180], results[0..180], max_ulps = 4);
        println!("{:?}", output_data);
    }*/

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
    }

    #[test]
    fn test_rms() {
        let shapes = vec![10, 18];
        let strides = vec![18, 1];
        let length = shapes.iter().product();
        let batch_size = 10;
        let position_size = 0;
        let cpu_num = num_cpus::get();

        let input_data: Vec<f32> = (1..=18).cycle().take(180).map(|x| x as f32).collect();
        let weight = [1.0f32; 180];
        let eps = 1e-6;
        let mut output_data: Vec<f32> = vec![0.0; length];

        let chunks = chunk_map(
            shapes,
            strides,
            input_data.as_ptr(),
            output_data.as_mut_ptr(),
        );
        let mut operator = Operator::RMSMap(RMSMap::new(18, weight.as_ptr(), eps, cpu_num));
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
        operator.set_map_chunk(chunks);
        let thread_num: usize = cpu_num;
        for i in 0..thread_num {
            operator.run(batch_size, 0, i);
        }

        assert_ulps_eq!(output_data[18..36], result, max_ulps = 4);
        println!("{:?}", output_data);
    }

    #[test]
    fn test_lookup_rms_map() {
        let batch_size = 10;
        let hidden_size = 18;
        let vocab_size = 10;
        let cpu_num = num_cpus::get();

        let shapes = vec![batch_size, hidden_size];
        let strides = vec![hidden_size, 1];
        let length = shapes.iter().product();
        let sequence_length = 16;
        let position = 0;

        let input_data: Vec<f32> = (1..=hidden_size)
            .cycle()
            .take(length)
            .map(|x| x as f32)
            .collect();
        let sequences = vec![1; sequence_length];
        let word_embedding: Vec<f32> = (1..=18)
            .cycle()
            .take(vocab_size * hidden_size)
            .map(|x| x as f32)
            .collect();

        let weight = vec![1.0f32; length];
        let eps = 1e-6;
        let mut output_data: Vec<f32> = vec![0.0; length];

        let chunks = chunk_map(
            shapes,
            strides,
            input_data.as_ptr(),
            output_data.as_mut_ptr(),
        );
        let mut Operator = Operator::LookupRMSMap(LookupRMSMap::new(
            hidden_size,
            weight.as_ptr(),
            eps,
            cpu_num,
            word_embedding.as_ptr(),
            sequences.as_ptr(),
            hidden_size,
            batch_size,
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
        Operator.set_map_chunk(chunks);

        let thread_num: usize = cpu_num;
        for i in 0..thread_num {
            Operator.run(batch_size, position, i);
        }

        assert_ulps_eq!(output_data[18..36], result, max_ulps = 4);
        println!("{:?}", output_data);
    }

    /*
    #[test]
    fn test_silu() {
        let batch_size = 10;
        let hidden_size = 19;
        let shapes = vec![batch_size, hidden_size];
        let input_strides1 = get_strides(&shapes);
        let input_strides2 = input_strides1.clone();
        let output_strides = input_strides1.clone();
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
        .repeat(10);
        let input_data2: [f32; 190] = [1.0; 190];
        let mut output_data: Vec<f32> = vec![0.0; length];
        let chunks = chunk_zipmap(
            shapes,
            input_data1.as_ptr(),
            input_strides1,
            input_data2.as_ptr(),
            input_strides2,
            output_data.as_mut_ptr(),
            output_strides,
        );
        let thread_num: usize = num_cpus::get();
        let mut operator = Operator::SiluMulZipMap(SiluZipMap::new(hidden_size, thread_num));
        operator.set_zipmap_chunk(chunks);

        for i in 0..thread_num {
            operator.run(batch_size, 1usize, i);
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
        .repeat(10);
        assert_ulps_eq!(output_data[..], result, max_ulps = 4);
    } */

    /* 
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
    }*/

    #[test]
    fn test_matmul_batch_size_1() {
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

        let params = MatMulParams {
            a_row: max_batch_size,
            b_row: hidden_size,
            column: hidden_size,
            a_row_step_macro: 1,
            b_row_step_macro: 1,
            column_step_macro: hidden_size,
            a_row_step_micro: 1,
            b_row_step_micro: 1,
        };

        let chunks = chunk_matmul(data1.as_ptr(), data2.as_ptr(), data3.as_mut_ptr(), &params);

        let thread_num: usize = num_cpus::get();
        let barrier = Barrier::new(thread_num);
        let barrier_arc = Arc::new(barrier);
        let mut operator = Operator::MatMul(MatMul::<f32>::new(
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
    fn test_matmul_batch_size_1_sequence() {
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

        let params = MatMulParams {
            a_row: max_batch_size,
            b_row: hidden_size,
            column: hidden_size,
            a_row_step_macro: 1,
            b_row_step_macro: 1,
            column_step_macro: hidden_size,
            a_row_step_micro: 1,
            b_row_step_micro: 1,
        };

        let chunks = chunk_matmul(data1.as_ptr(), data2.as_ptr(), data3.as_mut_ptr(), &params);

        let thread_num: usize = num_cpus::get();
        let barrier = Barrier::new(thread_num);
        let barrier_arc = Arc::new(barrier);
        let mut operator = Operator::MatMul(MatMul::<f32>::new(
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
    fn test_complexmul_with_broadcast() {
        let head_size = 34;
        let head_num = 10;
        let batch_size = 10;
        let sequence_length = 10;

        let shape1 = vec![batch_size, head_num, head_size];
        let shape2 = vec![sequence_length, 1, 1, head_size];
        let broadcast_shape = get_broadcast_shape(&shape1, &shape2);

        let length: usize = broadcast_shape.iter().product();
        let input_strides1 = get_aligned_strides(&shape1, &broadcast_shape);
        let input_strides2 = get_aligned_strides(&shape2, &broadcast_shape);
        let output_strides = get_strides(&broadcast_shape);

        let length1: usize = shape1.iter().product();
        let length2: usize = shape2.iter().product();
        let input_data1: Vec<f32> = (1..=34).cycle().take(length1).map(|x| x as f32).collect();
        let input_data2: Vec<f32> = (1..=34).cycle().take(length2).map(|x| x as f32).collect();
        let mut output_data: Vec<f32> = vec![0.0; length];

        let expected: Vec<f32> = vec![
            -3.0, 4.0, -7.0, 24.0, -11.0, 60.0, -15.0, 112.0, -19.0, 180.0, -23.0, 264.0, -27.0,
            364.0, -31.0, 480.0, -35.0, 612.0, -39.0, 760.0, -43.0, 924.0, -47.0, 1104.0, -51.0,
            1300.0, -55.0, 1512.0, -59.0, 1740.0, -63.0, 1984.0, -67.0, 2244.0,
        ];

        let chunks = chunk_zipmap(
            broadcast_shape,
            input_data1.as_ptr(),
            input_strides1,
            input_data2.as_ptr(),
            input_strides2,
            output_data.as_mut_ptr(),
            output_strides,
        );

        let thread_num: usize = num_cpus::get();
        let mut operator = Operator::ComplexZip(ComplexZipMap::<f32>::new(
            head_size, head_num, batch_size, thread_num,
        ));
        operator.set_zipmap_chunk(chunks);

        for i in 0..thread_num {
            operator.run(batch_size, 1, i);
        }

        assert_eq!(output_data[3434..3468], expected);
    }

    /*
    #[test]
    fn test_chunk_vec() {
        let head_size = 128;
        let head_num = 64;
        let batch_size = 16;
        let sequence_length = 256;

        let q_shape = vec![batch_size, head_num, 1, head_size];
        let q_size = q_shape.iter().product();
        let q_data: Vec<f32> = vec![1.0; q_size];
        let q_strides = get_strides(&q_shape);

        let k_shape = vec![batch_size, head_num, sequence_length, head_size];
        let k_size = k_shape.iter().product();
        let k_data: Vec<f32> = vec![1.0; k_size];
        let k_strides = get_strides(&k_shape);

        let s_shape = vec![batch_size, head_num, 1, sequence_length];
        let s_size = s_shape.iter().product();
        let mut s_data: Vec<f32> = vec![0.0; s_size];
        let s_strides = get_strides(&s_shape);

        let result = vec![head_size as f32; s_size];

        let chunks = chunk_vecmul(
            q_data.as_ptr(),
            q_shape,
            q_strides,
            k_data.as_ptr(),
            k_shape,
            k_strides,
            s_data.as_mut_ptr(),
            s_shape,
            s_strides,
        );

        let thread_num: usize = num_cpus::get();
        let mut operator = Operator::VecMul(VecMul::<f32>::new(head_size, head_num, sequence_length, thread_num));
        operator.set_zipmap_chunk(chunks);
        for i in 0..thread_num {
            operator.run(batch_size, sequence_length, i);
        }

        assert_ulps_eq!(s_data[..], result[..], max_ulps = 4);
    }

    #[test]
    fn test_col_mul() {
        let head_size = 128;
        let head_num = 64;
        let batch_size = 16;
        let sequence_length = 256;

        let shape1 = vec![batch_size, head_num, sequence_length];
        let size1 = shape1.iter().product();
        let data1 = vec![1.0; size1];
        let strides1 = get_strides(&shape1);

        let shape2 = vec![batch_size, head_num, sequence_length, head_size];
        let size2 = shape2.iter().product();
        let data2 = vec![1.0; size2];
        let strides2 = get_strides(&shape2);

        let shape3 = vec![batch_size, head_num, head_size];
        let size3 = shape3.iter().product();
        let mut data3 = vec![0.0; size3];
        let strides3 = get_strides(&shape3);

        let result = vec![sequence_length as f32; size3];

        let chunks = chunk_colmul(
            data1.as_ptr(),
            shape1,
            strides1,
            data2.as_ptr(),
            shape2,
            strides2,
            data3.as_mut_ptr(),
            shape3,
            strides3,
        );

        let thread_num: usize = num_cpus::get();
        let mut operator = Operator::ColMul(ColMul::<f32>::new(head_size, head_num, thread_num));
        operator.set_zipmap_chunk(chunks);
        for i in 0..thread_num {
            operator.run(batch_size, sequence_length, i);
        }
        assert_ulps_eq!(data3[..], result[..], max_ulps = 4);
    }*/
}
