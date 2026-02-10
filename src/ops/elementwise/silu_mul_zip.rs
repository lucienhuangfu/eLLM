use std::f16;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::ops::traits::zip_map_trait::ZipMapTrait;
use crate::ops::assign::assign;
use crate::common::send_sync_ptr::{ConstPtr, MutPtr};
use crate::kernel;
use crate::num_traits::Sigmoid;

#[derive(Clone)]
pub struct SiluMulZipMap<T> {
    // chunks: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>,
    ptr1: ConstPtr<T>,
    ptr2: ConstPtr<T>,
    output_ptr: MutPtr<T>,
    // max_batch_size: usize,
    head_num: usize,
    head_size: usize,
    // cpu_num: usize
}

impl<T> SiluMulZipMap<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>
        + Sigmoid,
{
    pub fn new(
        //chunks: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>,
        ptr1: *const T,
        ptr2: *const T,
        output_ptr: *mut T,
        // max_batch_size: usize,
        head_num: usize,
        head_size: usize,
        // cpu_num: usize
    ) -> Self {
        Self {
            // chunks: vec![],
            ptr1: ConstPtr { ptr: ptr1 },
            ptr2: ConstPtr { ptr: ptr2 },
            output_ptr: MutPtr { ptr: output_ptr },
            // max_batch_size: max_batch_size,
            head_num: head_num,
            head_size: head_size,
            // cpu_num: cpu_num
        }
    }

    // pub fn set_chunk(&mut self, chunks: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>) {
    //    self.chunks = chunks;
    // }

    pub fn run(&self, prefill_size: usize, cpu_num: usize, thread_id: usize) {
        let total_len = prefill_size * self.head_num;
        if let Some((begin, end)) = assign(total_len, cpu_num, thread_id) {
            println!("thread_id: {}, begin: {}, end: {}, ", thread_id, begin, end,);

            let mut ptr1 = self.ptr1.ptr;
            let mut ptr2 = self.ptr2.ptr;
            let mut output_ptr = self.output_ptr.ptr;

            // 遍历每个chunk
            for index in begin..end {
                unsafe {
                    let p = index * self.head_size;
                    self.compute(ptr1.add(p), ptr2.add(p), output_ptr.add(p));
                }
            }
        }
    }
}
// unsafe impl <T:Float>Send for SiluZipMap<T> {}
// unsafe impl <T:Float>Sync for SiluZipMap<T> {}

impl<T> ZipMapTrait<T> for SiluMulZipMap<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>
        + Sigmoid,
{
    default fn compute(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T) {
        // print!("generic runner\n");
        kernel::scalar::silu::silu_multiply(input_ptr1, input_ptr2, output_ptr, self.head_size);
    }
}
impl ZipMapTrait<f16> for SiluMulZipMap<f16> {
    fn compute(&self, input_ptr1: *const f16, input_ptr2: *const f16, output_ptr: *mut f16) {
        // print!("f16 runner\n");

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::silu::silu_multiply(
                input_ptr1,
                input_ptr2,
                output_ptr,
                self.head_size,
            );
        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::scalar::silu::silu_multiply(input_ptr1, input_ptr2, output_ptr, self.head_size);
    }
}

impl ZipMapTrait<f32> for SiluMulZipMap<f32> {
    fn compute(&self, input_ptr1: *const f32, input_ptr2: *const f32, output_ptr: *mut f32) {
        //print!("f32 runner\n");
        /*#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            SIMD_f32_256_silu_multiply_block(
                input_ptr1 , input_ptr2 , output_ptr , self.length
            );
        };
        // #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]*/
        kernel::scalar::silu::silu_multiply(input_ptr1, input_ptr2, output_ptr, self.head_size);
    }
}
/*
impl ZipMapTrait<f64> for SiluZipMap<f64> {
    fn compute (&self, input_ptr1: *const f64, input_ptr2: *const f64, output_ptr: *mut f64) {
        // print!("f64 runner\n");
        silu_multiply_block(input_ptr1 , input_ptr2 , output_ptr , self.length);
    }
}*/

#[cfg(test)]
mod test {

    use super::*;
    // use super::super::chunk_zipmap::chunk_zipmap;
    use crate::common::tensor_utils::get_strides;
    use approx::assert_ulps_eq;
    // use rand::seq;
    #[test]
    fn test_silu_multiply() {
        let batch_size = 32;
        let head_num = 64;
        let head_size = 128;
        // let hidden_size = 19;
        let shapes = vec![batch_size, head_num, head_size];
        let length = shapes.iter().product(); // 总元素数量
        let input_strides1 = get_strides(&shapes);
        //println!("{:?}", input_strides1);
        let input_strides2 = input_strides1.clone();
        let output_strides = input_strides1.clone();
        // let length: usize = shapes.iter().product();
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
            // -0.5069465041160583,
            // -0.37254711985588074,
            // 0.4420417249202728,
            // -0.9305257201194763,
            // 0.5145581364631653,
            // 0.6260590553283691,
        ]
        .repeat(batch_size * head_num * 4);
        let input_data2 = vec![1.0; length];
        let mut output_data: Vec<f32> = vec![0.0; length];

        // let chunks = chunk_zipmap(shapes, input_data1.as_ptr(), input_strides1, input_data2.as_ptr(), input_strides2, output_data.as_mut_ptr(), output_strides);
        let thread_num: usize = num_cpus::get();
        let mut _operator: SiluMulZipMap<f32> = SiluMulZipMap::new(
            input_data1.as_ptr(),
            input_data2.as_ptr(),
            output_data.as_mut_ptr(),
            head_num,
            head_size,
        );
        // _operator.set_chunk(chunks);
        for i in 0..thread_num {
            _operator.run(batch_size, thread_num, i);
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
        ]
        .repeat(batch_size * head_num * 4);
        //println!("{:?}",result.len());
        assert_ulps_eq!(output_data[..], result, max_ulps = 4);
        // println!("{:?}", output);
    }
}







