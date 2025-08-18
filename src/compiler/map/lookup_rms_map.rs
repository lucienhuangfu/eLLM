use std::f16;

use super::super::super::kernel;
use super::map_trait::MapTrait;
use crate::compiler::assign::assign;
use crate::init::send_sync_ptr::{ConstPtr, MutPtr};
use crate::kernel::generic::sqrt::Sqrt;

// Fuse embedding lookup with RMS normalization
#[derive(Clone)]
pub struct LookupRMSMap<T> {
    chunks: Vec<(ConstPtr<T>, MutPtr<T>)>,
        max_batch_size: usize,
    word_embedding: ConstPtr<T>,
    // sequences 维度是 [sequence, batch]
    sequences: ConstPtr<usize>,
    length: usize,
    weight: ConstPtr<T>,
    eps: T,
    cpu_num: usize,
    hidden_size: usize,

}

impl<T: Sqrt> LookupRMSMap<T> {
    // Constructor for LookupRMSMap
    pub fn new(
        max_batch_size: usize,
        length: usize,
        weight: *const T,
        eps: T,
        word_embedding: *const T,
        sequences: *const usize,
        hidden_size: usize,
        cpu_num: usize,
    ) -> Self {
        Self {
            chunks: vec![],
            length,
            weight: ConstPtr { ptr: weight },
            eps,
            cpu_num,
            word_embedding: ConstPtr {
                ptr: word_embedding,
            },
            sequences: ConstPtr { ptr: sequences },
            hidden_size,
            max_batch_size,
        }
    }

    // Set the chunks for the map
    pub fn set_chunk(&mut self, chunks: Vec<(ConstPtr<T>, MutPtr<T>)>) {
        self.chunks = chunks;
    }

    // Run the map for a given batch size, position interval, and thread ID
    pub fn run(&self, batch_size: usize, position_start: usize, position_interval: usize, thread_id: usize) {
        if let Some((begin, end)) = assign(batch_size * position_interval, self.cpu_num, thread_id) {
            let (mut row_index, mut col_index) = (begin / batch_size, begin % batch_size);

            // Calculate the current pointer for sequences
            let current = self
                .sequences
                .ptr
                .wrapping_add(self.max_batch_size * position_start);
            
            for i in begin..end {
                let index = row_index * self.max_batch_size + col_index;
                
                unsafe {
                    let (_, b) = self.chunks.get_unchecked(index);
                    let p = *current.add(index);
                    let a_ptr = self.word_embedding.ptr.add(p * self.hidden_size);
                    self.compute(a_ptr, b.ptr, self.length);
                }
                if col_index == batch_size {
                    col_index = 0;
                    row_index += 1;
                }
            }
        }

    }
}

impl<T: Sqrt> MapTrait<T> for LookupRMSMap<T> {
    default fn compute(&self, input_ptr: *const T, output_ptr: *mut T, length: usize) {
        kernel::generic::rms_norm::rms_norm(
            input_ptr,
            output_ptr,
            length,
            self.weight.ptr,
            self.eps,
        );
    }
}

// Specialized implementation of MapTrait for f16
impl MapTrait<f16> for LookupRMSMap<f16> {
    fn compute(&self, input_ptr: *const f16, output_ptr: *mut f16, length: usize) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        kernel::x86_64::f16_512::rms_norm::rms_norm(
            input_ptr,
            output_ptr,
            length,
            self.weight.ptr,
            self.eps,
        );

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::generic::rms_norm::rms_norm(
            input_ptr,
            output_ptr,
            length,
            self.weight.ptr,
            self.eps,
        );
    }
}

// Specialized implementation of MapTrait for f32
impl MapTrait<f32> for LookupRMSMap<f32> {
    fn compute(&self, input_ptr: *const f32, output_ptr: *mut f32, length: usize) {
        kernel::generic::rms_norm::rms_norm(
            input_ptr,
            output_ptr,
            length,
            self.weight.ptr,
            self.eps,
        );
    }
}

#[cfg(test)]
mod test {
    use approx::assert_ulps_eq;
    use num_cpus;
    use std::ptr;

    use crate::memory::allocator::allocate_init;
    use super::super::chunk_map::chunk_map;
    use super::*;

    #[test]
    fn test_lookup_f32() {
        let batch_size = 10; // Each batch processes 10 elements
        let hidden_size = 18;
        let vocab_size = 10;
        let cpu_num = num_cpus::get();

        let shapes = vec![batch_size, hidden_size];
        let strides = vec![hidden_size, 1]; // Corresponding strides
        let length = shapes.iter().product(); // Total number of elements
        let sequence_length: usize = 16;
        let position = 8; 
        let eps = 1e-6;

        // Create mock input and output data
        let input_data: Vec<f32> = (1..=hidden_size)
            .cycle()
            .take(length)
            .map(|x| x as f32)
            .collect();
        let sequences: Vec<usize> = vec![1; sequence_length];
        let word_embedding: Vec<f32> = (1..=18)
            .cycle()
            .take(vocab_size * hidden_size)
            .map(|x| x as f32)
            .collect();
        let weight = vec![1.0f32; length];
        let mut output_data: Vec<f32> = vec![0.0; length];

        // Create chunks using chunk_map function
        let chunks = chunk_map(
            shapes,
            strides,
            input_data.as_ptr(),
            output_data.as_mut_ptr(),
        );
        // Initialize LookupRMSMap with these chunks and length
        let mut o = LookupRMSMap::new(
            hidden_size,
            weight.as_ptr(),
            eps,
            cpu_num,
            word_embedding.as_ptr(),
            sequences.as_ptr(),
            hidden_size,
            batch_size,
        );
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
        o.set_chunk(chunks);

        let thread_num: usize = cpu_num;
        for i in 0..thread_num {
            o.run(batch_size, position, i);
        }

        assert_ulps_eq!(output_data[18..36], result, max_ulps = 4);
    }

    #[test]
    fn test_lookup_f16() {
        // let length = 64;
        let batch_size = 64; // Each batch processes 2 elements
        let hidden_size = 128;
        let vocab_size = 512;
        let cpu_num = num_cpus::get();

        let shapes = vec![batch_size, hidden_size];
        let strides = vec![hidden_size, 1]; // Corresponding strides
        let sequence_length: usize = 16;
        let position = 8; // end position
        let eps = 1e-6;

        /*
        // Create mock input and output data
        let input_data: Vec<f16> = (0..sequence_length)
            .into_iter()
            .map(|x| x as f16)
            .collect();
        let sequences: Vec<usize> = vec![1; sequence_length];
        let word_embedding: Vec<f16> = (0..hidden_size*vocab_size)
            .into_iter()
            .map(|x| x as f16)
            .collect();
        let weight = vec![1.0; hidden_size];
        let mut output_data: Vec<f16> = vec![0.0; hidden_size];
         */
        let input_data = allocate_init::<f16>(sequence_length, 0.0);
        for i in 0..sequence_length {
            unsafe {
                ptr::write(input_data.wrapping_add(i), i as f16);
            }
        }
        let sequences = allocate_init::<usize>(sequence_length, 1);
        let word_embedding = allocate_init::<f16>(hidden_size*vocab_size, 0.0);
        for i in 0..hidden_size*vocab_size {
            unsafe {
                ptr::write(word_embedding.wrapping_add(i), i as f16);
            }
        }
        let weight = allocate_init::<f16>(hidden_size, 1.0);
        let output_data = allocate_init::<f16>(hidden_size, 0.0);


        // Create chunks using chunk_map function
        let chunks = chunk_map(
            shapes,
            strides,
            input_data,
            output_data,
        );
        // Initialize LookupRMSMap with these chunks and length
        let mut o = LookupRMSMap::new(
            hidden_size,
            weight,
            eps,
            cpu_num,
            word_embedding,
            sequences,
            hidden_size,
            batch_size,
        );
        let mut expected: Vec<f16> = vec![0.0; hidden_size];

        o.set_chunk(chunks);
        let thread_num: usize = cpu_num;
        for i in 0..thread_num {
            o.run(1, position, i);
        }

        // for j in 0..length {
        //     assert!(f16::abs(output_data[j] - expected[j]) < 1e-6);
        // }
    }
}
