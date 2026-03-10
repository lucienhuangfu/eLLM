use std::f16;
use std::ptr;

use crate::kernel;
use crate::operators::traits::MapTrait;

// use crate::runtime::inference::state::TaskList;
use crate::common::sequence_slice::SequenceSlice;
use crate::common::send_sync_ptr::{ConstPtr, MutPtr};
use crate::common::num_traits::Sqrt;

// Fuse embedding lookup with RMS normalization
#[derive(Clone)]
pub struct LookupRMSMap<T> {
    sequences_ptr: ConstPtr<usize>,
    word_embedding: ConstPtr<T>,
    output_hidden_ptr: MutPtr<T>,
    output_normal_ptr: MutPtr<T>,
    batch_size: usize,
    hidden_size: usize,
    eps: T,
}

impl<T: Sqrt> LookupRMSMap<T> {
    // Constructor for LookupRMSMap
    pub fn new(
        sequences_ptr: *const usize,
        word_embedding: *const T,
        output_hidden_ptr: *mut T,
        output_normal_ptr: *mut T,
        batch_size: usize,
        hidden_size: usize,
        eps: T,
    ) -> Self {
        Self {
            sequences_ptr: ConstPtr { ptr: sequences_ptr },
            output_hidden_ptr: MutPtr {
                ptr: output_hidden_ptr,
            },
            output_normal_ptr: MutPtr {
                ptr: output_normal_ptr,
            },
            batch_size,
            hidden_size,
            word_embedding: ConstPtr {
                ptr: word_embedding,
            },
            eps,
        }
    }

    // Run the map for a given batch size and thread ID
    pub fn run(
        &self,
        prefill_size: usize,
        decode_size: usize,
        _thread_num: usize,
        _thread_id: usize,
        prefill_list: &[SequenceSlice],
        decode_list: &[SequenceSlice],
    ) {
        if prefill_size > 0 {
            self.run_task_list(prefill_list);
        } else if decode_size > 0 {
            self.run_task_list(decode_list);
        }
    }

    fn run_task_list(&self, slices: &[SequenceSlice]) {
        unsafe {
            let sequences_ptr = self.sequences_ptr.ptr;
            let output_normal_ptr = self.output_normal_ptr.ptr;
            let output_hidden_ptr = self.output_hidden_ptr.ptr;

            for slice in slices {
                let batch_index = slice.batch_index;
                let position_start = slice.sequence_index;
                let token_start = slice.token_start_index;

                for t in 0..slice.length {
                    let position_index = position_start + t;
                    let token_index = token_start + t;
                    let token_id =
                        *sequences_ptr.add((position_index * self.batch_size + batch_index));
                    let embedding_ptr = self.word_embedding.ptr.add(token_id * self.hidden_size);
                    let offset = token_index * self.hidden_size;

                    let hidden_ptr = output_hidden_ptr.add(offset);
                    // Copy embedding to output hidden
                    ptr::copy_nonoverlapping(embedding_ptr, hidden_ptr, self.hidden_size);
                    self.compute(
                        embedding_ptr,
                        output_normal_ptr.add(offset),
                        self.hidden_size,
                    );
                }
            }
        }
    }
}

impl<T: Sqrt> MapTrait<T> for LookupRMSMap<T> {
    default fn compute(&self, input_ptr: *const T, output_ptr: *mut T, length: usize) {
        kernel::scalar::rms_norm::rms_norm(
            input_ptr, output_ptr, length, // self.weight.ptr,
            self.eps,
        );
    }
}

// Specialized implementation of MapTrait for f16
impl MapTrait<f16> for LookupRMSMap<f16> {
    fn compute(&self, input_ptr: *const f16, output_ptr: *mut f16, length: usize) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        kernel::x86_64::f16_512::rms_norm::rms_norm(
            input_ptr, output_ptr, length, // self.weight.ptr,
            self.eps,
        );

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::scalar::rms_norm::rms_norm(input_ptr, output_ptr, length, self.eps);
    }
}

// Specialized implementation of MapTrait for f32
impl MapTrait<f32> for LookupRMSMap<f32> {
    fn compute(&self, input_ptr: *const f32, output_ptr: *mut f32, length: usize) {
        kernel::scalar::rms_norm::rms_norm(
            input_ptr, output_ptr, length, // self.weight.ptr,
            self.eps,
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::common::sequence_slice::SequenceSlice;
    use approx::assert_ulps_eq;

    #[test]
    fn test_lookup_f32() {
        let batch_size = 10; // Each batch processes 10 elements
        let hidden_size = 18;
        let vocab_size = 10;
        let thread_num = 4;

        let shapes = vec![batch_size, hidden_size];
        let length = shapes.iter().product::<usize>(); // Total number of elements

        let eps = 1e-6;
        let tokens_per_thread = (batch_size + thread_num - 1) / thread_num;
        let mut prefill_lists = Vec::with_capacity(thread_num);
        for tid in 0..thread_num {
            let start = tid * tokens_per_thread;
            let end = (start + tokens_per_thread).min(batch_size);
            let mut slices = Vec::with_capacity(end.saturating_sub(start));
            for i in start..end {
                slices.push(SequenceSlice {
                    batch_index: i,
                    sequence_index: 0,
                    token_start_index: i,
                    lift_index: 0,
                    length: 1,
                });
            }
            prefill_lists.push(slices);
        }

        let decode_lists = (0..thread_num).map(|_| Vec::new()).collect::<Vec<_>>();

        let mut sequences = vec![0; (batch_size * batch_size)];
        for i in 0..batch_size {
            sequences[i] = 1;
        }

        let word_embedding: Vec<f32> = (1..=hidden_size)
            .cycle()
            .take((vocab_size * hidden_size))
            .map(|x| x as f32)
            .collect();
        // let weight = vec![1.0f32; hidden_size];
        let mut output_hidden_data: Vec<f32> = vec![0.0; length];
        let mut output_normal_data: Vec<f32> = vec![0.0; length];

        // Initialize LookupRMSMap with these chunks and length
        let o = LookupRMSMap::new(
            sequences.as_ptr(),
            word_embedding.as_ptr(),
            output_hidden_data.as_mut_ptr(),
            output_normal_data.as_mut_ptr(),
            batch_size,
            hidden_size,
            eps,
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

        // Expected hidden output (copied embeddings for sequence index 1)
        let expected_hidden: Vec<f32> = (1..=hidden_size).map(|x| x as f32).collect();

        for i in 0..thread_num {
            o.run(
                batch_size,
                0,
                thread_num,
                i,
                &prefill_lists[i],
                &decode_lists[i],
            );
        }

        // Verify output_normal_data
        assert_ulps_eq!(output_normal_data[18..36], result, max_ulps = 4);

        // Verify output_hidden_data (should contain copied embeddings)
        assert_ulps_eq!(output_hidden_data[18..36], expected_hidden, max_ulps = 1);
    }
}








