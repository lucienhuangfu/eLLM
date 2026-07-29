use std::f16;
use std::ptr;

use crate::kernel;
use crate::operators::assign::assign;
use crate::operators::traits::MapTrait;

// use crate::runtime::inference::state::TaskList;
use crate::num_traits::Sqrt;
use crate::operators::send_sync_ptr::{ConstPtr, MutPtr};
use crate::runtime::SequenceSlice;

// Fuse embedding lookup with RMS normalization
#[derive(Clone)]
pub struct LookupRMSMap<T> {
    pub sequences_ptr: ConstPtr<usize>,
    word_embedding: ConstPtr<T>,
    norm_weight: ConstPtr<T>,
    pub output_hidden_ptr: MutPtr<T>,
    pub output_normal_ptr: MutPtr<T>,
    sequence_stride: usize,
    hidden_size: usize,
    eps: T,
}

impl<T: Sqrt> LookupRMSMap<T> {
    // Constructor for LookupRMSMap
    pub fn new(
        sequences_ptr: *const usize,
        word_embedding: *const T,
        norm_weight: *const T,
        output_hidden_ptr: *mut T,
        output_normal_ptr: *mut T,
        sequence_stride: usize,
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
            sequence_stride,
            hidden_size,
            word_embedding: ConstPtr {
                ptr: word_embedding,
            },
            norm_weight: ConstPtr { ptr: norm_weight },
            eps,
        }
    }

    pub fn run(
        &self,
        total_size: usize,
        thread_num: usize,
        thread_id: usize,
        computing_slices: &[SequenceSlice],
    ) {
        let Some((begin, end)) = assign(total_size, thread_num, thread_id) else {
            return;
        };

        let mut lo = 0usize;
        let mut hi = computing_slices.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let s = &computing_slices[mid];
            if s.token_start_index + s.length <= begin {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let start_idx = lo;

        let hidden_size = self.hidden_size;
        let sequence_stride = self.sequence_stride;
        let sequences_ptr = self.sequences_ptr.ptr;
        let embedding_base = self.word_embedding.ptr;
        let output_hidden_ptr = self.output_hidden_ptr.ptr;
        let output_normal_ptr = self.output_normal_ptr.ptr;

        let mut token_cursor = begin;

        for slice in computing_slices.iter().skip(start_idx) {
            if token_cursor >= end {
                break;
            }
            // Gap between slices — no tokens to process.
            if token_cursor < slice.token_start_index {
                break;
            }

            let slice_end = (slice.token_start_index + slice.length).min(end);
            let mut position = slice.next_sequence_index + (token_cursor - slice.token_start_index);

            while token_cursor < slice_end {
                unsafe {
                    let token_id =
                        *sequences_ptr.add(slice.batch_index * sequence_stride + position);
                    let embedding_ptr = embedding_base.add(token_id * hidden_size);
                    let offset = token_cursor * hidden_size;

                    ptr::copy_nonoverlapping(
                        embedding_ptr,
                        output_hidden_ptr.add(offset),
                        hidden_size,
                    );
                    self.compute(embedding_ptr, output_normal_ptr.add(offset), hidden_size);
                }
                token_cursor += 1;
                position += 1;
            }
        }
    }
}

impl<T: Sqrt> MapTrait<T> for LookupRMSMap<T> {
    default fn compute(&self, input_ptr: *const T, output_ptr: *mut T, length: usize) {
        kernel::scalar::rms_norm::rms_norm(
            input_ptr,
            self.norm_weight.ptr,
            output_ptr,
            length,
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
            self.norm_weight.ptr,
            output_ptr,
            length,
            self.eps,
        );

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::scalar::rms_norm::rms_norm(
            input_ptr,
            self.norm_weight.ptr,
            output_ptr,
            length,
            self.eps,
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::runtime::SequenceSlice;
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
        let mut computing_slices = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            computing_slices.push(SequenceSlice {
                batch_index: i,
                next_sequence_index: 0,
                token_start_index: i,
                length: 1,
                last_token_flag: false,
                lift_index: 0,
            });
        }

        let mut sequences = vec![0; batch_size * batch_size];
        for i in 0..batch_size {
            sequences[i] = 1;
        }

        let word_embedding: Vec<f32> = (1..=hidden_size)
            .cycle()
            .take(vocab_size * hidden_size)
            .map(|x| x as f32)
            .collect();
        let weight = vec![1.0f32; hidden_size];
        let mut output_hidden_data: Vec<f32> = vec![0.0; length];
        let mut output_normal_data: Vec<f32> = vec![0.0; length];

        // Initialize LookupRMSMap with these chunks and length
        let o = LookupRMSMap::new(
            sequences.as_ptr(),
            word_embedding.as_ptr(),
            weight.as_ptr(),
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
            o.run(batch_size, thread_num, i, &computing_slices);
        }

        // Verify output_normal_data
        assert_ulps_eq!(output_normal_data[18..36], result, max_ulps = 4);

        // Verify output_hidden_data (should contain copied embeddings)
        assert_ulps_eq!(output_hidden_data[18..36], expected_hidden, max_ulps = 1);
    }

    #[test]
    fn test_lookup_decode_f32_uses_assigned_decode_list() {
        let batch_size = 4;
        let hidden_size = 4;
        let vocab_size = 4;
        let thread_num = 2;
        let eps = 1e-6f32;

        let sequences = vec![0, 1, 2, 3];
        let word_embedding: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let mut output_hidden_data: Vec<f32> = vec![0.0; batch_size * hidden_size];
        let mut output_normal_data: Vec<f32> = vec![0.0; batch_size * hidden_size];
        let weight = vec![1.0f32; hidden_size];

        let decode_list = vec![
            SequenceSlice {
                batch_index: 0,
                next_sequence_index: 0,
                token_start_index: 0,
                length: 1,
                last_token_flag: false,
                lift_index: 0,
            },
            SequenceSlice {
                batch_index: 1,
                next_sequence_index: 0,
                token_start_index: 1,
                length: 1,
                last_token_flag: false,
                lift_index: 0,
            },
            SequenceSlice {
                batch_index: 2,
                next_sequence_index: 0,
                token_start_index: 2,
                length: 1,
                last_token_flag: false,
                lift_index: 0,
            },
            SequenceSlice {
                batch_index: 3,
                next_sequence_index: 0,
                token_start_index: 3,
                length: 1,
                last_token_flag: false,
                lift_index: 0,
            },
        ];

        let operator = LookupRMSMap::new(
            sequences.as_ptr(),
            word_embedding.as_ptr(),
            weight.as_ptr(),
            output_hidden_data.as_mut_ptr(),
            output_normal_data.as_mut_ptr(),
            1,
            hidden_size,
            eps,
        );

        for thread_id in 0..thread_num {
            operator.run(decode_list.len(), thread_num, thread_id, &decode_list);
        }

        let expected_hidden = word_embedding[..vocab_size * hidden_size].to_vec();
        assert_ulps_eq!(
            output_hidden_data.as_slice(),
            expected_hidden.as_slice(),
            max_ulps = 1
        );
        assert!(output_normal_data.iter().all(|value| *value > 0.0));
    }

    #[test]
    fn test_lookup_prefill_reads_row_major_batch_sequence_storage() {
        let batch_size = 2;
        let sequence_stride = 5;
        let hidden_size = 2;
        let eps = 1e-6f32;

        let sequences = vec![
            0, 1, 2, 3, 0, // slot 0
            3, 2, 1, 0, 0, // slot 1
        ];
        let word_embedding: Vec<f32> = vec![
            1.0, 10.0, // token 0
            2.0, 20.0, // token 1
            3.0, 30.0, // token 2
            4.0, 40.0, // token 3
        ];
        let mut output_hidden_data = vec![0.0f32; 4 * hidden_size];
        let mut output_normal_data = vec![0.0f32; 4 * hidden_size];
        let weight = vec![1.0f32; hidden_size];
        let computing_slices = vec![
            SequenceSlice {
                batch_index: 0,
                next_sequence_index: 1,
                token_start_index: 0,
                length: 2,
                last_token_flag: false,
                lift_index: 0,
            },
            SequenceSlice {
                batch_index: 1,
                next_sequence_index: 0,
                token_start_index: 2,
                length: 2,
                last_token_flag: false,
                lift_index: 0,
            },
        ];

        let operator = LookupRMSMap::new(
            sequences.as_ptr(),
            word_embedding.as_ptr(),
            weight.as_ptr(),
            output_hidden_data.as_mut_ptr(),
            output_normal_data.as_mut_ptr(),
            sequence_stride,
            hidden_size,
            eps,
        );

        operator.run(4, 1, 0, &computing_slices);

        let expected_hidden = [2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 3.0, 30.0];
        assert_ulps_eq!(
            output_hidden_data.as_slice(),
            expected_hidden.as_slice(),
            max_ulps = 1
        );
        assert!(output_normal_data.iter().all(|value| *value > 0.0));
    }
}
