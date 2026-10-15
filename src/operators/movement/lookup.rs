use std::ptr;

use crate::operators::assign::assign;
use crate::operators::send_sync_ptr::{ConstPtr, MutPtr};
use crate::runtime::SequenceSlice;

// Standalone embedding lookup: gather the word-embedding row of every
// scheduled token into the token buffer. This is the lookup half of
// LookupRMSMap, kept as an independent operator without RMS normalization.
#[derive(Clone)]
pub struct Lookup<T> {
    pub sequences_ptr: ConstPtr<usize>,
    word_embedding: ConstPtr<T>,
    pub output_ptr: MutPtr<T>,
    sequence_stride: usize,
    hidden_size: usize,
}

impl<T: Copy> Lookup<T> {
    // Constructor for Lookup
    pub fn new(
        sequences_ptr: *const usize,
        word_embedding: *const T,
        output_ptr: *mut T,
        sequence_stride: usize,
        hidden_size: usize,
    ) -> Self {
        Self {
            sequences_ptr: ConstPtr { ptr: sequences_ptr },
            output_ptr: MutPtr { ptr: output_ptr },
            sequence_stride,
            hidden_size,
            word_embedding: ConstPtr {
                ptr: word_embedding,
            },
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
            if s.global_start_index + s.local_length <= begin {
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
        let output_ptr = self.output_ptr.ptr;

        let mut token_cursor = begin;

        for slice in computing_slices.iter().skip(start_idx) {
            if token_cursor >= end {
                break;
            }
            // Gap between slices — no tokens to process.
            if token_cursor < slice.global_start_index {
                break;
            }

            let slice_end = (slice.global_start_index + slice.local_length).min(end);
            let mut position = slice.local_next_index + (token_cursor - slice.global_start_index);

            while token_cursor < slice_end {
                unsafe {
                    let token_id =
                        *sequences_ptr.add(slice.batch_index * sequence_stride + position);
                    let embedding_ptr = embedding_base.add(token_id * hidden_size);

                    ptr::copy_nonoverlapping(
                        embedding_ptr,
                        output_ptr.add(token_cursor * hidden_size),
                        hidden_size,
                    );
                }
                token_cursor += 1;
                position += 1;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn test_lookup_f32() {
        let batch_size = 10; // Each batch processes 10 elements
        let hidden_size = 18;
        let vocab_size = 10;
        let thread_num = 4;

        let shapes = vec![batch_size, hidden_size];
        let length = shapes.iter().product::<usize>(); // Total number of elements

        let mut computing_slices = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            computing_slices.push(SequenceSlice {
                batch_index: i,
                local_next_index: 0,
                global_start_index: i,
                local_length: 1,
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
        let mut output_data: Vec<f32> = vec![0.0; length];

        // Initialize Lookup with these chunks and length
        let o = Lookup::new(
            sequences.as_ptr(),
            word_embedding.as_ptr(),
            output_data.as_mut_ptr(),
            batch_size,
            hidden_size,
        );

        // Expected output (copied embeddings for sequence index 1)
        let expected: Vec<f32> = (1..=hidden_size).map(|x| x as f32).collect();

        for i in 0..thread_num {
            o.run(batch_size, thread_num, i, &computing_slices);
        }

        // Verify output_data (should contain copied embeddings)
        assert_ulps_eq!(output_data[18..36], expected, max_ulps = 1);
    }

    #[test]
    fn test_lookup_decode_f32_uses_assigned_decode_list() {
        let batch_size = 4;
        let hidden_size = 4;
        let vocab_size = 4;
        let thread_num = 2;

        let sequences = vec![0, 1, 2, 3];
        let word_embedding: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let mut output_data: Vec<f32> = vec![0.0; batch_size * hidden_size];

        let decode_list = vec![
            SequenceSlice {
                batch_index: 0,
                local_next_index: 0,
                global_start_index: 0,
                local_length: 1,
                last_token_flag: false,
                lift_index: 0,
            },
            SequenceSlice {
                batch_index: 1,
                local_next_index: 0,
                global_start_index: 1,
                local_length: 1,
                last_token_flag: false,
                lift_index: 0,
            },
            SequenceSlice {
                batch_index: 2,
                local_next_index: 0,
                global_start_index: 2,
                local_length: 1,
                last_token_flag: false,
                lift_index: 0,
            },
            SequenceSlice {
                batch_index: 3,
                local_next_index: 0,
                global_start_index: 3,
                local_length: 1,
                last_token_flag: false,
                lift_index: 0,
            },
        ];

        let operator = Lookup::new(
            sequences.as_ptr(),
            word_embedding.as_ptr(),
            output_data.as_mut_ptr(),
            1,
            hidden_size,
        );

        for thread_id in 0..thread_num {
            operator.run(decode_list.len(), thread_num, thread_id, &decode_list);
        }

        let expected = word_embedding[..vocab_size * hidden_size].to_vec();
        assert_ulps_eq!(output_data.as_slice(), expected.as_slice(), max_ulps = 1);
    }

    #[test]
    fn test_lookup_prefill_reads_row_major_slot_sequence_storage() {
        let sequence_stride = 5;
        let hidden_size = 2;

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
        let mut output_data = vec![0.0f32; 4 * hidden_size];
        let computing_slices = vec![
            SequenceSlice {
                batch_index: 0,
                local_next_index: 1,
                global_start_index: 0,
                local_length: 2,
                last_token_flag: false,
                lift_index: 0,
            },
            SequenceSlice {
                batch_index: 1,
                local_next_index: 0,
                global_start_index: 2,
                local_length: 2,
                last_token_flag: false,
                lift_index: 0,
            },
        ];

        let operator = Lookup::new(
            sequences.as_ptr(),
            word_embedding.as_ptr(),
            output_data.as_mut_ptr(),
            sequence_stride,
            hidden_size,
        );

        operator.run(4, 1, 0, &computing_slices);

        let expected = [2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 3.0, 30.0];
        assert_ulps_eq!(output_data.as_slice(), expected.as_slice(), max_ulps = 1);
    }
}
