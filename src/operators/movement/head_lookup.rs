use std::ptr;

use crate::operators::assign::assign;
use crate::operators::send_sync_ptr::{ConstPtr, MutPtr};

// Standalone embedding lookup: gather the word-embedding row of each decode
// entry's head token into the token buffer. This is the lookup half of
// LookupRMSMap, kept as an independent operator without RMS normalization.
#[derive(Clone)]
pub struct HeadLookup<T> {
    pub head_token_ptr: ConstPtr<usize>,
    word_embedding: ConstPtr<T>,
    pub output_ptr: MutPtr<T>,
    hidden_size: usize,
}

impl<T: Copy> HeadLookup<T> {
    // Constructor for HeadLookup
    pub fn new(
        head_token_ptr: *const usize,
        word_embedding: *const T,
        output_ptr: *mut T,
        hidden_size: usize,
    ) -> Self {
        Self {
            head_token_ptr: ConstPtr {
                ptr: head_token_ptr,
            },
            output_ptr: MutPtr { ptr: output_ptr },
            hidden_size,
            word_embedding: ConstPtr {
                ptr: word_embedding,
            },
        }
    }

    // Tasks are the decode entries: `decode_index` reads `head_token_ptr`
    // at the same index and writes the gathered row to output row
    // `decode_index`.
    pub fn run(&self, decode_size: usize, thread_num: usize, thread_id: usize) {
        let Some((begin, end)) = assign(decode_size, thread_num, thread_id) else {
            return;
        };

        let hidden_size = self.hidden_size;
        let head_token_ptr = self.head_token_ptr.ptr;
        let embedding_base = self.word_embedding.ptr;
        let output_ptr = self.output_ptr.ptr;

        for decode_index in begin..end {
            unsafe {
                let token_id = *head_token_ptr.add(decode_index);
                let embedding_ptr = embedding_base.add(token_id * hidden_size);

                ptr::copy_nonoverlapping(
                    embedding_ptr,
                    output_ptr.add(decode_index * hidden_size),
                    hidden_size,
                );
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn test_head_lookup_f32() {
        let decode_size = 10; // One decode entry per task
        let hidden_size = 18;
        let vocab_size = 10;
        let thread_num = 4;

        let shapes = vec![decode_size, hidden_size];
        let length = shapes.iter().product::<usize>(); // Total number of elements

        let head_tokens = vec![1usize; decode_size];

        let word_embedding: Vec<f32> = (1..=hidden_size)
            .cycle()
            .take(vocab_size * hidden_size)
            .map(|x| x as f32)
            .collect();
        let mut output_data: Vec<f32> = vec![0.0; length];

        // Initialize HeadLookup with these chunks and length
        let o = HeadLookup::new(
            head_tokens.as_ptr(),
            word_embedding.as_ptr(),
            output_data.as_mut_ptr(),
            hidden_size,
        );

        // Expected output (copied embeddings for token index 1)
        let expected: Vec<f32> = (1..=hidden_size).map(|x| x as f32).collect();

        for i in 0..thread_num {
            o.run(decode_size, thread_num, i);
        }

        // Verify output_data (should contain copied embeddings in every row)
        assert_ulps_eq!(output_data[18..36], expected, max_ulps = 1);
    }

    #[test]
    fn test_head_lookup_decode_f32_splits_tasks_across_threads() {
        let decode_size = 4;
        let hidden_size = 4;
        let vocab_size = 4;
        let thread_num = 2;

        let head_tokens = vec![0, 1, 2, 3];
        let word_embedding: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let mut output_data: Vec<f32> = vec![0.0; decode_size * hidden_size];

        let operator = HeadLookup::new(
            head_tokens.as_ptr(),
            word_embedding.as_ptr(),
            output_data.as_mut_ptr(),
            hidden_size,
        );

        for thread_id in 0..thread_num {
            operator.run(decode_size, thread_num, thread_id);
        }

        let expected = word_embedding[..vocab_size * hidden_size].to_vec();
        assert_ulps_eq!(output_data.as_slice(), expected.as_slice(), max_ulps = 1);
    }

    #[test]
    fn test_head_lookup_gathers_by_decode_index() {
        let decode_size = 4;
        let hidden_size = 2;

        let head_tokens = vec![3, 0, 2, 1];
        let word_embedding: Vec<f32> = vec![
            1.0, 10.0, // token 0
            2.0, 20.0, // token 1
            3.0, 30.0, // token 2
            4.0, 40.0, // token 3
        ];
        let mut output_data = vec![0.0f32; decode_size * hidden_size];

        let operator = HeadLookup::new(
            head_tokens.as_ptr(),
            word_embedding.as_ptr(),
            output_data.as_mut_ptr(),
            hidden_size,
        );

        // Two threads, so each task range must stay aligned with its decode
        // index.
        for thread_id in 0..2 {
            operator.run(decode_size, 2, thread_id);
        }

        let expected = [4.0, 40.0, 1.0, 10.0, 3.0, 30.0, 2.0, 20.0];
        assert_ulps_eq!(output_data.as_slice(), expected.as_slice(), max_ulps = 1);
    }
}
