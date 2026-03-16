use std::ptr;

use crate::common::send_sync_ptr::MutPtr;
use crate::common::sequence_slice::SequenceSlice;
use crate::operators::assign::assign;

#[derive(Clone)]
pub struct LiftVector<T> {
    ptr: MutPtr<T>,
    length: usize,
}

impl<T> LiftVector<T> {
    pub fn new(ptr: *mut T, length: usize) -> Self {
        Self {
            ptr: MutPtr { ptr },
            length,
        }
    }

    pub fn run(
        &self,
        prefill_size: usize,
        _decode_size: usize,
        round_token_slices: &[SequenceSlice],
        thread_num: usize,
        thread_id: usize,
    ) {
        if prefill_size == 0 {
            return;
        }

        let total_tokens = round_token_slices.len();
        let Some((begin, end)) = assign(total_tokens, thread_num, thread_id) else {
            return;
        };

        unsafe {
            let ptr = self.ptr.ptr;

            for (offset, slice) in round_token_slices[begin..end].iter().enumerate() {
                if slice.length == 0 {
                    continue;
                }

                let source_token_index = slice.token_start_index + slice.length - 1;
                let destination_index = begin + offset;
                let source_ptr = ptr.add(source_token_index * self.length);
                let destination_ptr = ptr.add(destination_index * self.length);

                ptr::copy_nonoverlapping(source_ptr, destination_ptr, self.length);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::common::sequence_slice::SequenceSlice;

    #[test]
    fn test_lift_vector() {
        let length = 4;
        let mut data: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ];

        let round_token_slices = vec![
            SequenceSlice {
                batch_index: 0,
                sequence_index: 0,
                token_start_index: 2,
                length: 1,
            },
            SequenceSlice {
                batch_index: 0,
                sequence_index: 0,
                token_start_index: 2,
                length: 2,
            },
            SequenceSlice {
                batch_index: 0,
                sequence_index: 0,
                token_start_index: 2,
                length: 3,
            },
        ];

        let lift_vector = LiftVector::new(data.as_mut_ptr(), length);
        lift_vector.run(1, 0, &round_token_slices, 2, 0);
        lift_vector.run(1, 0, &round_token_slices, 2, 1);

        assert_eq!(data[0..4], [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(data[4..8], [5.0, 6.0, 7.0, 8.0]);
        assert_eq!(data[8..12], [9.0, 10.0, 11.0, 12.0]);
    }
}
