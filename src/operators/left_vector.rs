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
        decode_list: &[SequenceSlice],
        thread_num: usize,
        thread_id: usize,
    ) {
        if prefill_size == 0 {
            return;
        }

        let total_tokens = decode_list.len();
        let Some((begin, end)) = assign(total_tokens, thread_num, thread_id) else {
            return;
        };

        unsafe {
            let ptr = self.ptr.ptr;

            for slice in &decode_list[begin..end] {
                debug_assert_eq!(slice.length, 1);

                let source_ptr = ptr.add(slice.token_start_index * self.length);
                let destination_ptr = ptr.add(slice.lift_index * self.length);

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

        let decode_list = vec![
            SequenceSlice {
                batch_index: 0,
                sequence_index: 0,
                token_start_index: 2,
                lift_index: 0,
                length: 1,
            },
            SequenceSlice {
                batch_index: 0,
                sequence_index: 0,
                token_start_index: 3,
                lift_index: 1,
                length: 1,
            },
            SequenceSlice {
                batch_index: 0,
                sequence_index: 0,
                token_start_index: 4,
                lift_index: 2,
                length: 1,
            },
        ];

        let lift_vector = LiftVector::new(data.as_mut_ptr(), length);
        lift_vector.run(1, 0, &decode_list, 2, 0);
        lift_vector.run(1, 0, &decode_list, 2, 1);

        assert_eq!(data[0..4], [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(data[4..8], [5.0, 6.0, 7.0, 8.0]);
        assert_eq!(data[8..12], [9.0, 10.0, 11.0, 12.0]);
    }
}
