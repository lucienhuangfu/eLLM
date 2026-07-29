use std::ptr;

use crate::operators::assign::assign;
use crate::operators::send_sync_ptr::MutPtr;
use crate::runtime::SequenceSlice;

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
        computing_slices: &[SequenceSlice],
        thread_num: usize,
        thread_id: usize,
    ) {
        let total_tokens = computing_slices.len();
        let Some((begin, end)) = assign(total_tokens, thread_num, thread_id) else {
            return;
        };

        unsafe {
            let ptr = self.ptr.ptr;

            for (offset, slice) in computing_slices
                .iter()
                .skip(begin)
                .take(end - begin)
                .enumerate()
            {
                if !slice.last_token_flag {
                    continue;
                }

                let source_token_index = slice.token_start_index + slice.length - 1;
                let destination_index = slice.lift_index;
                let source_ptr = ptr.add(source_token_index * self.length);
                let destination_ptr = ptr.add(destination_index * self.length);

                ptr::copy(source_ptr, destination_ptr, self.length);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::runtime::SequenceSlice;

    const EMPTY_SLICES: &[SequenceSlice] = &[];

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
                next_sequence_index: 0,
                token_start_index: 2,
                length: 1,
                last_token_flag: true,
                lift_index: 0,
            },
            SequenceSlice {
                batch_index: 0,
                next_sequence_index: 0,
                token_start_index: 2,
                length: 2,
                last_token_flag: true,
                lift_index: 1,
            },
            SequenceSlice {
                batch_index: 0,
                next_sequence_index: 0,
                token_start_index: 2,
                length: 3,
                last_token_flag: true,
                lift_index: 2,
            },
        ];

        let lift_vector = LiftVector::new(data.as_mut_ptr(), length);
        lift_vector.run(&decode_list, 2, 0);
        lift_vector.run(&decode_list, 2, 1);

        assert_eq!(data[0..4], [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(data[4..8], [5.0, 6.0, 7.0, 8.0]);
        assert_eq!(data[8..12], [9.0, 10.0, 11.0, 12.0]);
    }
}
