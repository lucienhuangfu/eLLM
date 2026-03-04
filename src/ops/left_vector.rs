use std::ptr;

use crate::common::sequence_slice::SequenceSlice;
use crate::common::send_sync_ptr::MutPtr;

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
		decode_tokens: &[SequenceSlice],
		_thread_num: usize,
		_thread_id: usize,
	) {
		if prefill_size > 0 {
			unsafe {
				let ptr = self.ptr.ptr;

				for slice in decode_tokens {
					for offset in 0..slice.length {
						let source_index = slice.token_start_index + offset;
						let destination_index = slice.lift_index + offset;

						let source_ptr = ptr.add(source_index * self.length);
						let destination_ptr = ptr.add(destination_index * self.length);

						ptr::copy_nonoverlapping(source_ptr, destination_ptr, self.length);
					}
				}
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
			0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0,
			1.0, 2.0, 3.0, 4.0,
			5.0, 6.0, 7.0, 8.0,
		];

		let decode_tokens = vec![
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
		];

		let lift_vector = LiftVector::new(data.as_mut_ptr(), length);
		lift_vector.run(1, 0, &decode_tokens, 1, 0);

		assert_eq!(data[0..4], [1.0, 2.0, 3.0, 4.0]);
		assert_eq!(data[4..8], [5.0, 6.0, 7.0, 8.0]);
	}
}
