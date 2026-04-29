use std::f16;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::common::num_traits::Sigmoid;
use crate::common::send_sync_ptr::{ConstPtr, MutPtr};
use crate::kernel;
use crate::operators::assign::assign;
use crate::operators::traits::MapTrait;

#[derive(Clone)]
pub struct SigmoidMap<T> {
    ptr1: ConstPtr<T>,
    output_ptr: MutPtr<T>,
    length: usize,
}

impl<T> SigmoidMap<T>
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
    pub fn new(ptr1: *const T, output_ptr: *mut T, length: usize) -> Self {
        Self {
            ptr1: ConstPtr { ptr: ptr1 },
            output_ptr: MutPtr { ptr: output_ptr },
            length,
        }
    }

    pub fn run(
        &self,
        _prefill_size: usize,
        _decode_size: usize,
        thread_num: usize,
        thread_id: usize,
    ) {
        if let Some((begin, end)) = assign(self.length, thread_num, thread_id) {
            let ptr1 = self.ptr1.ptr;
            let output_ptr = self.output_ptr.ptr;
            for index in begin..end {
                unsafe {
                    self.compute(ptr1.add(index), output_ptr.add(index), 1);
                }
            }
        }
    }
}

impl<T> MapTrait<T> for SigmoidMap<T>
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
    default fn compute(&self, input_ptr: *const T, output_ptr: *mut T, length: usize) {
        kernel::scalar::sigmoid::sigmoid(input_ptr, output_ptr, length);
    }
}

impl MapTrait<f16> for SigmoidMap<f16> {
    fn compute(&self, input_ptr: *const f16, output_ptr: *mut f16, length: usize) {
        kernel::scalar::sigmoid::sigmoid(input_ptr, output_ptr, length);
    }
}

impl MapTrait<f32> for SigmoidMap<f32> {
    fn compute(&self, input_ptr: *const f32, output_ptr: *mut f32, length: usize) {
        kernel::scalar::sigmoid::sigmoid(input_ptr, output_ptr, length);
    }
}
