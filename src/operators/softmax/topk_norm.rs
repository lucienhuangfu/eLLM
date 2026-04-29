use std::f16;
use std::ops::{AddAssign, Div};

use crate::common::send_sync_ptr::{ConstPtr, MutPtr};
use crate::kernel;
use crate::mem_mgr::allocator::allocate_init;
use crate::operators::assign::assign;
use crate::operators::traits::ExpertsTopkNormTrait;

#[derive(Clone)]
pub struct ExpertsTopkNorm<T> {
    ptr1: ConstPtr<T>,
    topk_values_ptr: MutPtr<T>,
    topk_indices_ptr: MutPtr<usize>,
    experts_indicator: MutPtr<bool>,
    indice_ptr: MutPtr<bool>,
    value_ptr: MutPtr<T>,
    batch_size: usize,
    num_experts: usize,
    num_topk: usize,
    decode_only_flag: bool,
}

impl<T: Copy + Default> ExpertsTopkNorm<T> {
    pub fn new(
        ptr1: *const T,
        experts_indicator: *mut bool,
        indice_ptr: *mut bool,
        value_ptr: *mut T,
        topk_indices_ptr: *mut usize,
        batch_size: usize,
        num_experts: usize,
        num_topk: usize,
        decode_only_flag: bool,
    ) -> Self {
        Self {
            ptr1: ConstPtr { ptr: ptr1 },
            topk_values_ptr: MutPtr {
                ptr: allocate_init::<T>(batch_size * num_topk, T::default()),
            },
            topk_indices_ptr: MutPtr {
                ptr: topk_indices_ptr,
            },
            experts_indicator: MutPtr {
                ptr: experts_indicator,
            },
            indice_ptr: MutPtr { ptr: indice_ptr },
            value_ptr: MutPtr { ptr: value_ptr },
            batch_size,
            num_experts,
            num_topk,
            decode_only_flag,
        }
    }
}

impl<T> ExpertsTopkNorm<T>
where
    T: Copy + PartialOrd + PartialEq + Default + AddAssign + Div<Output = T>,
{
    pub fn run(
        &self,
        prefill_size: usize,
        decode_size: usize,
        thread_num: usize,
        thread_id: usize,
    ) {
        let task_size = if self.decode_only_flag {
            decode_size
        } else {
            prefill_size
        };

        if let Some((begin, end)) = assign(task_size, thread_num, thread_id) {
            for token_index in begin..end {
                unsafe {
                    let input_offset = token_index * self.num_experts;
                    self.compute(
                        self.ptr1.ptr.add(input_offset),
                        self.topk_values_ptr.ptr.add(token_index * self.num_topk),
                        self.experts_indicator.ptr,
                        self.indice_ptr.ptr,
                        self.value_ptr.ptr,
                        self.topk_indices_ptr.ptr.add(token_index * self.num_topk),
                        token_index,
                        self.batch_size,
                        self.num_experts,
                        self.num_topk,
                    );
                }
            }
        }
    }
}

impl<T> ExpertsTopkNormTrait<T> for ExpertsTopkNorm<T>
where
    T: Copy + PartialOrd + PartialEq + Default + AddAssign + Div<Output = T>,
{
    default fn compute(
        &self,
        ptr1: *const T,
        topk_values_ptr: *mut T,
        experts_indicator: *mut bool,
        indice_ptr: *mut bool,
        value_ptr: *mut T,
        topk_indices_ptr: *mut usize,
        token_index: usize,
        batch_size: usize,
        input_length: usize,
        output_length: usize,
    ) {
        kernel::scalar::experts_topk_norm::experts_topk_norm(
            ptr1,
            topk_values_ptr,
            experts_indicator,
            indice_ptr,
            value_ptr,
            topk_indices_ptr,
            token_index,
            batch_size,
            input_length,
            output_length,
        );
    }
}

impl ExpertsTopkNormTrait<f16> for ExpertsTopkNorm<f16> {
    fn compute(
        &self,
        ptr1: *const f16,
        topk_values_ptr: *mut f16,
        experts_indicator: *mut bool,
        indice_ptr: *mut bool,
        value_ptr: *mut f16,
        topk_indices_ptr: *mut usize,
        token_index: usize,
        batch_size: usize,
        input_length: usize,
        output_length: usize,
    ) {
        kernel::scalar::experts_topk_norm::experts_topk_norm(
            ptr1,
            topk_values_ptr,
            experts_indicator,
            indice_ptr,
            value_ptr,
            topk_indices_ptr,
            token_index,
            batch_size,
            input_length,
            output_length,
        );
    }
}

impl ExpertsTopkNormTrait<f32> for ExpertsTopkNorm<f32> {
    fn compute(
        &self,
        ptr1: *const f32,
        topk_values_ptr: *mut f32,
        experts_indicator: *mut bool,
        indice_ptr: *mut bool,
        value_ptr: *mut f32,
        topk_indices_ptr: *mut usize,
        token_index: usize,
        batch_size: usize,
        input_length: usize,
        output_length: usize,
    ) {
        kernel::scalar::experts_topk_norm::experts_topk_norm(
            ptr1,
            topk_values_ptr,
            experts_indicator,
            indice_ptr,
            value_ptr,
            topk_indices_ptr,
            token_index,
            batch_size,
            input_length,
            output_length,
        );
    }
}
