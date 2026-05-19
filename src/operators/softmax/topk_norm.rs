use std::f16;
use std::ops::{AddAssign, Div};
use std::sync::atomic::Ordering;

use crate::common::expert_routing::ExpertRouting;
use crate::common::send_sync_ptr::{ConstPtr, MutPtr};
use crate::kernel;
use crate::mem_mgr::allocator::AlignedBox;
use crate::operators::assign::assign;
use crate::operators::traits::ExpertsTopkNormTrait;

#[derive(Clone)]
pub struct ExpertsTopkNorm<T> {
    ptr1: ConstPtr<T>,
    topk_values_ptr: MutPtr<T>,
    routing: ExpertRouting<T>,
    num_experts: usize,
    num_topk: usize,
    decode_only_flag: bool,
}

impl<T: Copy + Default> ExpertsTopkNorm<T> {
    pub fn new(
        ptr1: *const T,
        routing: ExpertRouting<T>,
        batch_size: usize,
        num_experts: usize,
        num_topk: usize,
        decode_only_flag: bool,
    ) -> Self {
        Self {
            ptr1: ConstPtr { ptr: ptr1 },
            topk_values_ptr: MutPtr {
                ptr: {
                    let mut boxed = AlignedBox::allocate_init(batch_size * num_topk, T::default());
                    let ptr = boxed.as_mut_ptr();
                    std::mem::forget(boxed);
                    ptr
                },
            },
            routing,
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
            let assigned_tokens = end - begin;
            let mini_capacity = (assigned_tokens * self.num_topk).max(1);
            let mut mini_counts = vec![0usize; self.num_experts];
            let mut mini_indices = vec![0usize; self.num_experts * mini_capacity];
            let mut mini_scores = vec![T::default(); self.num_experts * mini_capacity];

            for token_index in begin..end {
                unsafe {
                    let input_offset = token_index * self.num_experts;
                    let topk_offset = token_index * self.num_topk;
                    self.compute(
                        self.ptr1.ptr.add(input_offset),
                        self.topk_values_ptr.ptr.add(topk_offset),
                        self.routing.topk_indices.ptr.add(topk_offset),
                        self.num_experts,
                        self.num_topk,
                    );

                    for slot in 0..self.num_topk {
                        let expert_idx = *self.routing.topk_indices.ptr.add(topk_offset + slot);
                        let local_pos = mini_counts[expert_idx];
                        let mini_offset = expert_idx * mini_capacity + local_pos;
                        mini_indices[mini_offset] = token_index;
                        mini_scores[mini_offset] =
                            *self.topk_values_ptr.ptr.add(topk_offset + slot);
                        mini_counts[expert_idx] += 1;
                    }
                }
            }

            unsafe {
                for e in 0..self.num_experts {
                    let count = mini_counts[e];
                    if count == 0 {
                        continue;
                    }

                    let base = (&*self.routing.expert_counts.ptr.add(e))
                        .fetch_add(count, Ordering::AcqRel);
                    debug_assert!(base + count <= self.routing.capacity_per_expert);

                    for i in 0..count {
                        let src = e * mini_capacity + i;
                        let dst = self.routing.expert_offset(e, base + i);
                        *self.routing.index_tensor.ptr.add(dst) = mini_indices[src];
                        *self.routing.score_tensor.ptr.add(dst) = mini_scores[src];
                    }
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
        topk_indices_ptr: *mut usize,
        input_length: usize,
        output_length: usize,
    ) {
        kernel::scalar::experts_topk_norm::experts_topk_norm(
            ptr1,
            topk_values_ptr,
            topk_indices_ptr,
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
        topk_indices_ptr: *mut usize,
        input_length: usize,
        output_length: usize,
    ) {
        kernel::scalar::experts_topk_norm::experts_topk_norm(
            ptr1,
            topk_values_ptr,
            topk_indices_ptr,
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
        topk_indices_ptr: *mut usize,
        input_length: usize,
        output_length: usize,
    ) {
        kernel::scalar::experts_topk_norm::experts_topk_norm(
            ptr1,
            topk_values_ptr,
            topk_indices_ptr,
            input_length,
            output_length,
        );
    }
}
