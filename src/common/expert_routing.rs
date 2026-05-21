use std::sync::atomic::AtomicUsize;

use crate::common::send_sync_ptr::MutPtr;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ExpertTaskMeta {
    /// Expert id for this contiguous task range.
    /// 这个连续任务区间对应的 expert 编号。
    pub expert_id: usize,
    /// First routed-token offset in the compact per-thread routed token buffer.
    /// 当前 expert 在每线程紧凑 token buffer 中的起始位置。
    pub token_begin: usize,
    /// Number of tokens routed to this expert.
    /// 路由到当前 expert 的 token 数量。
    pub token_count: usize,
    /// First global task id owned by this expert.
    /// 当前 expert 覆盖的第一个全局 task id。
    pub task_begin: usize,
    /// One-past-last global task id owned by this expert.
    /// 当前 expert 覆盖的全局 task id 结束位置，左闭右开。
    pub task_end: usize,
}

/// Map one global task id to its expert and local matrix tile.
/// 将一个全局 task id 分配到对应 expert 以及该 expert 内部的局部矩阵 tile。
#[inline(always)]
pub(crate) fn task_assign(
    expert_tasks: &[ExpertTaskMeta],
    output_column_tile_count: usize,
    task_id: usize,
) -> Option<(ExpertTaskMeta, usize, usize)> {
    let task_meta_index = expert_tasks.partition_point(|meta| meta.task_end <= task_id);
    let task_meta = *expert_tasks.get(task_meta_index)?;
    debug_assert!(task_id >= task_meta.task_begin && task_id < task_meta.task_end);
    let local_task_id = task_id - task_meta.task_begin;
    Some((
        task_meta,
        local_task_id / output_column_tile_count,
        local_task_id % output_column_tile_count,
    ))
}

#[derive(Clone, Copy)]
pub struct ExpertRouting<T> {
    pub expert_counts: MutPtr<AtomicUsize>,
    pub index_tensor: MutPtr<usize>,
    pub score_tensor: MutPtr<T>,
    pub topk_indices: MutPtr<usize>,
    pub num_experts: usize,
    pub num_tokens: usize,
    pub num_topk: usize,
    pub capacity_per_expert: usize,
}

impl<T> ExpertRouting<T> {
    #[inline(always)]
    pub fn expert_offset(&self, expert_id: usize, pos: usize) -> usize {
        expert_id * self.capacity_per_expert + pos
    }

    #[inline(always)]
    pub fn topk_offset(&self, token_id: usize, slot: usize) -> usize {
        token_id * self.num_topk + slot
    }
}

#[cfg(test)]
pub unsafe fn routing_from_dense<T: Copy + Default>(
    num_experts: usize,
    num_tokens: usize,
    num_topk: usize,
    indice_ptr: *const bool,
    score_ptr: *const T,
    topk_indices_ptr: *const usize,
) -> ExpertRouting<T> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::common::send_sync_ptr::MutPtr;
    use crate::mem_mgr::allocator::AlignedBox;

    let expert_counts_box = AlignedBox::<AtomicUsize>::allocate(num_experts);
    let expert_counts = expert_counts_box.as_mut_ptr();
    std::mem::forget(expert_counts_box);
    let capacity_per_expert = num_tokens;
    let index_tensor_box = AlignedBox::allocate_init(num_experts * capacity_per_expert, 0usize);
    let index_tensor = index_tensor_box.as_mut_ptr();
    std::mem::forget(index_tensor_box);
    let score_tensor_box =
        AlignedBox::allocate_init(num_experts * capacity_per_expert, T::default());
    let score_tensor = score_tensor_box.as_mut_ptr();
    std::mem::forget(score_tensor_box);
    let topk_indices_box = AlignedBox::allocate_init(num_tokens * num_topk, 0usize);
    let topk_indices = topk_indices_box.as_mut_ptr();
    std::mem::forget(topk_indices_box);

    for t in 0..(num_tokens * num_topk) {
        *topk_indices.add(t) = *topk_indices_ptr.add(t);
    }

    for e in 0..num_experts {
        std::ptr::write(expert_counts.add(e), AtomicUsize::new(0));
        let mut pos = 0usize;
        for token in 0..num_tokens {
            let dense_offset = e * num_tokens + token;
            if *indice_ptr.add(dense_offset) {
                let compact_offset = e * capacity_per_expert + pos;
                *index_tensor.add(compact_offset) = token;
                *score_tensor.add(compact_offset) = *score_ptr.add(dense_offset);
                pos += 1;
            }
        }
        (&*expert_counts.add(e)).store(pos, Ordering::Release);
    }

    ExpertRouting {
        expert_counts: MutPtr { ptr: expert_counts },
        index_tensor: MutPtr { ptr: index_tensor },
        score_tensor: MutPtr { ptr: score_tensor },
        topk_indices: MutPtr { ptr: topk_indices },
        num_experts,
        num_tokens,
        num_topk,
        capacity_per_expert,
    }
}

#[cfg(test)]
pub unsafe fn empty_routing<T: Copy + Default>(
    num_experts: usize,
    num_tokens: usize,
    num_topk: usize,
) -> ExpertRouting<T> {
    use std::sync::atomic::AtomicUsize;

    use crate::common::send_sync_ptr::MutPtr;
    use crate::mem_mgr::allocator::AlignedBox;

    let expert_counts_box = AlignedBox::<AtomicUsize>::allocate(num_experts);
    let expert_counts = expert_counts_box.as_mut_ptr();
    std::mem::forget(expert_counts_box);
    for e in 0..num_experts {
        std::ptr::write(expert_counts.add(e), AtomicUsize::new(0));
    }

    let capacity_per_expert = num_tokens;
    let index_tensor_box = AlignedBox::allocate_init(num_experts * capacity_per_expert, 0usize);
    let index_tensor = index_tensor_box.as_mut_ptr();
    std::mem::forget(index_tensor_box);
    let score_tensor_box =
        AlignedBox::allocate_init(num_experts * capacity_per_expert, T::default());
    let score_tensor = score_tensor_box.as_mut_ptr();
    std::mem::forget(score_tensor_box);
    let topk_indices_box = AlignedBox::allocate_init(num_tokens * num_topk, 0usize);
    let topk_indices = topk_indices_box.as_mut_ptr();
    std::mem::forget(topk_indices_box);

    ExpertRouting {
        expert_counts: MutPtr { ptr: expert_counts },
        index_tensor: MutPtr { ptr: index_tensor },
        score_tensor: MutPtr { ptr: score_tensor },
        topk_indices: MutPtr { ptr: topk_indices },
        num_experts,
        num_tokens,
        num_topk,
        capacity_per_expert,
    }
}
