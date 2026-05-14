use std::sync::atomic::AtomicUsize;

use crate::common::send_sync_ptr::MutPtr;

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
    use crate::mem_mgr::allocator::allocate_init;

    let expert_counts = crate::mem_mgr::allocator::allocate::<AtomicUsize>(num_experts);
    let capacity_per_expert = num_tokens;
    let index_tensor = allocate_init(num_experts * capacity_per_expert, 0usize);
    let score_tensor = allocate_init(num_experts * capacity_per_expert, T::default());
    let topk_indices = allocate_init(num_tokens * num_topk, 0usize);

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
    use crate::mem_mgr::allocator::allocate_init;

    let expert_counts = crate::mem_mgr::allocator::allocate::<AtomicUsize>(num_experts);
    for e in 0..num_experts {
        std::ptr::write(expert_counts.add(e), AtomicUsize::new(0));
    }

    let capacity_per_expert = num_tokens;
    ExpertRouting {
        expert_counts: MutPtr { ptr: expert_counts },
        index_tensor: MutPtr {
            ptr: allocate_init(num_experts * capacity_per_expert, 0usize),
        },
        score_tensor: MutPtr {
            ptr: allocate_init(num_experts * capacity_per_expert, T::default()),
        },
        topk_indices: MutPtr {
            ptr: allocate_init(num_tokens * num_topk, 0usize),
        },
        num_experts,
        num_tokens,
        num_topk,
        capacity_per_expert,
    }
}
