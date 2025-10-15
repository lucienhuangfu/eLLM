use std::f16;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

use super::super::super::init::{
    matmul_params::MatmulParams,
    send_sync_ptr::{ConstPtr, MutPtr},
};
use super::super::super::kernel;
use super::super::assign::assign;
use super::mul_trait::Matmul2Trait;
use crate::memory::allocator::allocate_init;
use crate::memory::cache::Cache;

// [(experts_id, [(token_id, weight)])]
// sorted_ids: Vec<(usize, Vec<(usize, T)>)>,
// 专家路由信息的内存管理结构
#[derive(Clone)]
pub struct ExpertsRouting<T> {
    // 专家数量
    pub num_experts: usize,
    // 每个专家的token信息起始位置
    pub experts_offsets_ptr: ConstPtr<usize>,
    // 每个专家的token数量
    pub experts_token_counts_ptr: ConstPtr<usize>,
    // 所有token_id的连续存储
    pub token_ids_ptr: ConstPtr<usize>,
    // 所有weight的连续存储
    pub weights_ptr: ConstPtr<T>,
    _marker: PhantomData<T>,
}

impl<T> ExpertsRouting<T>
where
    T: Copy + Default,
{
    pub fn new(
        // 最大token_num = sequence_chunk_size * batch_size
        sequence_chunk_size: usize,
        batch_size: usize,
        num_experts: usize,
        num_experts_per_tok: usize,
        cache: &mut Cache<T>,
    ) -> Self {
        // 使用系统分配器分配专家偏移量和token数量的内存
        let experts_offsets_ptr = unsafe { allocate_init(num_experts, 0usize) };
        let experts_token_counts_ptr = unsafe { allocate_init(num_experts, 0usize) };

        // 计算最大可能的token数量
        let max_tokens = sequence_chunk_size * batch_size * num_experts_per_tok;

        // 使用系统分配器分配token_ids，使用cache分配weights
        let token_ids_ptr = unsafe { allocate_init(max_tokens, 0usize) };
        let weights_ptr = cache.get(
            &format!(
                "experts_routing_weights_{}_{}",
                sequence_chunk_size, batch_size
            ),
            max_tokens,
        );

        Self {
            num_experts,
            experts_offsets_ptr: ConstPtr {
                ptr: experts_offsets_ptr,
            },
            experts_token_counts_ptr: ConstPtr {
                ptr: experts_token_counts_ptr,
            },
            token_ids_ptr: ConstPtr { ptr: token_ids_ptr },
            weights_ptr: ConstPtr { ptr: weights_ptr },
            _marker: PhantomData,
        }
    }

    // 获取指定专家的token信息
    pub fn get_experts_tokens(&self, experts_idx: usize) -> (*const usize, *const T, usize) {
        unsafe {
            let offset = *self.experts_offsets_ptr.ptr.add(experts_idx);
            let count = *self.experts_token_counts_ptr.ptr.add(experts_idx);
            let token_ids = self.token_ids_ptr.ptr.add(offset);
            let weights = self.weights_ptr.ptr.add(offset);
            (token_ids, weights, count)
        }
    }
}
