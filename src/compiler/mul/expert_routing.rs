use std::f16;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

use super::super::super::init::{
    send_sync_ptr::{ConstPtr, MutPtr},
    matmul_params::MatmulParams,
};
use super::super::super::kernel;
use super::super::assign::assign;
use super::mul_trait::Matmul2Trait;
use crate::memory::cache::Cache;

// [(expert_id, [(token_id, weight)])]
// sorted_ids: Vec<(usize, Vec<(usize, T)>)>,
// 专家路由信息的内存管理结构
#[derive(Clone)]
pub struct ExpertRouting<T> {
    // 专家数量
    pub num_experts: usize,
    // 每个专家的token信息起始位置
    pub expert_offsets_ptr: ConstPtr<usize>,
    // 每个专家的token数量
    pub expert_token_counts_ptr: ConstPtr<usize>,
    // 所有token_id的连续存储
    pub token_ids_ptr: ConstPtr<usize>,
    // 所有weight的连续存储
    pub weights_ptr: ConstPtr<T>,
    _marker: PhantomData<T>,
}

impl<T> ExpertRouting<T>
where
    T: Copy + Default,
{
    pub fn new(
        sorted_ids: Vec<(usize, Vec<(usize, T)>)>,
        cache: &mut Cache<T>,
        cache_usize: &mut Cache<usize>,
    ) -> Self {
        let num_experts = sorted_ids.len();
        
        // 分配专家偏移量和token数量的内存
        let expert_offsets_ptr = cache_usize.get(
            &format!("expert_routing_offsets_{}", num_experts),
            num_experts,
        );
        let expert_token_counts_ptr = cache_usize.get(
            &format!("expert_routing_counts_{}", num_experts),
            num_experts,
        );
        
        // 计算总的token数量
        let total_tokens: usize = sorted_ids.iter().map(|(_, tokens)| tokens.len()).sum();
        
        // 分配token_ids和weights的连续内存
        let token_ids_ptr = cache_usize.get(
            &format!("expert_routing_token_ids_{}", total_tokens),
            total_tokens,
        );
        let weights_ptr = cache.get(
            &format!("expert_routing_weights_{}", total_tokens),
            total_tokens,
        );
        
        unsafe {
            let mut current_offset = 0;
            
            // 填充数据
            for (expert_idx, (expert_id, tokens)) in sorted_ids.iter().enumerate() {
                // 设置专家的偏移量和token数量
                *expert_offsets_ptr.add(expert_idx) = current_offset;
                *expert_token_counts_ptr.add(expert_idx) = tokens.len();
                
                // 复制token_ids和weights
                for (token_idx, (token_id, weight)) in tokens.iter().enumerate() {
                    *token_ids_ptr.add(current_offset + token_idx) = *token_id;
                    *weights_ptr.add(current_offset + token_idx) = *weight;
                }
                
                current_offset += tokens.len();
            }
        }
        
        Self {
            num_experts,
            expert_offsets_ptr: ConstPtr { ptr: expert_offsets_ptr },
            expert_token_counts_ptr: ConstPtr { ptr: expert_token_counts_ptr },
            token_ids_ptr: ConstPtr { ptr: token_ids_ptr },
            weights_ptr: ConstPtr { ptr: weights_ptr },
            _marker: PhantomData,
        }
    }
    
    // 获取指定专家的token信息
    pub fn get_expert_tokens(&self, expert_idx: usize) -> (*const usize, *const T, usize) {
        unsafe {
            let offset = *self.expert_offsets_ptr.ptr.add(expert_idx);
            let count = *self.expert_token_counts_ptr.ptr.add(expert_idx);
            let token_ids = self.token_ids_ptr.ptr.add(offset);
            let weights = self.weights_ptr.ptr.add(offset);
            (token_ids, weights, count)
        }
    }
}