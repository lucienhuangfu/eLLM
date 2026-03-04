use std::cell::RefCell;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::rc::Rc;

use crate::common::num_traits::Sigmoid;
use crate::common::num_traits::Sqrt;
use crate::common::num_traits::{exp::Exp, neg_infinity::NegInfinity};

use super::super::runtime::tensor::{Tensor, TensorCtx};

use super::mlp::MLP;
use super::sparse_moe_block::SparseMoeBlock;

#[derive(Clone)]
pub enum MoeLayer<T>
where
    T: Copy + PartialOrd,
{
    MLP(MLP<T>),
    SparseMoe(SparseMoeBlock<T>),
}

impl<T> MoeLayer<T>
where
    T: Copy
        + PartialOrd
        + Default
        + Sub<Output = T>
        + Neg<Output = T>
        + Exp
        + NegInfinity
        + Sigmoid
        + Sqrt
        + AddAssign,
{
    pub fn new_mlp(
        hidden_size: usize,
        intermediate_size: usize,
        parent_scope_name: &str,
        ctx: Rc<TensorCtx<T>>,
    ) -> Self {
        MoeLayer::MLP(MLP::new(
            hidden_size,
            intermediate_size,
            parent_scope_name,
            ctx,
        ))
    }

    pub fn new_sparse_moe(
        // sequence_chunk_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
        num_experts: usize,
        num_topk: usize,
        norm_topk_prob: bool,
        parent_scope_name: &str,
        ctx: Rc<TensorCtx<T>>,
    ) -> Self {
        MoeLayer::SparseMoe(SparseMoeBlock::new(
            hidden_size,
            intermediate_size,
            num_experts,
            num_topk,
            norm_topk_prob,
            parent_scope_name,
            ctx,
        ))
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<T>,
        residual: &Tensor<T>,
        tensor_name: String,
    ) -> Tensor<T> {
        match self {
            MoeLayer::MLP(mlp) => mlp.forward(hidden_states, residual, tensor_name),
            MoeLayer::SparseMoe(sparse_moe) => {
                sparse_moe.forward(hidden_states, residual, tensor_name)
            }
        }
    }

    pub fn is_mlp(&self) -> bool {
        matches!(self, MoeLayer::MLP(_))
    }

    pub fn is_sparse_moe(&self) -> bool {
        matches!(self, MoeLayer::SparseMoe(_))
    }

    pub fn as_mlp(&self) -> Option<&MLP<T>> {
        match self {
            MoeLayer::MLP(mlp) => Some(mlp),
            _ => None,
        }
    }

    pub fn as_sparse_moe(&self) -> Option<&SparseMoeBlock<T>> {
        match self {
            MoeLayer::SparseMoe(sparse_moe) => Some(sparse_moe),
            _ => None,
        }
    }
}





