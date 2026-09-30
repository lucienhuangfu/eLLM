use crate::transformer::attention::Attention;
use crate::transformer::dense_mlp::DenseMlp;
use crate::transformer::sparse_moe::SparseMoe;

pub enum AttentionBlock<T>
where
    T: Copy + PartialOrd,
{
    Full(Attention<T>),
    SlidingWindow(Attention<T>),
}

pub enum FfnBlock<T>
where
    T: Copy + PartialOrd,
{
    Dense(DenseMlp<T>),
    SparseMoe(SparseMoe<T>),
}
