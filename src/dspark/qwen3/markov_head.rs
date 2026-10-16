//! VanillaMarkov draft head (Rust port of DeepSpec
//! `deepspec/modeling/dspark/markov_head.py::VanillaMarkov`).
//!
//! Architecture note: eLLM builds a static operator graph once and replays it
//! every decoding round. The Python `sample_block_tokens` is an eager
//! `for step_idx in range(proposal_len)` loop; here that loop is *unrolled at
//! graph-build time* into `proposal_len` nodes, so a single call emits the whole
//! draft block. Steps are chained through per-step draft-id buffers: step `k`
//! samples into its own `[batch]` buffer, which step `k + 1` gathers from. This
//! relies on the replay model running operators in enqueue order (each operator
//! completes before the next), so `SampleTokens(k)` precedes `HeadLookup(k + 1)`.
//!
//! Only the `Vanilla` variant is ported. `hidden_states` is dropped by the
//! Vanilla Python (`del`), so it is not part of the graph here either.

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use crate::kernel::common::matmul_params::MatMulParams;
use crate::mem_mgr::mem_pool::{GlobalMemPool, GlobalScratchPool};
use crate::num_traits::{Exp, FromNumber, NegInfinity, Sigmoid, Sqrt};
use crate::tensor::{GlobalOperatorQueue, Tensor};

/// HF safetensors keys of the `VanillaMarkov` head, scoped under a caller
/// supplied `scope` (e.g. `model.draft.markov`). Leaf names end in `.weight` so
/// weight loading fills them (see `mem_pool::REGEX_SET`).
#[derive(Debug, Clone)]
pub struct VanillaMarkovTensorNames {
    pub scope: String,
    pub markov_w1: String,
    pub markov_w2: String,
}

pub fn vanilla_markov_tensor_names(scope: &str) -> VanillaMarkovTensorNames {
    VanillaMarkovTensorNames {
        scope: scope.to_string(),
        markov_w1: format!("{scope}.markov_w1.weight"),
        markov_w2: format!("{scope}.markov_w2.weight"),
    }
}

/// Low-rank Markov bias head: `bias = E[token_prev] @ W2^T`, added to the base
/// logits of each draft step. Weights only; no activations are retained.
#[derive(Clone)]
pub struct VanillaMarkov<T>
where
    T: Copy + PartialOrd,
{
    // nn.Embedding(vocab_size, markov_rank): previous-token lookup table.
    markov_w1: Tensor<T>,
    // nn.Linear(markov_rank, vocab_size, bias=False): stored [out, in] so the
    // tensor matmul computes `latent @ markov_w2^T` -> [batch, vocab].
    markov_w2: Tensor<T>,
    scope_name: String,
}

impl<T> VanillaMarkov<T>
where
    T: Copy
        + PartialOrd
        + Default
        + Sub<Output = T>
        + Neg<Output = T>
        + Add<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Exp
        + NegInfinity
        + Sigmoid
        + Sqrt
        + AddAssign
        + FromNumber
        + GlobalMemPool
        + GlobalOperatorQueue,
{
    pub fn new(vocab_size: usize, markov_rank: usize, names: VanillaMarkovTensorNames) -> Self {
        Self {
            markov_w1: Tensor::zeros(vec![vocab_size, markov_rank], names.markov_w1),
            markov_w2: Tensor::zeros(vec![vocab_size, markov_rank], names.markov_w2),
            scope_name: names.scope,
        }
    }

    /// Graph-build entry point (this module's "forward"): unroll the
    /// `proposal_len` draft steps into one static graph.
    ///
    /// Each step: `lookup(prev token) -> matmul_add(bias + base_logits)
    /// -> sample_tokens(write ids_buf in place)`. All graph nodes reuse the same
    /// `ids_buf` scratch: FIFO replay guarantees `Sample(k)` completes before
    /// `HeadLookup(k + 1)` starts, so overwriting `ids_buf` is safe. `MemPool`
    /// regex stripping collapses `step<K>.` names to a single `MemoryBlock`,
    /// aliasing `latent` and `corrected` across steps as well.
    ///
    /// `base_logits` holds one `[batch, vocab]` tensor per step (the target-model
    /// logits this head corrects); its length is `proposal_len`. `hidden_states`
    /// is intentionally absent — Vanilla drops it.
    ///
    /// Returns every step's corrected logits (all aliasing one buffer, only the
    /// last write is meaningful) and a pointer to the shared `ids_buf` holding
    /// the last step's sampled draft id.
    pub fn sample_block_tokens(
        &self,
        base_logits: &[Tensor<T>], // len == proposal_len, each [batch, vocab]
        first_prev_token_ids: *mut usize, // [batch], token preceding the block
        batch_temperature: *mut T, // [batch]
        batch_size: usize,
    ) -> (Vec<Tensor<T>>, *const usize) {
        let proposal_len = base_logits.len();
        // Per-step draft buffers are addressed one row per batch slot.
        let sequence_stride = 1usize;
        let mut corrected_all = Vec::with_capacity(proposal_len);

        // 单一 in-place `ids_buf`：所有 step 的 Sample 都写这里，HeadLookup(k+1) 也读
        // 这里。走 ScratchPool 同 name 幂复用，无需 N 次 `Box::leak`。
        let ids_buf: *mut usize = usize::with_scratch_pool(|sp| {
            sp.get_init("model.draft.markov.ids_buf", batch_size, 0usize)
        });

        for step_idx in 0..proposal_len {
            // 1) Gather the previous token's latent vector: [batch, markov_rank].
            //    Step 0 reads the external seed; steps 1..N read the shared
            //    `ids_buf` written by the previous step's Sample.
            let prev_ids: *mut usize = if step_idx == 0 {
                first_prev_token_ids
            } else {
                ids_buf
            };
            let latent = Tensor::lookup(
                prev_ids,
                &self.markov_w1,
                batch_size,
                format!("{}.step{}.prev_embedding", self.scope_name, step_idx),
            );

            // 2) Markov bias + residual fuse: latent @ W2^T + base_logits[step].
            //    `matmul_add` now packs via `MemPool::get_packed`; same
            //    `(markov_w2.tensor_name, params)` across steps hits the cache,
            //    so `packed_b` is materialized exactly once for the whole loop.
            //    Weight stays in the pool; later steps re-read the same Arc.
            let corrected = latent.matmul_add(
                &self.markov_w2,
                &base_logits[step_idx],
                MatMulParams {
                    a_row_step_macro: 3,
                    b_row_step_macro: 128,
                    column_step_macro: 64,
                    a_row_step_micro: 3,
                    b_row_step_micro: 32,
                },
                true,
                format!("{}.step{}.output", self.scope_name, step_idx),
            );

            // 3) Sample in place into `ids_buf`. Step `k + 1`'s HeadLookup reads
            //    the value written here; FIFO ordering makes the overwrite safe
            //    because this step's HeadLookup has already finished consuming
            //    whatever was in `ids_buf` previously.
            corrected.sample_tokens(
                ids_buf,
                batch_temperature,
                sequence_stride,
                format!("{}.step{}.sample", self.scope_name, step_idx),
            );

            corrected_all.push(corrected);
        }

        (corrected_all, ids_buf as *const usize)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::runtime::SequenceSlice;
    use std::collections::HashMap;

    const EMPTY_SLICES: &[SequenceSlice] = &[];

    // Build the unrolled block graph for a 2-step draft block and assert shapes;
    // no operator execution.
    #[test]
    fn test_vanilla_markov_build_graph_f32() {
        let vocab_size = 128; // multiple of micro_tile_cols (32)
        let markov_rank = 64; // multiple of reduction_block_cols (64)
        let batch_size = 4;
        let proposal_len = 2;

        f32::init_global(HashMap::new());
        f32::init_operator_queue();

        let head = VanillaMarkov::<f32>::new(
            vocab_size,
            markov_rank,
            vanilla_markov_tensor_names("model.draft.markov"),
        );

        let base_logits: Vec<Tensor<f32>> = (0..proposal_len)
            .map(|k| {
                Tensor::zeros(
                    vec![batch_size, vocab_size],
                    format!("model.draft.base_step{k}.output"),
                )
            })
            .collect();

        let mut first_prev_token_ids = vec![0usize; batch_size];
        let mut batch_temperature = vec![0.0f32; batch_size];

        let (corrected_all, _last_draft_ids) = head.sample_block_tokens(
            &base_logits,
            first_prev_token_ids.as_mut_ptr(),
            batch_temperature.as_mut_ptr(),
            batch_size,
        );

        assert_eq!(corrected_all.len(), proposal_len);
        for corrected in &corrected_all {
            assert_eq!(corrected.shape, vec![batch_size, vocab_size]);
        }

        // proposal_len steps, each contributing lookup + matmul_add + sample.
        f32::with_operator_queue(|queue| assert_eq!(queue.len(), proposal_len * 3));

        // 跨 step 复用：pool key 剥离后，corrected_all 中所有 Tensor 都
        // 指向同一个 MemoryBlock（只有最后一次写入的值有效）。
        let first_out = corrected_all[0].data;
        for corrected in corrected_all.iter().skip(1) {
            assert_eq!(
                first_out, corrected.data,
                "all step corrected tensors must alias the same pool block"
            );
        }

        // ids_buf 全局唯一：返回的指针就是 ScratchPool 里 name 对应的块。
        let sp_ptr = usize::with_scratch_pool(|sp| {
            sp.get_init("model.draft.markov.ids_buf", batch_size, 0usize)
        });
        assert_eq!(
            sp_ptr as *const usize, _last_draft_ids,
            "returned draft pointer must be the shared ids_buf"
        );
    }

    // Drain the built queue once. Left `#[ignore]`: full block-by-block draft
    // execution and HF numerical alignment depend on runtime scheduling of the
    // multi-step buffer chain being wired into the decoder.
    #[test]
    #[ignore = "requires runtime multi-step draft scheduling + HF alignment"]
    fn test_vanilla_markov_execute_f32() {
        let vocab_size = 128;
        let markov_rank = 64;
        let batch_size = 4;
        let proposal_len = 2;

        f32::init_global(HashMap::new());
        f32::init_operator_queue();

        let head = VanillaMarkov::<f32>::new(
            vocab_size,
            markov_rank,
            vanilla_markov_tensor_names("model.draft.markov"),
        );

        let base_logits: Vec<Tensor<f32>> = (0..proposal_len)
            .map(|k| {
                Tensor::zeros(
                    vec![batch_size, vocab_size],
                    format!("model.draft.base_step{k}.output"),
                )
            })
            .collect();

        let mut first_prev_token_ids = vec![0usize; batch_size];
        let mut batch_temperature = vec![0.0f32; batch_size];

        let _ = head.sample_block_tokens(
            &base_logits,
            first_prev_token_ids.as_mut_ptr(),
            batch_temperature.as_mut_ptr(),
            batch_size,
        );

        let thread_num = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        f32::with_operator_queue(|queue| {
            for operator in queue.iter() {
                for thread_id in 0..thread_num {
                    operator.run(
                        batch_size,
                        batch_size,
                        batch_size,
                        batch_size,
                        thread_num,
                        thread_id,
                        EMPTY_SLICES,
                        &mut Vec::new(),
                    );
                }
            }
        });
    }
}
