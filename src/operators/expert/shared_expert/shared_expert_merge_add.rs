// === operators/expert/shared_expert/shared_expert_merge_add.rs ===
#![allow(non_snake_case)]

use std::marker::PhantomData;
use std::ops::{Add, Mul};
use std::sync::atomic::Ordering;

use crate::num_traits::Sigmoid;

use crate::operators::assign::assign;
use crate::operators::expert::expert_routing::ExpertRouting;
use crate::operators::send_sync_ptr::{ConstPtr, MutPtr};
use crate::operators::traits::SharedMergeAddTrait;

// Variable naming used in this operator:
// - active_token_count: number of token rows handled by this run().
// - hidden_size / hidden_index: hidden columns in each token row.
// - num_experts_per_token / topk_slot: number of routed expert outputs merged per token.
// - shared_gate: sigmoid(dot(input_hidden[b], shared_gate_weight)) scaling the shared output.
// - token_id: token row currently being merged.
// 本算子的变量命名约定：
// - active_token_count：本次 run() 实际处理的 token 行数。
// - hidden_size / hidden_index：每个 token 行中的 hidden 列。
// - num_experts_per_token / topk_slot：每个 token 需要合并的 routed expert 输出数量。
// - shared_gate：sigmoid(dot(input_hidden[b], shared_gate_weight))，用于缩放 shared 输出。
// - token_id：当前正在合并的 token 行。

/// Merge routed expert outputs, residual, and the gated shared-expert output.
/// 合并 routed expert 输出、residual 以及带门控的 shared expert 输出。
///
///   out[b] = residual[b]
///            + Σ_slot routed_out[b, slot]
///            + sigmoid(dot(input_hidden[b], shared_gate_weight)) * shared_down[b]
///
/// The shared-expert gate (a Linear(H,1,bias=False)) is evaluated inline as a
/// per-token dot product followed by sigmoid, so no extra operator/tensor is needed.
/// shared expert 门控（Linear(H,1,bias=False)）以每 token 点积 + sigmoid 内联计算，
/// 因此无需额外算子/张量。
///
/// Compute is plain scalar Rust; no f16 / AVX-512 specialization yet.
/// compute 为普通标量 Rust；暂不做 f16 / AVX-512 特化。
#[derive(Clone)]
pub struct SharedExpertMergeAdd<T> {
    pub input_ptr: ConstPtr<T>,    // Routed expert outputs: [num_tokens,K,H].
    pub residual_ptr: ConstPtr<T>, // Residual rows: [num_tokens,H].
    pub shared_down_ptr: ConstPtr<T>, // Shared down output: [num_tokens,H].
    pub shared_gate_weight_ptr: ConstPtr<T>, // Shared gate weight: [H].
    pub input_hidden_ptr: ConstPtr<T>, // MoE block input x: [num_tokens,H].
    pub output_ptr: MutPtr<T>,     // Merged output: [num_tokens,H].

    pub routing: ExpertRouting<T>,

    pub sequence_chunk_size: usize,
    pub batch_size: usize,
    pub num_experts: usize,
    pub num_experts_per_token: usize,
    pub hidden_size: usize,

    /// Whether run() resets routing counters.
    /// 是否在 run() 中重置 routing 计数。
    pub reset_gating: bool,

    pub decode_only_flag: bool,

    _marker: PhantomData<T>,
}

impl<T> SharedExpertMergeAdd<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default + Sigmoid,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_ptr: *const T,              // Routed expert outputs: [num_tokens,K,H].
        residual_ptr: *const T,           // Residual rows: [num_tokens,H].
        shared_down_ptr: *const T,        // Shared down output: [num_tokens,H].
        shared_gate_weight_ptr: *const T, // Shared gate weight: [H].
        input_hidden_ptr: *const T,       // MoE block input x: [num_tokens,H].
        routing: ExpertRouting<T>,
        output_ptr: *mut T, // Merged output: [num_tokens,H].
        sequence_chunk_size: usize,
        batch_size: usize,
        num_experts: usize,
        num_experts_per_token: usize,
        hidden_size: usize,
        reset_gating: bool,
        decode_only_flag: bool,
    ) -> Self {
        Self {
            input_ptr: ConstPtr { ptr: input_ptr },
            residual_ptr: ConstPtr { ptr: residual_ptr },
            shared_down_ptr: ConstPtr {
                ptr: shared_down_ptr,
            },
            shared_gate_weight_ptr: ConstPtr {
                ptr: shared_gate_weight_ptr,
            },
            input_hidden_ptr: ConstPtr {
                ptr: input_hidden_ptr,
            },
            output_ptr: MutPtr { ptr: output_ptr },
            routing,
            sequence_chunk_size,
            batch_size,
            num_experts,
            num_experts_per_token,
            hidden_size,
            reset_gating,
            decode_only_flag,
            _marker: PhantomData,
        }
    }

    pub fn run(
        &self,
        _prefill_size: usize,
        _decode_size: usize,
        _total_size: usize,
        lift_size: usize,
        thread_num: usize,
        thread_id: usize,
    ) {
        unsafe {
            let thread_num = thread_num.max(1);

            let active_size = if self.decode_only_flag {
                lift_size
            } else {
                _total_size
            };
            let active_token_count = self.sequence_chunk_size * active_size;
            let hidden_size = self.hidden_size;
            let num_experts_per_token = self.num_experts_per_token;

            // Reset routing counters before the next routing pass.
            // 在下一轮 routing 前重置 expert 计数。
            if let Some((expert_begin, expert_end)) =
                assign(self.num_experts, thread_num, thread_id)
            {
                for expert_id in expert_begin..expert_end {
                    (&*self.routing.expert_counts.ptr.add(expert_id)).store(0, Ordering::Release);
                }
            }

            // Split by token rows, then merge residual + routed slots + gated shared output.
            // 按 token 行切分，然后合并 residual + routed slot + 带门控的 shared 输出。
            if let Some((token_begin, token_end)) =
                assign(active_token_count, thread_num, thread_id)
            {
                let input_base = self.input_ptr.ptr;
                let residual_base = self.residual_ptr.ptr;
                let shared_down_base = self.shared_down_ptr.ptr;
                let shared_gate_weight = self.shared_gate_weight_ptr.ptr;
                let input_hidden_base = self.input_hidden_ptr.ptr;
                let output_base = self.output_ptr.ptr;

                for token_id in token_begin..token_end {
                    let residual_row = residual_base.add(token_id * hidden_size);
                    let output_row = output_base.add(token_id * hidden_size);

                    // Start from residual.
                    // 先写入 residual。
                    for hidden_index in 0..hidden_size {
                        *output_row.add(hidden_index) = *residual_row.add(hidden_index);
                    }

                    // Add all routed expert slots for this token.
                    // 累加当前 token 的所有 routed expert slot。
                    let input_token_base =
                        input_base.add(token_id * (num_experts_per_token * hidden_size));
                    for topk_slot in 0..num_experts_per_token {
                        let expert_output_row = input_token_base.add(topk_slot * hidden_size);
                        self.merge_add(output_row, expert_output_row, hidden_size);
                    }

                    // Shared-expert gate: sigmoid(dot(x[b], w_sgate)).
                    // shared expert 门控：sigmoid(dot(x[b], w_sgate))。
                    let input_hidden_row = input_hidden_base.add(token_id * hidden_size);
                    let mut gate_acc = T::default();
                    for hidden_index in 0..hidden_size {
                        gate_acc = gate_acc
                            + *input_hidden_row.add(hidden_index)
                                * *shared_gate_weight.add(hidden_index);
                    }
                    let shared_gate = gate_acc.sigmoid();

                    // Add the gated shared-expert output.
                    // 累加带门控的 shared expert 输出。
                    let shared_down_row = shared_down_base.add(token_id * hidden_size);
                    self.merge_add_scaled(output_row, shared_down_row, shared_gate, hidden_size);
                }
            }
        }
    }
}

/* -------------------- SharedMergeAddTrait scalar implementation -------------------- */
/* -------------------- SharedMergeAddTrait 标量实现 -------------------- */

impl<T> SharedMergeAddTrait<T> for SharedExpertMergeAdd<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default + Sigmoid,
{
    /// merge_add: out_row[j] += add_row[j].
    /// merge_add：out_row[j] += add_row[j]。
    fn merge_add(&self, out_row: *mut T, add_row: *const T, len: usize) {
        unsafe {
            for hidden_index in 0..len {
                let output_value = *out_row.add(hidden_index);
                let add_value = *add_row.add(hidden_index);
                *out_row.add(hidden_index) = output_value + add_value;
            }
        }
    }

    /// merge_add_scaled: out_row[j] += add_row[j] * factor.
    /// merge_add_scaled：out_row[j] += add_row[j] * factor。
    fn merge_add_scaled(&self, out_row: *mut T, add_row: *const T, factor: T, len: usize) {
        unsafe {
            for hidden_index in 0..len {
                let output_value = *out_row.add(hidden_index);
                let add_value = *add_row.add(hidden_index);
                *out_row.add(hidden_index) = output_value + add_value * factor;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::expert::expert_routing::empty_routing;

    #[inline]
    fn sigmoid_f32(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    const H: usize = 16;
    const K: usize = 2; // num_experts_per_token
    const E: usize = 2; // num_experts

    unsafe fn set_all_counts(routing: ExpertRouting<f32>, count: usize) {
        for e in 0..routing.num_experts {
            (&*routing.expert_counts.ptr.add(e)).store(count, Ordering::Release);
        }
    }

    fn run_and_verify(batch: usize, run_batch: usize, cpu_num: usize) {
        let num_tokens = batch;

        let mut input = vec![0.0f32; num_tokens * K * H];
        let mut residual = vec![0.0f32; num_tokens * H];
        let mut shared_down = vec![0.0f32; num_tokens * H];
        let mut input_hidden = vec![0.0f32; num_tokens * H];
        let mut shared_gate_weight = vec![0.0f32; H];
        let mut out = vec![0.0f32; num_tokens * H];

        for h in 0..H {
            shared_gate_weight[h] = 0.01 * h as f32 - 0.05;
        }
        for t in 0..num_tokens {
            for h in 0..H {
                residual[t * H + h] = 0.1 * t as f32 + 0.001 * h as f32;
                shared_down[t * H + h] = 0.02 * t as f32 - 0.003 * h as f32 + 0.01;
                input_hidden[t * H + h] = 0.03 * t as f32 + 0.002 * h as f32 - 0.04;
                for s in 0..K {
                    input[t * (K * H) + s * H + h] =
                        0.01 * (s as f32 + 1.0) + 0.002 * t as f32 + 0.0003 * h as f32;
                }
            }
        }

        let routing = unsafe { empty_routing::<f32>(E, num_tokens, K) };
        unsafe { set_all_counts(routing, num_tokens) };

        unsafe {
            let op = SharedExpertMergeAdd::<f32>::new(
                input.as_ptr(),
                residual.as_ptr(),
                shared_down.as_ptr(),
                shared_gate_weight.as_ptr(),
                input_hidden.as_ptr(),
                routing,
                out.as_mut_ptr(),
                1,
                batch,
                E,
                K,
                H,
                true,
                false,
            );
            for tid in 0..cpu_num {
                op.run(run_batch, 0, run_batch, run_batch, cpu_num, tid);
            }
        }

        // Verify run-range rows: residual + Σslots routed + sigmoid(dot)·shared_down.
        for t in 0..run_batch {
            let mut dot = 0.0f32;
            for h in 0..H {
                dot += input_hidden[t * H + h] * shared_gate_weight[h];
            }
            let gate = sigmoid_f32(dot);
            for h in 0..H {
                let mut exp = residual[t * H + h];
                for s in 0..K {
                    exp += input[t * (K * H) + s * H + h];
                }
                exp += gate * shared_down[t * H + h];
                assert!(
                    (out[t * H + h] - exp).abs() < 1e-4,
                    "merge mismatch t={} h={} got={} exp={}",
                    t,
                    h,
                    out[t * H + h],
                    exp
                );
            }
        }

        // Rows beyond run_batch must stay untouched (zero-initialized).
        for t in run_batch..num_tokens {
            for h in 0..H {
                assert!(
                    out[t * H + h].abs() < 1e-6,
                    "out-of-range row written t={} h={} got={}",
                    t,
                    h,
                    out[t * H + h]
                );
            }
        }

        // reset_gating=true must clear all expert counters.
        for e in 0..E {
            let count = unsafe { (&*routing.expert_counts.ptr.add(e)).load(Ordering::Acquire) };
            assert_eq!(count, 0, "expert count not cleared at e={}", e);
        }
    }

    #[test]
    fn test_shared_merge_single_thread() {
        run_and_verify(5, 5, 1);
    }

    #[test]
    fn test_shared_merge_multithread() {
        run_and_verify(8, 8, 4);
    }

    #[test]
    fn test_shared_merge_run_batch_smaller_than_capacity() {
        run_and_verify(6, 3, 2);
    }
}
