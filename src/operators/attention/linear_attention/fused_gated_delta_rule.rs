use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};

use crate::num_traits::{Exp, Sigmoid, Sqrt};
use crate::operators::assign::assign;
use crate::operators::send_sync_ptr::{ConstPtr, MutPtr};
use crate::operators::traits::FusedGatedDeltaRuleTrait;
use crate::runtime::SequenceSlice;

// Fused GatedDeltaNet-style linear attention operator, collapsing the whole
// per-layer pipeline into a single operator:
// 融合的 GatedDeltaNet 线性注意力算子，把整条层内流水线压成单算子：
//
//   hidden -> four input projections (qkv / z / b / a, with the
//             beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)
//             epilogue, modeling_qwen3_5.py L555-L562, L607-L609)
//          -> depthwise causal conv1d + SiLU (L239-L279)
//          -> qk l2-norm (q additionally scaled by 1/sqrt(head_k_dim))
//          -> gated delta rule, chunked two-phase formulas of
//             torch_chunk_gated_delta_rule (L290-L424) restructured into a
//             single interleaved loop: one iteration = one chunk compute
//             (per-token projection/conv/qk-norm into scratch + chunk-local
//             cum_decay / ut_system / intra_chunk_attn / forward
//             substitution) + one recurrence step (v_new / o / S update,
//             L407-L415); no whole-sequence precomputation pass exists
//          -> gated RMSNorm epilogue (L207-L224)
//          -> attention output [sequence_length * batch_size, value_dim].
//
//   hidden -> 四路输入投影（qkv / z / b / a，含 beta = sigmoid(b)、
//             g = -exp(A_log) * softplus(a + dt_bias) epilogue）
//          -> depthwise 因果卷积 + SiLU
//          -> qk l2 归一化（q 额外乘 1/sqrt(head_k_dim)）
//          -> gated delta rule：采用 torch_chunk_gated_delta_rule 的
//             chunk 两阶段公式，但重排为单循环交错——每轮迭代 =
//             一个 chunk 计算（逐 token 投影/卷积/qk-norm 进 scratch +
//             chunk 内 cum_decay / ut_system / intra_chunk_attn /
//             前代求解）+ 一步递归（v_new / o / S 更新）；
//             不存在任何跨 chunk 的全序列预计算 pass
//          -> gated RMSNorm epilogue
//          -> 注意力输出 [sequence_length * batch_size, value_dim]。
//
// Task partitioning: task count = batch_size * num_v_heads; run() splits
// tasks with assign(). task_id -> batch = task_id / num_v_heads,
// h = task_id % num_v_heads; the q/k head of v head h follows the GQA
// mapping kv_head = h * num_k_heads / num_v_heads (repeat_interleave).
// A task owns its slices' rows strictly in order (conv window and
// recurrence are sequential); under GQA the q/k projection and conv of a
// shared kv_head are recomputed per task on purpose, trading redundant
// FLOPs for zero synchronization and zero intermediate global buffers.
// 任务划分：任务数 = batch_size * num_v_heads，run() 用 assign() 切分。
// task_id -> batch = task_id / num_v_heads、h = task_id % num_v_heads；
// v 头 h 对应的 q/k 头按 GQA 映射 kv_head = h * num_k_heads / num_v_heads
// （对应 repeat_interleave）。任务内严格按序处理所属 slice 的行
// （卷积窗口与递推均为时序依赖）；GQA 下共享 kv_head 的 q/k 投影与卷积
// 有意按任务重复计算，以冗余 FLOPs 换取零同步、零中间全局缓冲。
//
// conv_state uses a per-task-contiguous layout used only by this fused
// path:
//   [batch_size, num_v_heads, 2 * head_k_dim + head_v_dim, kernel_size - 1]
// with the in-segment order q(head_k_dim) | k(head_k_dim) | v(head_v_dim),
// so task (batch, h) owns one contiguous block and rolls the window in
// place without scratch copies. conv_weight[conv_dim, kernel_size] keeps
// the original channel order; compute indexes three sub-ranges (q starts
// at kv_head * head_k_dim, k at key_dim + kv_head * head_k_dim, v at
// 2 * key_dim + h * head_v_dim).
// conv_state 采用仅融合路径使用的逐任务连续布局：
//   [batch_size, num_v_heads, 2 * head_k_dim + head_v_dim, kernel_size - 1]
// 段内顺序 q(head_k_dim) | k(head_k_dim) | v(head_v_dim)，任务 (batch, h)
// 独占一段连续内存、原地滚动窗口，无需 scratch 拷贝。权重
// conv_weight[conv_dim, kernel_size] 保持原通道序，compute 用三个子区间
// 索引（q 段 kv_head * head_k_dim 起、k 段 key_dim + kv_head * head_k_dim
// 起、v 段 2 * key_dim + h * head_v_dim 起）。
//
// recurrent_state [batch_size, num_v_heads, head_k_dim, head_v_dim] is the
// layer cache, updated in place; the updated S of a chunk feeds the next
// iteration directly. Inference only: no backward-related constraints, the
// UT solve is fixed to forward substitution, and compute should include a
// chunk_len == 1 fast path (decode) that degenerates to the token-wise
// recurrence formulas.
// recurrent_state [batch_size, num_v_heads, head_k_dim, head_v_dim] 为层
// 缓存、原地更新；chunk 更新后的 S 直接供下一轮迭代使用。仅服务推理：
// 无反向传播约束，UT 求解固定为前代（forward substitution）形态，
// compute 实现时应包含 chunk_len == 1 快分支（decode），退化为逐 token
// 递推公式。
//
// Known trade-offs: hidden rows are re-read and re-projected (GEMV) per
// head task, GQA multiplies the q/k part by num_v_heads / num_k_heads;
// parallelism is capped at batch_size * num_v_heads and chunks run
// strictly sequentially, but the chunk-local matricization ([C, C] UT
// forward substitution, small GEMMs) keeps the prefill bulk on SIMD; the
// scratch is ~106 KB per thread (f32, C = 64, head_dim = 128, two [C, C]
// blocks), L2-resident, single-chunk sized and independent of sequence
// length.
// 已知权衡：hidden 行被每个头任务重复读取并重复 GEMV 投影，GQA 下 q/k
// 部分再乘 num_v_heads / num_k_heads；并行度上限 = batch_size *
// num_v_heads，chunk 间严格串行，但 chunk 内矩阵化（[C, C] UT 前代、
// 小 GEMM）让 prefill 主体计算吃上 SIMD；scratch 约 106 KB/线程
// （f32、C = 64、head_dim = 128、含两块 [C, C]），常驻 L2，只需单
// chunk 容量、不随序列长度增长。

// Per-thread scratch pool, allocated once at construction (thread_num
// lanes) so run() stays allocation-free; the layout mirrors
// full_attention/scratch.rs (Box<[T]> pools + thread_buffers views).
// 线程私有 scratch 池，构造时按 thread_num 一次性预分配，run() 零分配；
// 布局仿 full_attention/scratch.rs（Box<[T]> 池 + thread_buffers 视图）。
#[derive(Clone)]
pub struct FusedScratch<T> {
    pool: Box<[T]>,
    lane_stride: usize,
    qk_stride: usize,     // chunk lanes for q and k: [chunk_size, head_k_dim] each
    v_stride: usize, // chunk lanes for v / z / new_values / k_cumdecay / o: [chunk_size, head_v_dim] each
    cc_stride: usize, // chunk-local matrices: ut (pairwise_decay -> ut_system) and intra_chunk_attn, [chunk_size, chunk_size] each
    scalar_stride: usize, // chunk lanes for beta / g / cum_decay: [chunk_size] each
    thread_num: usize,
}

// One thread's view into the pool: every buffer is exactly one chunk wide.
// 单线程的池视图：所有缓冲均为单 chunk 容量。
pub struct FusedScratchSlice<'a, T> {
    pub q: &'a mut [T],                // [chunk_size, head_k_dim]
    pub k: &'a mut [T],                // [chunk_size, head_k_dim]
    pub v: &'a mut [T],                // [chunk_size, head_v_dim]
    pub z: &'a mut [T],                // [chunk_size, head_v_dim]
    pub new_values: &'a mut [T],       // [chunk_size, head_v_dim]
    pub k_cumdecay: &'a mut [T],       // [chunk_size, head_v_dim]
    pub o: &'a mut [T],                // [chunk_size, head_v_dim]
    pub ut: &'a mut [T], // [chunk_size, chunk_size]: pairwise_decay, then ut_system in place
    pub intra_chunk_attn: &'a mut [T], // [chunk_size, chunk_size]
    pub beta: &'a mut [T], // [chunk_size]
    pub g: &'a mut [T],  // [chunk_size]
    pub cum_decay: &'a mut [T], // [chunk_size]
}

impl<T> FusedScratch<T>
where
    T: Copy + Default,
{
    pub fn new(thread_num: usize, chunk_size: usize, head_k_dim: usize, head_v_dim: usize) -> Self {
        let qk_stride = chunk_size * head_k_dim;
        let v_stride = chunk_size * head_v_dim;
        let cc_stride = chunk_size * chunk_size;
        let scalar_stride = chunk_size;
        // 2 qk lanes + 5 v lanes + 2 [C, C] matrices + 3 scalar lanes.
        // 2 条 qk 通道 + 5 条 v 通道 + 两块 [C, C] 矩阵 + 3 条标量通道。
        let lane_stride = 2 * qk_stride + 5 * v_stride + 2 * cc_stride + 3 * scalar_stride;

        Self {
            pool: vec![T::default(); thread_num * lane_stride].into_boxed_slice(),
            lane_stride,
            qk_stride,
            v_stride,
            cc_stride,
            scalar_stride,
            thread_num,
        }
    }

    // Lane layout inside one thread's region (element offsets):
    //   q | k                       2 * qk_stride
    //   v | z | new_values | k_cumdecay | o   5 * v_stride
    //   ut | intra_chunk_attn       2 * cc_stride
    //   beta | g | cum_decay        3 * scalar_stride
    // 单线程区内的通道布局（元素偏移）：q | k 两条 qk 通道，
    // v | z | new_values | k_cumdecay | o 五条 v 通道，
    // ut | intra_chunk_attn 两块 [C, C] 矩阵，beta | g | cum_decay 三条标量通道。
    #[inline(always)]
    pub fn thread_buffers(&self, thread_id: usize) -> FusedScratchSlice<'_, T> {
        debug_assert!(thread_id < self.thread_num);
        unsafe {
            let base = self.pool.as_ptr().add(thread_id * self.lane_stride) as *mut T;
            let part =
                |offset: usize, len: usize| std::slice::from_raw_parts_mut(base.add(offset), len);
            let qk = self.qk_stride;
            let v = self.v_stride;
            let cc = self.cc_stride;
            let sc = self.scalar_stride;
            let v_base = 2 * qk;
            let cc_base = v_base + 5 * v;
            let sc_base = cc_base + 2 * cc;
            FusedScratchSlice {
                q: part(0, qk),
                k: part(qk, qk),
                v: part(v_base, v),
                z: part(v_base + v, v),
                new_values: part(v_base + 2 * v, v),
                k_cumdecay: part(v_base + 3 * v, v),
                o: part(v_base + 4 * v, v),
                ut: part(cc_base, cc),
                intra_chunk_attn: part(cc_base + cc, cc),
                beta: part(sc_base, sc),
                g: part(sc_base + sc, sc),
                cum_decay: part(sc_base + 2 * sc, sc),
            }
        }
    }
}

#[derive(Clone)]
pub struct FusedGatedDeltaRule<T> {
    // Inputs.
    // 输入。
    pub hidden_ptr: ConstPtr<T>,      // [m_max, hidden_size] token rows
    pub qkv_weight_ptr: ConstPtr<T>,  // [conv_dim, hidden_size], q | k | v channel order
    pub z_weight_ptr: ConstPtr<T>,    // [value_dim, hidden_size]
    pub b_weight_ptr: ConstPtr<T>,    // [num_v_heads, hidden_size]
    pub a_weight_ptr: ConstPtr<T>,    // [num_v_heads, hidden_size]
    pub conv_weight_ptr: ConstPtr<T>, // [conv_dim, kernel_size]
    pub norm_weight_ptr: ConstPtr<T>, // [head_v_dim], gated RMSNorm weight
    pub dt_bias_ptr: ConstPtr<T>,     // [num_v_heads]
    pub a_log_ptr: ConstPtr<T>,       // [num_v_heads]
    // States, updated in place.
    // 状态，原地更新。
    pub conv_state_ptr: MutPtr<T>, // [batch_size, num_v_heads, 2 * head_k_dim + head_v_dim, kernel_size - 1]
    pub recurrent_state_ptr: MutPtr<T>, // [batch_size, num_v_heads, head_k_dim, head_v_dim]
    // Output.
    // 输出。
    pub output_ptr: MutPtr<T>, // [sequence_length * batch_size, value_dim]

    pub hidden_size: usize,
    pub key_dim: usize,   // num_k_heads * head_k_dim
    pub value_dim: usize, // num_v_heads * head_v_dim
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub kernel_size: usize,
    pub chunk_size: usize, // delta rule chunking constant, aligned with the reference default 64

    // Shape of the hidden input / attention output caches.
    // hidden 输入与注意力输出缓存的形状。
    pub sequence_length: usize,
    pub batch_size: usize,
    pub m_max: usize, // hidden row capacity (token_index bound), matching MatMulProj
    pub k_max: usize, // hidden row width (reduction cols), matching MatMulProj

    pub thread_num: usize, // scratch lane count fixed at construction
    pub scratch: FusedScratch<T>,
    pub _marker: PhantomData<T>,
}

impl<T> FusedGatedDeltaRule<T>
where
    T: Copy
        + Default
        + PartialOrd
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Exp
        + Sigmoid
        + Sqrt,
{
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        hidden_ptr: *const T,        // [m_max, hidden_size]
        qkv_weight_ptr: *const T,    // [conv_dim, hidden_size]
        z_weight_ptr: *const T,      // [value_dim, hidden_size]
        b_weight_ptr: *const T,      // [num_v_heads, hidden_size]
        a_weight_ptr: *const T,      // [num_v_heads, hidden_size]
        conv_weight_ptr: *const T,   // [conv_dim, kernel_size]
        norm_weight_ptr: *const T,   // [head_v_dim]
        dt_bias_ptr: *const T,       // [num_v_heads]
        a_log_ptr: *const T,         // [num_v_heads]
        conv_state_ptr: *mut T, // [batch_size, num_v_heads, 2 * head_k_dim + head_v_dim, kernel_size - 1]
        recurrent_state_ptr: *mut T, // [batch_size, num_v_heads, head_k_dim, head_v_dim]
        output_ptr: *mut T,     // [sequence_length * batch_size, value_dim]
        hidden_size: usize,
        key_dim: usize,
        value_dim: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        kernel_size: usize,
        chunk_size: usize,
        sequence_length: usize,
        batch_size: usize,
        m_max: usize,
        k_max: usize,
        thread_num: usize,
    ) -> Self {
        debug_assert!(head_k_dim > 0);
        debug_assert!(head_v_dim > 0);
        debug_assert!(num_k_heads > 0);
        debug_assert!(kernel_size >= 1);
        debug_assert!(chunk_size >= 1);
        debug_assert!(thread_num >= 1);
        debug_assert_eq!(num_v_heads % num_k_heads, 0);
        debug_assert_eq!(key_dim, num_k_heads * head_k_dim);
        debug_assert_eq!(value_dim, num_v_heads * head_v_dim);

        Self {
            hidden_ptr: ConstPtr { ptr: hidden_ptr },
            qkv_weight_ptr: ConstPtr {
                ptr: qkv_weight_ptr,
            },
            z_weight_ptr: ConstPtr { ptr: z_weight_ptr },
            b_weight_ptr: ConstPtr { ptr: b_weight_ptr },
            a_weight_ptr: ConstPtr { ptr: a_weight_ptr },
            conv_weight_ptr: ConstPtr {
                ptr: conv_weight_ptr,
            },
            norm_weight_ptr: ConstPtr {
                ptr: norm_weight_ptr,
            },
            dt_bias_ptr: ConstPtr { ptr: dt_bias_ptr },
            a_log_ptr: ConstPtr { ptr: a_log_ptr },
            conv_state_ptr: MutPtr {
                ptr: conv_state_ptr,
            },
            recurrent_state_ptr: MutPtr {
                ptr: recurrent_state_ptr,
            },
            output_ptr: MutPtr { ptr: output_ptr },
            hidden_size,
            key_dim,
            value_dim,
            head_k_dim,
            head_v_dim,
            num_k_heads,
            num_v_heads,
            kernel_size,
            chunk_size,
            sequence_length,
            batch_size,
            m_max,
            k_max,
            thread_num,
            scratch: FusedScratch::new(thread_num, chunk_size, head_k_dim, head_v_dim),
            _marker: PhantomData,
        }
    }

    // Row width of the qkv projection: q + k + v segments.
    // qkv 投影的行宽：q + k + v 三段。
    #[inline(always)]
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim + self.value_dim
    }

    // Conv channels owned by one task in the per-head conv_state layout.
    // 逐头 conv_state 布局下单个任务拥有的卷积通道数。
    #[inline(always)]
    pub fn conv_channels_per_head(&self) -> usize {
        2 * self.head_k_dim + self.head_v_dim
    }

    pub fn run(
        &self,
        _total_size: usize,
        computing_slices: &[SequenceSlice],
        thread_num: usize,
        thread_id: usize,
    ) where
        Self: FusedGatedDeltaRuleTrait<T>,
    {
        debug_assert!(thread_num >= 1);
        debug_assert!(thread_id < thread_num);
        debug_assert_eq!(thread_num, self.thread_num);

        if computing_slices.is_empty() {
            return;
        }

        // Tasks are (batch, v_head) pairs; each task keeps the full time
        // axis of its batch, so no cross-thread state sharing exists.
        // 任务为 (batch, v_head) 对；每个任务持有其 batch 的完整时间轴，
        // 不存在跨线程状态共享。
        let total_tasks = self.batch_size * self.num_v_heads;
        if let Some((task_begin, task_end)) = assign(total_tasks, thread_num, thread_id) {
            let mut scratch = self.scratch.thread_buffers(thread_id);
            for task_id in task_begin..task_end {
                let batch = task_id / self.num_v_heads;
                let head = task_id % self.num_v_heads;
                // Fast path: this batch's slices are contiguous (the usual
                // scheduler layout), pass the subslice without allocating.
                // 快路径：该 batch 的 slice 连续（调度器常见布局），
                // 直接传子切片、零分配。
                let first = computing_slices
                    .iter()
                    .position(|slice| slice.batch_index == batch);
                if let Some(begin) = first {
                    let end = begin
                        + computing_slices[begin..]
                            .iter()
                            .take_while(|slice| slice.batch_index == batch)
                            .count();
                    let total = computing_slices
                        .iter()
                        .filter(|slice| slice.batch_index == batch)
                        .count();
                    if total == end - begin {
                        self.compute(batch, head, &computing_slices[begin..end], &mut scratch);
                        continue;
                    }
                }
                // General path: gather the batch's scattered slices.
                // 通用路径：收集该 batch 分散的 slice。
                let owned: Vec<SequenceSlice> = computing_slices
                    .iter()
                    .filter(|slice| slice.batch_index == batch)
                    .cloned()
                    .collect();
                if owned.is_empty() {
                    continue;
                }
                self.compute(batch, head, &owned, &mut scratch);
            }
        }
    }
}

impl<T> FusedGatedDeltaRuleTrait<T> for FusedGatedDeltaRule<T>
where
    T: Copy
        + Default
        + PartialOrd
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Exp
        + Sigmoid
        + Sqrt,
{
    default fn compute(
        &self,
        _batch: usize,
        _head: usize,
        _slices: &[SequenceSlice],
        _scratch: &mut FusedScratchSlice<'_, T>,
    ) {
        // TODO: compute logic, filled in later
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Small-dimension skeleton test: hidden=8, num_k_heads=1, num_v_heads=2,
    // head_k_dim=2, head_v_dim=2, kernel=4, seq=4, batch=1 -> 2 tasks.
    // 小维度骨架测试：任务数 = batch * num_v_heads = 2。
    #[test]
    fn test_fused_gated_delta_rule_construct_and_partition() {
        const HIDDEN: usize = 8;
        const NUM_K_HEADS: usize = 1;
        const NUM_V_HEADS: usize = 2; // GQA ratio 2, exercises the kv_head mapping
        const HEAD_K_DIM: usize = 2;
        const HEAD_V_DIM: usize = 2;
        const KEY_DIM: usize = NUM_K_HEADS * HEAD_K_DIM;
        const VALUE_DIM: usize = NUM_V_HEADS * HEAD_V_DIM;
        const CONV_DIM: usize = 2 * KEY_DIM + VALUE_DIM;
        const KERNEL_SIZE: usize = 4;
        const CHUNK_SIZE: usize = 64;
        const SEQUENCE_LENGTH: usize = 4;
        const BATCH_SIZE: usize = 1;

        let hidden_data: Vec<f32> = (0..SEQUENCE_LENGTH * BATCH_SIZE * HIDDEN)
            .map(|x| (x % 5) as f32 * 0.01)
            .collect();
        let qkv_weight_data = vec![0.0f32; CONV_DIM * HIDDEN];
        let z_weight_data = vec![0.0f32; VALUE_DIM * HIDDEN];
        let b_weight_data = vec![0.0f32; NUM_V_HEADS * HIDDEN];
        let a_weight_data = vec![0.0f32; NUM_V_HEADS * HIDDEN];
        let conv_weight_data = vec![0.0f32; CONV_DIM * KERNEL_SIZE];
        let norm_weight_data = vec![1.0f32; HEAD_V_DIM];
        let dt_bias_data = vec![0.0f32; NUM_V_HEADS];
        let a_log_data = vec![0.0f32; NUM_V_HEADS];
        let mut conv_state_data =
            vec![
                0.0f32;
                BATCH_SIZE * NUM_V_HEADS * (2 * HEAD_K_DIM + HEAD_V_DIM) * (KERNEL_SIZE - 1)
            ];
        let mut recurrent_state_data =
            vec![0.0f32; BATCH_SIZE * NUM_V_HEADS * HEAD_K_DIM * HEAD_V_DIM];
        let mut output_data = vec![0.0f32; SEQUENCE_LENGTH * BATCH_SIZE * VALUE_DIM];

        let thread_num = 4;
        let operator = unsafe {
            FusedGatedDeltaRule::<f32>::new(
                hidden_data.as_ptr(),
                qkv_weight_data.as_ptr(),
                z_weight_data.as_ptr(),
                b_weight_data.as_ptr(),
                a_weight_data.as_ptr(),
                conv_weight_data.as_ptr(),
                norm_weight_data.as_ptr(),
                dt_bias_data.as_ptr(),
                a_log_data.as_ptr(),
                conv_state_data.as_mut_ptr(),
                recurrent_state_data.as_mut_ptr(),
                output_data.as_mut_ptr(),
                HIDDEN,
                KEY_DIM,
                VALUE_DIM,
                HEAD_K_DIM,
                HEAD_V_DIM,
                NUM_K_HEADS,
                NUM_V_HEADS,
                KERNEL_SIZE,
                CHUNK_SIZE,
                SEQUENCE_LENGTH,
                BATCH_SIZE,
                SEQUENCE_LENGTH * BATCH_SIZE,
                HIDDEN,
                thread_num,
            )
        };
        assert_eq!(operator.conv_dim(), CONV_DIM);
        assert_eq!(
            operator.conv_channels_per_head(),
            2 * HEAD_K_DIM + HEAD_V_DIM
        );

        // compute is still empty: running all threads over the slice must
        // not panic and must leave states and output untouched.
        // compute 仍为空：所有线程跑完该 slice 不应 panic，
        // 状态与输出保持原样。
        let computing_slices = [SequenceSlice {
            global_start_index: 0,
            batch_index: 0,
            local_next_index: 0,
            local_length: SEQUENCE_LENGTH,
            last_token_flag: true,
            lift_index: 0,
        }];
        for thread_id in 0..thread_num {
            operator.run(SEQUENCE_LENGTH, &computing_slices, thread_num, thread_id);
        }
        assert!(conv_state_data.iter().all(|&value| value == 0.0));
        assert!(recurrent_state_data.iter().all(|&value| value == 0.0));
        assert!(output_data.iter().all(|&value| value == 0.0));
    }

    // Sequence longer than chunk_size: exercises the multi-chunk loop and
    // the tail-chunk partitioning with a second slice / batch.
    // 序列长度超过 chunk_size：覆盖多 chunk 循环与尾部 chunk 分块，
    // 并用第二个 slice / batch。
    #[test]
    fn test_fused_gated_delta_rule_multi_chunk_partition() {
        const HIDDEN: usize = 16;
        const NUM_K_HEADS: usize = 2;
        const NUM_V_HEADS: usize = 2;
        const HEAD_K_DIM: usize = 4;
        const HEAD_V_DIM: usize = 4;
        const KEY_DIM: usize = NUM_K_HEADS * HEAD_K_DIM;
        const VALUE_DIM: usize = NUM_V_HEADS * HEAD_V_DIM;
        const CONV_DIM: usize = 2 * KEY_DIM + VALUE_DIM;
        const KERNEL_SIZE: usize = 4;
        const CHUNK_SIZE: usize = 64;
        const SEQUENCE_LENGTH: usize = 160; // 2 full chunks + one tail chunk of 32
        const BATCH_SIZE: usize = 2;

        let hidden_data = vec![0.0f32; SEQUENCE_LENGTH * BATCH_SIZE * HIDDEN];
        let qkv_weight_data = vec![0.0f32; CONV_DIM * HIDDEN];
        let z_weight_data = vec![0.0f32; VALUE_DIM * HIDDEN];
        let b_weight_data = vec![0.0f32; NUM_V_HEADS * HIDDEN];
        let a_weight_data = vec![0.0f32; NUM_V_HEADS * HIDDEN];
        let conv_weight_data = vec![0.0f32; CONV_DIM * KERNEL_SIZE];
        let norm_weight_data = vec![1.0f32; HEAD_V_DIM];
        let dt_bias_data = vec![0.0f32; NUM_V_HEADS];
        let a_log_data = vec![0.0f32; NUM_V_HEADS];
        let mut conv_state_data =
            vec![
                0.0f32;
                BATCH_SIZE * NUM_V_HEADS * (2 * HEAD_K_DIM + HEAD_V_DIM) * (KERNEL_SIZE - 1)
            ];
        let mut recurrent_state_data =
            vec![0.0f32; BATCH_SIZE * NUM_V_HEADS * HEAD_K_DIM * HEAD_V_DIM];
        let mut output_data = vec![0.0f32; SEQUENCE_LENGTH * BATCH_SIZE * VALUE_DIM];

        let thread_num = 2;
        let operator = unsafe {
            FusedGatedDeltaRule::<f32>::new(
                hidden_data.as_ptr(),
                qkv_weight_data.as_ptr(),
                z_weight_data.as_ptr(),
                b_weight_data.as_ptr(),
                a_weight_data.as_ptr(),
                conv_weight_data.as_ptr(),
                norm_weight_data.as_ptr(),
                dt_bias_data.as_ptr(),
                a_log_data.as_ptr(),
                conv_state_data.as_mut_ptr(),
                recurrent_state_data.as_mut_ptr(),
                output_data.as_mut_ptr(),
                HIDDEN,
                KEY_DIM,
                VALUE_DIM,
                HEAD_K_DIM,
                HEAD_V_DIM,
                NUM_K_HEADS,
                NUM_V_HEADS,
                KERNEL_SIZE,
                CHUNK_SIZE,
                SEQUENCE_LENGTH,
                BATCH_SIZE,
                SEQUENCE_LENGTH * BATCH_SIZE,
                HIDDEN,
                thread_num,
            )
        };

        let computing_slices = [
            SequenceSlice {
                global_start_index: 0,
                batch_index: 0,
                local_next_index: 0,
                local_length: SEQUENCE_LENGTH,
                last_token_flag: true,
                lift_index: 0,
            },
            SequenceSlice {
                global_start_index: SEQUENCE_LENGTH,
                batch_index: 1,
                local_next_index: 0,
                local_length: SEQUENCE_LENGTH,
                last_token_flag: true,
                lift_index: 0,
            },
        ];
        for thread_id in 0..thread_num {
            operator.run(SEQUENCE_LENGTH, &computing_slices, thread_num, thread_id);
        }
        assert!(conv_state_data.iter().all(|&value| value == 0.0));
        assert!(recurrent_state_data.iter().all(|&value| value == 0.0));
        assert!(output_data.iter().all(|&value| value == 0.0));
    }
}
