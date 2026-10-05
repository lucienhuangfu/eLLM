use std::marker::PhantomData;
use std::ops::{Add, Mul};

use crate::operators::assign::assign;
use crate::operators::send_sync_ptr::{ConstPtr, MutPtr};
use crate::operators::traits::MatMulProjTrait;
use crate::runtime::SequenceSlice;

// Fused input projection for GatedDeltaNet-style linear attention layers.
// It schedules the four projections (in_proj_qkv / in_proj_z / in_proj_b /
// in_proj_a) in one operator over the shared hidden_states input, but each
// projection keeps its own weight matrix and its own output buffer: all
// inputs and outputs are split, only the scheduling is fused.
// The computation is split into four independent GEMMs (never merged into
// one), with the gate preparation fused as epilogues:
// 面向 GatedDeltaNet 类线性注意力层的融合输入投影。
// 在一个算子内调度 in_proj_qkv / in_proj_z / in_proj_b / in_proj_a 四个投影，
// 共享 hidden_states 输入，但每个投影保留各自的权重矩阵与输出缓冲：
// 输入和输出全部分开，只融合调度。
// 计算拆成四次独立的矩阵乘法（不合并成一次），并把门控准备融合为 epilogue：
//
// Per-row outputs land in four separate buffers: mixed_qkv[qkv_cols],
// z[z_cols], beta[head_cols], g[head_cols]. mixed_qkv is stored like the
// KV cache in MatMul3 (k_state_ptr / v_state_ptr): one row per
// (sequence_position, batch) pair so the qkv of every token is kept for
// the whole sequence, indexed as
// qkv_output[(next_sequence_index * batch_size + batch_index) * qkv_cols].
// z / beta / g stay row-local and are indexed by token_index.
// 每行输出写入四个独立的缓冲：mixed_qkv[qkv_cols]、z[z_cols]、
// beta[head_cols]、g[head_cols]。mixed_qkv 与 MatMul3 的 KV cache
// （k_state_ptr / v_state_ptr）采用相同的存储方式：每个 (序列位置, batch)
// 一行，整条序列中每个 token 的 qkv 都保留，寻址为
// qkv_output[(next_sequence_index * batch_size + batch_index) * qkv_cols]。
// z / beta / g 仍按 token_index 行内寻址。
// The four GEMMs per row and their epilogues:
//   1. qkv segment: plain GEMM, no epilogue.
//   2. z segment: plain GEMM, no epilogue.
//   3. b segment: GEMM fused with sigmoid, beta[h] = sigmoid(b[row, h]).
//   4. a segment: GEMM fused with the decay gate,
//      g[row, h] = -exp(a_log[h]) * softplus(a[row, h] + dt_bias[h]).
// 每行的四次矩阵乘法及其 epilogue：
//   1. qkv 段：纯乘法，无 epilogue。
//   2. z 段：纯乘法，无 epilogue。
//   3. b 段：乘法融合 sigmoid，beta[h] = sigmoid(b[row, h])。
//   4. a 段：乘法融合衰减门，
//      g[row, h] = -exp(a_log[h]) * softplus(a[row, h] + dt_bias[h])。

#[derive(Clone)]
pub struct MatMulProj<T> {
    pub ptr1: ConstPtr<T>, // Shared input matrix A: [input_rows, reduction_cols].

    // One weight matrix B_nt[segment_cols, reduction_cols] per projection.
    // 每个投影一份权重矩阵 B_nt[段列宽, reduction_cols]。
    pub qkv_weight_ptr: ConstPtr<T>,
    pub z_weight_ptr: ConstPtr<T>,
    pub b_weight_ptr: ConstPtr<T>,
    pub a_weight_ptr: ConstPtr<T>,

    // One output matrix per projection. The qkv output is a per-token cache
    // laid out like the KV cache in MatMul3:
    // C[sequence_length * batch_size, qkv_cols]; the other three are
    // row-local: C[input_rows, segment_cols].
    // 每个投影一份输出。qkv 输出是与 MatMul3 KV cache 同布局的逐 token 缓存：
    // C[sequence_length * batch_size, qkv_cols]；其余三份按行寻址：
    // C[input_rows, 段列宽]。
    pub qkv_output_ptr: MutPtr<T>,
    pub z_output_ptr: MutPtr<T>,
    pub b_output_ptr: MutPtr<T>, // beta after the sigmoid epilogue
    pub a_output_ptr: MutPtr<T>, // g after the decay-gate epilogue

    // Per-head gate parameters consumed by the epilogue.
    // epilogue 使用的逐 head 门控参数。
    pub dt_bias_ptr: ConstPtr<T>, // dt_bias[head_cols]
    pub a_log_ptr: ConstPtr<T>,   // A_log[head_cols]

    // Column widths of the four projection segments.
    // 四个投影段的列宽。
    pub qkv_cols: usize,  // key_dim * 2 + value_dim
    pub z_cols: usize,    // value_dim
    pub head_cols: usize, // num_v_heads, shared by the b and a segments

    // Shape of the qkv cache: sequence_length * batch_size rows.
    // qkv 缓存的形状：共 sequence_length * batch_size 行。
    pub sequence_length: usize,
    pub batch_size: usize,

    pub m_max: usize,
    pub k_max: usize,
    pub _marker: PhantomData<T>,
}

impl<T> MatMulProj<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    /// Create the projection operator from the four separate weight matrices.
    /// 从四份独立的权重矩阵创建投影算子。
    ///
    /// Each weight is B_nt[segment_cols, reduction_cols]; z / b / a outputs
    /// are C[input_rows, segment_cols]; the qkv output is a per-token cache
    /// C[sequence_length * batch_size, qkv_cols].
    /// 每份权重为 B_nt[段列宽, reduction_cols]；z / b / a 输出为
    /// C[input_rows, 段列宽]；qkv 输出是逐 token 缓存
    /// C[sequence_length * batch_size, qkv_cols]。
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        input_ptr: *const T, // A[input_rows, reduction_cols]
        qkv_weight_ptr: *const T,
        z_weight_ptr: *const T,
        b_weight_ptr: *const T,
        a_weight_ptr: *const T,
        qkv_output_ptr: *mut T,
        z_output_ptr: *mut T,
        b_output_ptr: *mut T,
        a_output_ptr: *mut T,
        dt_bias_ptr: *const T,
        a_log_ptr: *const T,
        qkv_cols: usize,
        z_cols: usize,
        head_cols: usize,
        sequence_length: usize,
        batch_size: usize,
        m_max: usize,
        k_max: usize,
    ) -> Self {
        Self {
            ptr1: ConstPtr { ptr: input_ptr },
            qkv_weight_ptr: ConstPtr {
                ptr: qkv_weight_ptr,
            },
            z_weight_ptr: ConstPtr { ptr: z_weight_ptr },
            b_weight_ptr: ConstPtr { ptr: b_weight_ptr },
            a_weight_ptr: ConstPtr { ptr: a_weight_ptr },
            qkv_output_ptr: MutPtr {
                ptr: qkv_output_ptr,
            },
            z_output_ptr: MutPtr { ptr: z_output_ptr },
            b_output_ptr: MutPtr { ptr: b_output_ptr },
            a_output_ptr: MutPtr { ptr: a_output_ptr },
            dt_bias_ptr: ConstPtr { ptr: dt_bias_ptr },
            a_log_ptr: ConstPtr { ptr: a_log_ptr },
            qkv_cols,
            z_cols,
            head_cols,
            sequence_length,
            batch_size,
            m_max,
            k_max,
            _marker: PhantomData,
        }
    }

    // Collect the active token rows from attention_list, like MatMul3::build_row_map.
    // Each entry carries (token_index, batch_index, next_sequence_index) so
    // the qkv output can be placed at its cache row.
    // 从 attention_list 收集有效 token 行，与 MatMul3::build_row_map 一致。
    // 每行同时记录 (token_index, batch_index, next_sequence_index)，
    // 供 qkv 输出按缓存行落位。
    #[inline(always)]
    fn build_row_map(&self, attention_list: &[SequenceSlice]) -> Vec<(usize, usize, usize)> {
        let mut rows = Vec::new();
        for slice in attention_list {
            for offset in 0..slice.length {
                let token_index = slice.token_start_index + offset;
                let next_sequence_index = slice.next_sequence_index + offset;
                if token_index >= self.m_max || next_sequence_index >= self.sequence_length {
                    continue;
                }
                rows.push((token_index, slice.batch_index, next_sequence_index));
            }
        }
        rows
    }

    pub fn run(
        &self,
        _total_size: usize,
        attention_list: &[SequenceSlice],
        thread_num: usize,
        thread_id: usize,
    ) where
        Self: MatMulProjTrait<T>,
    {
        debug_assert!(thread_num >= 1);
        debug_assert!(thread_id < thread_num);

        let row_map = self.build_row_map(attention_list);
        let row_count = row_map.len();
        if row_count == 0 {
            return;
        }

        // Split work into (row, GEMM) tasks: four independent matrix
        // multiplications per row, never merged into one.
        // 把任务拆成 (行, 乘法) 粒度：每行四次独立的矩阵乘法，绝不合并。
        let total_tasks = row_count * 4;
        if let Some((task_begin, task_end)) = assign(total_tasks, thread_num, thread_id) {
            for task_id in task_begin..task_end {
                let (row, batch_index, next_sequence_index) = row_map[task_id % row_count];
                let segment_id = task_id / row_count;

                unsafe {
                    let input_row_ptr = self.ptr1.ptr.add(row * self.k_max);

                    match segment_id {
                        0 => {
                            // qkv segment: one plain GEMM, no epilogue. The
                            // result is stored at the cache row of this
                            // (sequence_position, batch) pair, same layout
                            // as the KV cache in MatMul3.
                            // qkv 段：一次纯乘法，无 epilogue。结果写入该
                            // (序列位置, batch) 对应的缓存行，与 MatMul3
                            // 的 KV cache 布局一致。
                            let cache_row = next_sequence_index * self.batch_size + batch_index;
                            self.compute(
                                input_row_ptr,
                                self.qkv_weight_ptr.ptr,
                                self.qkv_output_ptr.ptr.add(cache_row * self.qkv_cols),
                                self.qkv_cols,
                            );
                        }
                        1 => {
                            // z segment: one plain GEMM, no epilogue.
                            // z 段：一次纯乘法，无 epilogue。
                            self.compute(
                                input_row_ptr,
                                self.z_weight_ptr.ptr,
                                self.z_output_ptr.ptr.add(row * self.z_cols),
                                self.z_cols,
                            );
                        }
                        2 => {
                            // b segment: GEMM fused with sigmoid -> beta.
                            // b 段：乘法融合 sigmoid，原地得到 beta。
                            self.compute_sigmoid_b(
                                input_row_ptr,
                                self.b_weight_ptr.ptr,
                                self.b_output_ptr.ptr.add(row * self.head_cols),
                                self.head_cols,
                            );
                        }
                        _ => {
                            // a segment: GEMM fused with the decay gate -> g.
                            // a 段：乘法融合衰减门，原地得到 g。
                            self.compute_gate_a(
                                input_row_ptr,
                                self.a_weight_ptr.ptr,
                                self.a_output_ptr.ptr.add(row * self.head_cols),
                                self.head_cols,
                            );
                        }
                    }
                }
            }
        }
    }
}

impl<T> MatMulProjTrait<T> for MatMulProj<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    // Plain GEMM: C = A @ B_nt^T over output_cols columns of one segment.
    // 纯乘法：对一个段的 output_cols 列做一次 GEMM。
    #[inline]
    default fn compute(
        &self,
        _input_ptr1: *const T,
        _input_ptr2: *const T,
        _output_ptr: *mut T,
        _output_cols: usize,
    ) {
        // TODO: compute logic, filled in later
    }

    // GEMM with the b-segment epilogue fused in place (see the file header):
    //   beta[h] = sigmoid(b[row, h])
    // 乘法混合 sigmoid，作用于输出的 b 段，原地得到 beta：
    //   beta[h] = sigmoid(b[row, h])
    #[inline]
    default fn compute_sigmoid_b(
        &self,
        _input_ptr1: *const T,
        _input_ptr2: *const T,
        _output_ptr: *mut T,
        _output_cols: usize,
    ) {
        // TODO: compute logic, filled in later
    }

    // GEMM with the a-segment epilogue fused in place (see the file header):
    //   g[row, h] = -exp(A_log[h]) * softplus(a[row, h] + dt_bias[h])
    // 乘法混合衰减门（softplus + exp 缩放），作用于输出的 a 段，原地得到 g：
    //   g[row, h] = -exp(A_log[h]) * softplus(a[row, h] + dt_bias[h])
    #[inline]
    default fn compute_gate_a(
        &self,
        _input_ptr1: *const T,
        _input_ptr2: *const T,
        _output_ptr: *mut T,
        _output_cols: usize,
    ) {
        // TODO: compute logic, filled in later
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_proj_construct_and_partition() {
        const M: usize = 4;
        const K: usize = 8;
        const QKV_COLS: usize = 12;
        const Z_COLS: usize = 4;
        const HEAD_COLS: usize = 2;

        let input_data: Vec<f32> = (0..M * K).map(|x| (x % 7) as f32 * 0.01).collect();
        let qkv_weight: Vec<f32> = (0..QKV_COLS * K).map(|x| (x % 11) as f32 * 0.01).collect();
        let z_weight: Vec<f32> = (0..Z_COLS * K).map(|x| (x % 13) as f32 * 0.01).collect();
        let b_weight: Vec<f32> = (0..HEAD_COLS * K).map(|x| (x % 5) as f32 * 0.01).collect();
        let a_weight: Vec<f32> = (0..HEAD_COLS * K).map(|x| (x % 3) as f32 * 0.01).collect();
        let dt_bias: Vec<f32> = vec![1.0f32; HEAD_COLS];
        let a_log: Vec<f32> = (0..HEAD_COLS).map(|x| x as f32 * 0.1).collect();
        // qkv 输出按缓存分配：sequence_length * batch_size 行。
        // qkv output is allocated as a cache: sequence_length * batch_size rows.
        const SEQUENCE_LENGTH: usize = M;
        const BATCH_SIZE: usize = 1;
        let mut qkv_output = vec![0.0f32; SEQUENCE_LENGTH * BATCH_SIZE * QKV_COLS];
        let mut z_output = vec![0.0f32; M * Z_COLS];
        let mut b_output = vec![0.0f32; M * HEAD_COLS];
        let mut a_output = vec![0.0f32; M * HEAD_COLS];

        let operator = unsafe {
            MatMulProj::<f32>::new(
                input_data.as_ptr(),
                qkv_weight.as_ptr(),
                z_weight.as_ptr(),
                b_weight.as_ptr(),
                a_weight.as_ptr(),
                qkv_output.as_mut_ptr(),
                z_output.as_mut_ptr(),
                b_output.as_mut_ptr(),
                a_output.as_mut_ptr(),
                dt_bias.as_ptr(),
                a_log.as_ptr(),
                QKV_COLS,
                Z_COLS,
                HEAD_COLS,
                SEQUENCE_LENGTH,
                BATCH_SIZE,
                M,
                K,
            )
        };

        // While compute is empty: only assert no panic / partition coverage.
        // compute 为空期间：只验证不 panic、分区覆盖完整。
        let thread_num = 4;
        let attention_list = [SequenceSlice {
            token_start_index: 0,
            batch_index: 0,
            next_sequence_index: 0,
            length: M,
            last_token_flag: true,
            lift_index: 0,
        }];
        for thread_id in 0..thread_num {
            operator.run(M, &attention_list, thread_num, thread_id);
        }
        assert!(qkv_output.iter().all(|&value| value == 0.0));
        assert!(z_output.iter().all(|&value| value == 0.0));
        assert!(b_output.iter().all(|&value| value == 0.0));
        assert!(a_output.iter().all(|&value| value == 0.0));
    }
}
