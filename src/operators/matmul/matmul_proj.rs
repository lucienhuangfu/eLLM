use std::marker::PhantomData;
use std::ops::{Add, Mul};

use crate::operators::assign::assign;
use crate::operators::send_sync_ptr::{ConstPtr, MutPtr};
use crate::operators::traits::MatMulTrait;

// Fused input projection GEMM for GatedDeltaNet-style linear attention layers.
// It replaces the four separate projections (in_proj_qkv / in_proj_z /
// in_proj_b / in_proj_a) with a single GEMM whose weights are concatenated
// along the output dimension, so the shared hidden_states input is read once.
// The gate preparation (beta / g) is fused as a per-row epilogue on the
// b and a segments of the output.
// 面向 GatedDeltaNet 类线性注意力层的融合输入投影 GEMM。
// 用一次 GEMM 取代 in_proj_qkv / in_proj_z / in_proj_b / in_proj_a 四个独立投影，
// 权重沿输出维拼接，共享的 hidden_states 输入只读一遍。
// 门控准备（beta / g）作为行 epilogue 融合在输出的 b 段和 a 段上。
//
// Output layout per row: [mixed_qkv (qkv_cols) | z (z_cols) | beta (head_cols) | g (head_cols)].
// Epilogue on the last two segments, applied in place:
//   beta[h] = sigmoid(b[row, h])
//   g[row, h] = -exp(a_log[h]) * softplus(a[row, h] + dt_bias[h])
// 每行输出布局：[mixed_qkv (qkv_cols) | z (z_cols) | beta (head_cols) | g (head_cols)]。
// 最后两段原地应用 epilogue：
//   beta[h] = sigmoid(b[row, h])
//   g[row, h] = -exp(a_log[h]) * softplus(a[row, h] + dt_bias[h])

#[derive(Clone)]
pub struct MatMulProj<T> {
    pub ptr1: ConstPtr<T>,     // Input matrix A: [input_rows, reduction_cols].
    pub ptr2: ConstPtr<T>,     // Fused weight B_nt: [output_cols, reduction_cols].
    pub output_ptr: MutPtr<T>, // Output matrix C: [input_rows, output_cols].

    // Per-head gate parameters consumed by the epilogue.
    // epilogue 使用的逐 head 门控参数。
    pub dt_bias_ptr: ConstPtr<T>, // dt_bias[head_cols]
    pub a_log_ptr: ConstPtr<T>,   // A_log[head_cols]

    // Column widths of the four fused projection segments.
    // 四个融合投影段的列宽。
    pub qkv_cols: usize,  // key_dim * 2 + value_dim
    pub z_cols: usize,    // value_dim
    pub head_cols: usize, // num_v_heads, shared by the b and a segments

    pub m_max: usize,
    pub n_max: usize,
    pub k_max: usize,
    pub _marker: PhantomData<T>,
}

impl<T> MatMulProj<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    /// Create the fused projection operator from the concatenated B weights.
    /// 从拼接好的 B 权重创建融合投影算子。
    ///
    /// B is expected as B_nt[output_cols, reduction_cols] with the rows
    /// ordered as qkv | z | b | a.
    /// B 要求传入 B_nt[output_cols, reduction_cols]，行序为 qkv | z | b | a。
    pub unsafe fn new(
        input_ptr: *const T,  // A[input_rows, reduction_cols]
        weight_ptr: *const T, // B_nt[output_cols, reduction_cols]
        output_ptr: *mut T,   // C[input_rows, output_cols]
        dt_bias_ptr: *const T,
        a_log_ptr: *const T,
        qkv_cols: usize,
        z_cols: usize,
        head_cols: usize,
        m_max: usize,
        k_max: usize,
    ) -> Self {
        let n_max = qkv_cols + z_cols + head_cols * 2;

        Self {
            ptr1: ConstPtr { ptr: input_ptr },
            ptr2: ConstPtr { ptr: weight_ptr },
            output_ptr: MutPtr { ptr: output_ptr },
            dt_bias_ptr: ConstPtr { ptr: dt_bias_ptr },
            a_log_ptr: ConstPtr { ptr: a_log_ptr },
            qkv_cols,
            z_cols,
            head_cols,
            m_max,
            n_max,
            k_max,
            _marker: PhantomData,
        }
    }

    pub fn run(
        &self,
        prefill_size: usize,
        decode_size: usize,
        _total_size: usize,
        thread_num: usize,
        thread_id: usize,
    ) {
        let active_input_rows = if prefill_size == 0 {
            decode_size
        } else {
            prefill_size
        };

        debug_assert!(active_input_rows <= self.m_max);
        debug_assert!(thread_num >= 1);
        debug_assert!(thread_id < thread_num);

        if let Some((row_begin, row_end)) = assign(active_input_rows, thread_num, thread_id) {
            for row in row_begin..row_end {
                unsafe {
                    let input_row_ptr = self.ptr1.ptr.add(row * self.k_max);
                    let output_row_ptr = self.output_ptr.ptr.add(row * self.n_max);
                    // TODO: compute one fused projection row
                    // C[row, :] = A[row, :] @ B_nt^T over all qkv | z | b | a columns,
                    // then apply the beta / g epilogue in place on the b and a segments.
                    self.compute(input_row_ptr, self.ptr2.ptr, output_row_ptr);
                }
            }
        }
    }
}

impl<T> MatMulTrait<T> for MatMulProj<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    default fn compute(&self, _input_ptr1: *const T, _input_ptr2: *const T, _output_ptr: *mut T) {
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
        const N: usize = QKV_COLS + Z_COLS + HEAD_COLS * 2;

        let input_data: Vec<f32> = (0..M * K).map(|x| (x % 7) as f32 * 0.01).collect();
        let weight_data: Vec<f32> = (0..N * K).map(|x| (x % 11) as f32 * 0.01).collect();
        let dt_bias: Vec<f32> = vec![1.0f32; HEAD_COLS];
        let a_log: Vec<f32> = (0..HEAD_COLS).map(|x| x as f32 * 0.1).collect();
        let mut output_data = vec![0.0f32; M * N];

        let operator = unsafe {
            MatMulProj::<f32>::new(
                input_data.as_ptr(),
                weight_data.as_ptr(),
                output_data.as_mut_ptr(),
                dt_bias.as_ptr(),
                a_log.as_ptr(),
                QKV_COLS,
                Z_COLS,
                HEAD_COLS,
                M,
                K,
            )
        };
        assert_eq!(operator.n_max, N);

        // While compute is empty: only assert no panic / partition coverage.
        // compute 为空期间：只验证不 panic、分区覆盖完整。
        let thread_num = 4;
        for thread_id in 0..thread_num {
            operator.run(M, 0, M, thread_num, thread_id);
        }
        assert!(output_data.iter().all(|&value| value == 0.0));
    }
}
