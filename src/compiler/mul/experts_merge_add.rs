// === runner/experts_merge_add.rs ===
#![allow(non_snake_case)]

use std::f16;
use std::ops::{Add, Mul};
use super::super::super::kernel;
use super::super::super::init::{
    send_sync_ptr::{ConstPtr, MutPtr},
};
use super::super::assign::assign;
use super::mul_trait::MoeMergeTrait; // ← 新 trait

/// Merge num_experts_per_token 个 experts 的输出，并加 residual：
///
/// 输入：
///   input   : [num_tokens, num_experts_per_token, hidden_size]
///   residual: [num_tokens, hidden_size]
///
/// 输出：
///   output  : [num_tokens, hidden_size]
///
/// 其中：num_tokens = sequence_chunk_size * batch_size
///
/// 第三步不做矩阵乘法，只做逐元素加法；
/// 是否使用 SIMD 由 MoeMergeTrait::merge_add 决定。
#[derive(Clone)]
pub struct ExpertsMergeAdd<T> {
    pub input_ptr: ConstPtr<T>,    // [num_tokens, K, H]
    pub residual_ptr: ConstPtr<T>, // [num_tokens, H]
    pub output_ptr: MutPtr<T>,     // [num_tokens, H]

    // 下面两个仅用于 reset gate_routing，保持兼容你原来的数据结构
    pub experts_indicator: MutPtr<bool>, // [num_experts]
    pub indice_ptr: MutPtr<bool>,        // [num_experts, num_tokens]

    pub sequence_chunk_size: usize,
    pub batch_size: usize,             // 注意：这里 batch_size 只是 shape 信息
    pub num_experts: usize,
    pub num_experts_per_token: usize,  // K
    pub hidden_size: usize,            // H
}

impl<T> ExpertsMergeAdd<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    pub fn new(
        input_ptr: *const T,        // [num_tokens, K, H]
        residual_ptr: *const T,     // [num_tokens, H]
        experts_indicator: *mut bool,
        indice_ptr: *mut bool,
        output_ptr: *mut T,         // [num_tokens, H]
        sequence_chunk_size: usize,
        batch_size: usize,
        num_experts: usize,
        num_experts_per_token: usize,
        hidden_size: usize,
    ) -> Self {
        Self {
            input_ptr: ConstPtr { ptr: input_ptr },
            residual_ptr: ConstPtr { ptr: residual_ptr },
            output_ptr: MutPtr { ptr: output_ptr },
            experts_indicator: MutPtr { ptr: experts_indicator },
            indice_ptr: MutPtr { ptr: indice_ptr },
            sequence_chunk_size,
            batch_size,
            num_experts,
            num_experts_per_token,
            hidden_size,
        }
    }

    /// 外层并行逻辑：
    /// - 第一步：各线程按 expert 维度 reset gate_routing（可选）
    /// - 第二步：按 token 维度切片，每个线程负责若干 token 的 merge + residual
    pub fn run(
        &self,
        _position_index: usize,
        _position_interval: usize,
        _batch_size: usize, // 这里与你结构体里的 batch_size 一致，可忽略
        thread_num: usize,
        thread_id: usize,
    ) {
        unsafe {
            let num_tokens = self.sequence_chunk_size * self.batch_size;
            let H = self.hidden_size;
            let K = self.num_experts_per_token;

            // ===== (1) 按 expert 维度 reset gate_routing（与你原来的骨架一致） =====
            if let Some((begin, end)) = assign(self.num_experts, thread_num, thread_id) {
                let experts_indicator_ptr = self.experts_indicator.ptr;
                let indices_ptr = self.indice_ptr.ptr;

                for e in begin..end {
                    if *experts_indicator_ptr.add(e) {
                        *experts_indicator_ptr.add(e) = false;

                        // reset indices_ptr[e, :] 为 0（false）
                        let p = indices_ptr.add(e * num_tokens);
                        // 将 num_tokens 个 bool 置 0；write_bytes 以字节计数，bool=1字节
                        std::ptr::write_bytes(p, 0, num_tokens);
                    }
                }
            }

            // ===== (2) 按 token 维度切片，每个线程负责一部分 token =====
            if let Some((t0, t1)) = assign(num_tokens, thread_num, thread_id) {
                let in_ptr  = self.input_ptr.ptr;     // [num_tokens, K, H]
                let res_ptr = self.residual_ptr.ptr;  // [num_tokens, H]
                let out_ptr = self.output_ptr.ptr;    // [num_tokens, H]

                for t in t0..t1 {
                    // --- 本 token 的 residual / output 行 ---
                    let res_row = res_ptr.add(t * H);
                    let out_row = out_ptr.add(t * H);

                    // 1) 先把 residual 拷到 output（这一版保持标量）
                    for h in 0..H {
                        *out_row.add(h) = *res_row.add(h);
                    }

                    // 2) 再累加 num_experts_per_token 个 experts 的结果
                    let in_token_base = in_ptr.add(t * (K * H)); // 本 token 的 [K, H] 块起点

                    for s in 0..K {
                        let add_row = in_token_base.add(s * H);

                        // 使用 MoeMergeTrait::merge_add 做 out_row += add_row
                        self.merge_add(
                            out_row,        // 输出行（原位累加）
                            add_row,        // 要加的这一行
                            H,              // 长度
                        );
                    }
                }
            }
        }
    }
}

/* ------------------ MoeMergeTrait 默认实现（generic 标量版本） ------------------ */

impl<T> MoeMergeTrait<T> for ExpertsMergeAdd<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    /// 默认标量实现：out[h] += add[h]
    default fn merge_add(
        &self,
        out_row: *mut T,
        add_row: *const T,
        len: usize,
    ) {
        unsafe {
            for h in 0..len {
                let d = *out_row.add(h);
                let a = *add_row.add(h);
                *out_row.add(h) = d + a;
            }
        }
    }
}

/* ------------------ MoeMergeTrait<f16> 专用 AVX-512 实现 ------------------ */

impl MoeMergeTrait<f16> for ExpertsMergeAdd<f16> {
    fn merge_add(
        &self,
        out_row: *mut f16,
        add_row: *const f16,
        len: usize,
    ) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            // 调用内核：dst[i] += add[i]
            kernel::x86_64::f16_512::moe_merge::moe_merge_add(
                out_row,
                add_row,
                len,
            );
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        {
            unreachable!("avx512fp16 required for MoeMergeTrait<f16>::merge_add");
        }
    }
}
