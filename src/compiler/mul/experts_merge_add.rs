// === runner/experts_merge_add.rs ===
#![allow(non_snake_case)]

use super::super::super::init::send_sync_ptr::{ConstPtr, MutPtr};
use super::super::super::kernel;
use super::super::assign::assign;
use super::mul_trait::MoeMergeTrait;
use std::f16;
use std::ops::{Add, Mul}; // ← 新 trait

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
    pub batch_size: usize, // 注意：这里 batch_size 只是 shape 信息
    pub num_experts: usize,
    pub num_experts_per_token: usize, // K
    pub hidden_size: usize,           // H
}

impl<T> ExpertsMergeAdd<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    pub fn new(
        input_ptr: *const T,    // [num_tokens, K, H]
        residual_ptr: *const T, // [num_tokens, H]
        experts_indicator: *mut bool,
        indice_ptr: *mut bool,
        output_ptr: *mut T, // [num_tokens, H]
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
            experts_indicator: MutPtr {
                ptr: experts_indicator,
            },
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
                let in_ptr = self.input_ptr.ptr; // [num_tokens, K, H]
                let res_ptr = self.residual_ptr.ptr; // [num_tokens, H]
                let out_ptr = self.output_ptr.ptr; // [num_tokens, H]

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
                            out_row, // 输出行（原位累加）
                            add_row, // 要加的这一行
                            H,       // 长度
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
    default fn merge_add(&self, out_row: *mut T, add_row: *const T, len: usize) {
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
    fn merge_add(&self, out_row: *mut f16, add_row: *const f16, len: usize) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            // 调用内核：dst[i] += add[i]
            kernel::x86_64::f16_512::moe_merge::moe_merge_add(out_row, add_row, len);
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        {
            unreachable!("avx512fp16 required for MoeMergeTrait<f16>::merge_add");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    // 纯 Rust 参考实现：验证 Output = Residual + Sum(Experts)
    fn reference_implementation(
        input: &[f16],    // [Tokens, K, H]
        residual: &[f16], // [Tokens, H]
        num_tokens: usize,
        k: usize,
        h: usize,
    ) -> Vec<f16> {
        let mut output = vec![0.0 as f16; num_tokens * h];
        for t in 0..num_tokens {
            for i in 0..h {
                // Start with residual
                let mut acc = residual[t * h + i] as f32;
                // Add each expert's contribution
                for s in 0..k {
                    let val = input[t * (k * h) + s * h + i];
                    acc += val as f32;
                }
                output[t * h + i] = acc as f16;
            }
        }
        output
    }

    #[test]
    fn test_experts_merge_add_correctness() {
        let sequence_chunk_size = 16;
        let batch_size = 4;
        let num_tokens = sequence_chunk_size * batch_size;
        let hidden = 64;
        let num_experts = 8;
        let k = 2; // experts per token
        let num_threads = 4;

        let mut rng = rand::thread_rng();

        // 1. 准备数据
        // Input: [Tokens, K, H]
        let input: Vec<f16> = (0..num_tokens * k * hidden)
            .map(|_| rng.gen_range(-0.5..0.5) as f16)
            .collect();
        // Residual: [Tokens, H]
        let residual: Vec<f16> = (0..num_tokens * hidden)
            .map(|_| rng.gen_range(-0.5..0.5) as f16)
            .collect();
        // Output: [Tokens, H]
        let mut output = vec![0.0 as f16; num_tokens * hidden];

        // 2. 准备需要被 Reset 的 Flags
        // 填充为 true，验证运行后是否被清零
        let mut experts_indicator = vec![true; num_experts];
        let mut indice_ptr = vec![true; num_experts * num_tokens];

        // 3. 运行算子
        unsafe {
            let op = ExpertsMergeAdd::new(
                input.as_ptr(),
                residual.as_ptr(),
                experts_indicator.as_mut_ptr(),
                indice_ptr.as_mut_ptr(),
                output.as_mut_ptr(),
                sequence_chunk_size,
                batch_size,
                num_experts,
                k,
                hidden,
            );

            // 模拟多线程运行
            for tid in 0..num_threads {
                op.run(0, 0, batch_size, num_threads, tid);
            }
        }

        // 4. 验证计算结果 (Merge + Residual)
        let ref_out = reference_implementation(&input, &residual, num_tokens, k, hidden);
        let tolerance = 0.05; // f16 误差容忍度

        for i in 0..output.len() {
            let val = output[i] as f32;
            let ref_val = ref_out[i] as f32;
            let diff = (val - ref_val).abs();

            if diff > tolerance {
                panic!(
                    "Mismatch at index {}: Got {}, Expected {}, Diff {}",
                    i, val, ref_val, diff
                );
            }
        }

        // 5. 验证 Flags 是否被清零 (Reset Logic)
        for (i, &val) in experts_indicator.iter().enumerate() {
            assert!(!val, "experts_indicator[{}] was not reset to false", i);
        }
        // indice_ptr 只有在 experts_indicator 为 true 的行才会被清零
        // 但我们在测试开始时把 experts_indicator 全设为 true 了，所以应该全被清零
        for (i, &val) in indice_ptr.iter().enumerate() {
            assert!(!val, "indice_ptr[{}] was not reset to false", i);
        }

        println!("ExpertsMergeAdd test passed!");
    }
}
