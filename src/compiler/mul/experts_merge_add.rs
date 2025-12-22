// === runner/experts_merge_add.rs ===
#![allow(non_snake_case)]

use std::f16;
use std::ops::{Add, Mul};

use super::super::super::kernel;
use super::super::super::init::send_sync_ptr::{ConstPtr, MutPtr};
use super::super::assign::assign;
use super::mul_trait::MoeMergeTrait;

/// Merge num_experts_per_token 个 experts 的输出，并加 residual：
///
/// input   : [num_tokens, K, H]
/// residual: [num_tokens, H]
/// output  : [num_tokens, H]
///
/// 第三步不做矩阵乘法，只做逐元素加法；
/// 是否使用 SIMD 由 MoeMergeTrait::merge_add 决定。
#[derive(Clone)]
pub struct ExpertsMergeAdd<T> {
    pub input_ptr: ConstPtr<T>,    // [num_tokens, K, H]
    pub residual_ptr: ConstPtr<T>, // [num_tokens, H]
    pub output_ptr: MutPtr<T>,     // [num_tokens, H]

    // reset gate_routing（保持兼容你原来的数据结构）
    pub experts_indicator: MutPtr<bool>, // [num_experts]
    pub indice_ptr: MutPtr<bool>,        // [num_experts, num_tokens]

    pub sequence_chunk_size: usize,
    pub batch_size: usize,
    pub num_experts: usize,
    pub num_experts_per_token: usize, // K
    pub hidden_size: usize,           // H

    /// 是否在 run() 中执行 reset gate_routing（避免每次都做 O(E*tokens)）
    pub reset_gating: bool,
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
        reset_gating: bool,
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
            reset_gating,
        }
    }

    pub fn run(
        &self,
        _position_index: usize,
        _position_interval: usize,
        _batch_size: usize,
        thread_num: usize,
        thread_id: usize,
    ) {
        unsafe {
            let thread_num = thread_num.max(1);

            let num_tokens = self.sequence_chunk_size * self.batch_size;
            let H = self.hidden_size;
            let K = self.num_experts_per_token;

            // ===== (1) 可选 reset gate_routing =====
            if self.reset_gating {
                if let Some((begin, end)) = assign(self.num_experts, thread_num, thread_id) {
                    let experts_indicator_ptr = self.experts_indicator.ptr;
                    let indices_ptr = self.indice_ptr.ptr;

                    for e in begin..end {
                        if *experts_indicator_ptr.add(e) {
                            *experts_indicator_ptr.add(e) = false;

                            // reset indices_ptr[e, :] 为 0（false）
                            let p = indices_ptr.add(e * num_tokens);
                            std::ptr::write_bytes(p, 0, num_tokens); // bool=1 byte
                        }
                    }
                }
            }

            // ===== (2) 按 token 维度切片：merge + residual =====
            if let Some((t0, t1)) = assign(num_tokens, thread_num, thread_id) {
                let in_ptr = self.input_ptr.ptr;     // [num_tokens, K, H]
                let res_ptr = self.residual_ptr.ptr; // [num_tokens, H]
                let out_ptr = self.output_ptr.ptr;   // [num_tokens, H]

                for t in t0..t1 {
                    let res_row = res_ptr.add(t * H);
                    let out_row = out_ptr.add(t * H);

                    // 1) out = residual（标量拷贝，保持你原意）
                    for h in 0..H {
                        *out_row.add(h) = *res_row.add(h);
                    }

                    // 2) out += sum_s input[t,s,:]
                    let in_token_base = in_ptr.add(t * (K * H));
                    for s in 0..K {
                        let add_row = in_token_base.add(s * H);
                        self.merge_add(out_row, add_row, H);
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
    use std::arch::is_x86_feature_detected;
    use std::mem;

    use crate::kernel::generic::from_f32::FromF32;

    #[inline]
    fn f16_from_f32(x: f32) -> f16 {
        <f16 as FromF32>::from_f32(x)
    }

    #[inline]
    fn f32_from_f16(x: f16) -> f32 {
        let bits: u16 = unsafe { mem::transmute(x) };
        let sign = ((bits & 0x8000) as u32) << 16;
        let exp = (bits & 0x7C00) >> 10;
        let mant = bits & 0x03FF;

        let f_bits: u32 = if exp == 0 {
            if mant == 0 {
                sign
            } else {
                let mut e: i32 = -14;
                let mut m = mant as u32;
                while (m & 0x0400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x03FF;
                let exp_f = (e + 127) as u32;
                sign | (exp_f << 23) | (m << 13)
            }
        } else if exp == 0x1F {
            let exp_f = 0xFFu32;
            sign | (exp_f << 23) | ((mant as u32) << 13)
        } else {
            let exp_f = (exp as i32 - 15 + 127) as u32;
            sign | (exp_f << 23) | ((mant as u32) << 13)
        };

        f32::from_bits(f_bits)
    }

    #[inline]
    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn test_merge_add_k1_basic() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }

        let seq = 2usize;
        let batch = 3usize;
        let num_tokens = seq * batch;

        let K = 1usize;
        let H = 64usize;

        let num_experts = 4usize; // 只是 reset 用，不影响 merge

        let mut input = vec![f16_from_f32(0.0); num_tokens * K * H];
        let mut residual = vec![f16_from_f32(0.0); num_tokens * H];
        let mut out = vec![f16_from_f32(0.0); num_tokens * H];

        let mut experts_indicator = vec![false; num_experts];
        let mut indice = vec![false; num_experts * num_tokens];

        for t in 0..num_tokens {
            for h in 0..H {
                residual[t * H + h] = f16_from_f32(0.1 * t as f32 + 0.001 * h as f32);
                input[(t * K + 0) * H + h] = f16_from_f32(0.05 * t as f32 + 0.0007 * h as f32);
            }
        }

        let runner = ExpertsMergeAdd::<f16>::new(
            input.as_ptr(),
            residual.as_ptr(),
            experts_indicator.as_mut_ptr(),
            indice.as_mut_ptr(),
            out.as_mut_ptr(),
            seq,
            batch,
            num_experts,
            K,
            H,
            false, // reset_gating
        );

        runner.run(0, 1, batch, 1, 0);

        for t in 0..num_tokens {
            for h in 0..H {
                let got = f32_from_f16(out[t * H + h]);
                let exp = f32_from_f16(residual[t * H + h]) + f32_from_f16(input[(t * K + 0) * H + h]);
                assert!(approx(got, exp, 5e-2), "mismatch t={}, h={}, got={}, exp={}", t, h, got, exp);
            }
        }
    }

    #[test]
    fn test_merge_add_k3_sum_and_residual() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }

        let num_tokens = 5usize;
        let K = 3usize;
        let H = 64usize;
        let num_experts = 8usize;

        let mut input = vec![f16_from_f32(0.0); num_tokens * K * H];
        let mut residual = vec![f16_from_f32(0.0); num_tokens * H];
        let mut out = vec![f16_from_f32(0.0); num_tokens * H];

        let mut experts_indicator = vec![false; num_experts];
        let mut indice = vec![false; num_experts * num_tokens];

        for t in 0..num_tokens {
            for h in 0..H {
                residual[t * H + h] = f16_from_f32(0.03 * t as f32 + 0.0009 * h as f32);
                for s in 0..K {
                    input[(t * K + s) * H + h] =
                        f16_from_f32(0.01 * (s as f32 + 1.0) + 0.002 * t as f32 + 0.0002 * h as f32);
                }
            }
        }

        let runner = ExpertsMergeAdd::<f16>::new(
            input.as_ptr(),
            residual.as_ptr(),
            experts_indicator.as_mut_ptr(),
            indice.as_mut_ptr(),
            out.as_mut_ptr(),
            1,
            num_tokens,
            num_experts,
            K,
            H,
            false,
        );

        runner.run(0, 1, num_tokens, 1, 0);

        for t in 0..num_tokens {
            for h in 0..H {
                let mut exp = f32_from_f16(residual[t * H + h]);
                for s in 0..K {
                    exp += f32_from_f16(input[(t * K + s) * H + h]);
                }
                let got = f32_from_f16(out[t * H + h]);
                assert!(approx(got, exp, 5e-2), "mismatch t={}, h={}, got={}, exp={}", t, h, got, exp);
            }
        }
    }

    #[test]
    fn test_merge_add_tail_h48() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }

        // 关键：H=48，要求 moe_merge_add(len=48) 内部 tail-safe（不能固定写 64/32 越界）
        let num_tokens = 4usize;
        let K = 2usize;
        let H = 48usize;
        let num_experts = 2usize;

        let mut input = vec![f16_from_f32(0.0); num_tokens * K * H];
        let mut residual = vec![f16_from_f32(0.0); num_tokens * H];
        let mut out = vec![f16_from_f32(0.0); num_tokens * H];

        let mut experts_indicator = vec![false; num_experts];
        let mut indice = vec![false; num_experts * num_tokens];

        for t in 0..num_tokens {
            for h in 0..H {
                residual[t * H + h] = f16_from_f32(0.02 * t as f32 + 0.001 * h as f32);
                for s in 0..K {
                    input[(t * K + s) * H + h] = f16_from_f32(0.01 * (s as f32 + 1.0) + 0.0003 * h as f32);
                }
            }
        }

        let runner = ExpertsMergeAdd::<f16>::new(
            input.as_ptr(),
            residual.as_ptr(),
            experts_indicator.as_mut_ptr(),
            indice.as_mut_ptr(),
            out.as_mut_ptr(),
            1,
            num_tokens,
            num_experts,
            K,
            H,
            false,
        );

        runner.run(0, 1, num_tokens, 1, 0);

        for t in 0..num_tokens {
            for h in 0..H {
                let exp = f32_from_f16(residual[t * H + h])
                    + f32_from_f16(input[(t * K + 0) * H + h])
                    + f32_from_f16(input[(t * K + 1) * H + h]);
                let got = f32_from_f16(out[t * H + h]);
                assert!(approx(got, exp, 5e-2), "tail mismatch t={}, h={}, got={}, exp={}", t, h, got, exp);
            }
        }
    }

    #[test]
    fn test_reset_gating_clears_indicator_and_indice() {
        // reset 逻辑不需要 avx512fp16；但保持一致也可跳过
        let seq = 1usize;
        let batch = 5usize;
        let num_tokens = seq * batch;

        let K = 1usize;
        let H = 16usize;
        let num_experts = 3usize;

        let input = vec![f16_from_f32(0.0); num_tokens * K * H];
        let residual = vec![f16_from_f32(0.0); num_tokens * H];
        let mut out = vec![f16_from_f32(0.0); num_tokens * H];

        let mut experts_indicator = vec![true; num_experts];
        let mut indice = vec![true; num_experts * num_tokens]; // 全 true，期待被清零

        let runner = ExpertsMergeAdd::<f16>::new(
            input.as_ptr(),
            residual.as_ptr(),
            experts_indicator.as_mut_ptr(),
            indice.as_mut_ptr(),
            out.as_mut_ptr(),
            seq,
            batch,
            num_experts,
            K,
            H,
            true, // reset_gating
        );

        // 用 2 线程跑一遍 reset
        runner.run(0, 1, batch, 2, 0);
        runner.run(0, 1, batch, 2, 1);

        for e in 0..num_experts {
            assert_eq!(experts_indicator[e], false, "experts_indicator not cleared at e={}", e);
            for t in 0..num_tokens {
                assert_eq!(indice[e * num_tokens + t], false, "indice not cleared at e={}, t={}", e, t);
            }
        }
    }
}
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
