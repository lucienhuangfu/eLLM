// === operators/experts/experts_merge_add.rs ===
#![allow(non_snake_case)]

use std::f16;
use std::ops::{Add, Mul};

use crate::common::send_sync_ptr::{ConstPtr, MutPtr};
use crate::kernel;
use crate::operators::assign::assign;
use crate::operators::traits::MoeMergeTrait;

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

    pub decode_only_flag: bool,
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
        decode_only_flag: bool,
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
            decode_only_flag,
        }
    }

    pub fn run(&self, prefill_size: usize, _decode_size: usize, thread_num: usize, thread_id: usize) {
        unsafe {
            let thread_num = thread_num.max(1);

            let num_tokens = self.sequence_chunk_size * prefill_size;
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
                let in_ptr = self.input_ptr.ptr; // [num_tokens, K, H]
                let res_ptr = self.residual_ptr.ptr; // [num_tokens, H]
                let out_ptr = self.output_ptr.ptr; // [num_tokens, H]

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

    #[inline]
    fn approx_eq_f32(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn verify_output(out: &[f16], out_ref: &[f32], tol: f32, msg: &str) {
        for i in 0..out.len() {
            let got = out[i] as f32;
            let exp = out_ref[i];
            assert!(
                approx_eq_f32(got, exp, tol),
                "{} mismatch at {}: got={}, exp={}",
                msg,
                i,
                got,
                exp
            );
        }
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

        let num_experts = 4usize;

        let mut input = vec![0.0 as f16; num_tokens * K * H];
        let mut residual = vec![0.0 as f16; num_tokens * H];
        let mut out = vec![0.0 as f16; num_tokens * H];

        let mut experts_indicator = vec![false; num_experts];
        let mut indice = vec![false; num_experts * num_tokens];

        let mut out_ref = vec![0.0f32; num_tokens * H];

        for t in 0..num_tokens {
            for h in 0..H {
                let r_val = 0.1 * t as f32 + 0.001 * h as f32;
                let i_val = 0.05 * t as f32 + 0.0007 * h as f32;
                residual[t * H + h] = r_val as f16;
                input[(t * K + 0) * H + h] = i_val as f16;
                out_ref[t * H + h] = r_val + i_val;
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
            false,
            false,
        );

        runner.run(batch, 0, 1, 0);

        verify_output(&out, &out_ref, 5e-2, "k1_basic");
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

        let mut input = vec![0.0 as f16; num_tokens * K * H];
        let mut residual = vec![0.0 as f16; num_tokens * H];
        let mut out = vec![0.0 as f16; num_tokens * H];

        let mut experts_indicator = vec![false; num_experts];
        let mut indice = vec![false; num_experts * num_tokens];

        let mut out_ref = vec![0.0f32; num_tokens * H];

        for t in 0..num_tokens {
            for h in 0..H {
                let r_val = 0.03 * t as f32 + 0.0009 * h as f32;
                residual[t * H + h] = r_val as f16;

                let mut sum_k = 0.0f32;
                for s in 0..K {
                    let val = 0.01 * (s as f32 + 1.0) + 0.002 * t as f32 + 0.0002 * h as f32;
                    input[(t * K + s) * H + h] = val as f16;
                    sum_k += val;
                }
                out_ref[t * H + h] = r_val + sum_k;
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
            false,
        );

        runner.run(num_tokens, 0, 1, 0);

        verify_output(&out, &out_ref, 5e-2, "k3_sum");
    }

    #[test]
    fn test_merge_add_tail_h48() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }

        let num_tokens = 4usize;
        let K = 2usize;
        let H = 48usize;
        let num_experts = 2usize;

        let mut input = vec![0.0 as f16; num_tokens * K * H];
        let mut residual = vec![0.0 as f16; num_tokens * H];
        let mut out = vec![0.0 as f16; num_tokens * H];

        let mut experts_indicator = vec![false; num_experts];
        let mut indice = vec![false; num_experts * num_tokens];

        let mut out_ref = vec![0.0f32; num_tokens * H];

        for t in 0..num_tokens {
            for h in 0..H {
                let r_val = 0.02 * t as f32 + 0.001 * h as f32;
                residual[t * H + h] = r_val as f16;

                let mut sum_k = 0.0f32;
                for s in 0..K {
                    let val = 0.01 * (s as f32 + 1.0) + 0.0003 * h as f32;
                    input[(t * K + s) * H + h] = val as f16;
                    sum_k += val;
                }
                out_ref[t * H + h] = r_val + sum_k;
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
            false,
        );

        runner.run(num_tokens, 0, 1, 0);

        verify_output(&out, &out_ref, 5e-2, "tail_h48");
    }

    #[test]
    fn test_reset_gating_clears_indicator_and_indice() {
        let seq = 1usize;
        let batch = 5usize;
        let num_tokens = seq * batch;

        let K = 1usize;
        let H = 16usize;
        let num_experts = 3usize;

        let input = vec![0.0 as f16; num_tokens * K * H];
        let residual = vec![0.0 as f16; num_tokens * H];
        let mut out = vec![0.0 as f16; num_tokens * H];

        let mut experts_indicator = vec![true; num_experts];
        let mut indice = vec![true; num_experts * num_tokens];

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
            true,
            false,
        );

        runner.run(batch, 0, 2, 0);
        runner.run(batch, 0, 2, 1);

        for e in 0..num_experts {
            assert_eq!(experts_indicator[e], false, "experts_indicator not cleared at e={}", e);
            for t in 0..num_tokens {
                assert_eq!(
                    indice[e * num_tokens + t],
                    false,
                    "indice not cleared at e={}, t={}",
                    e,
                    t
                );
            }
        }
    }

    #[test]
    fn test_merge_add_multithreaded_correctness() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }

        let sequence_chunk_size = 16;
        let batch_size = 4;
        let num_tokens = sequence_chunk_size * batch_size;
        let hidden = 64;
        let num_experts = 8;
        let k = 2;
        let num_threads = 4;

        let mut input = vec![0.0 as f16; num_tokens * k * hidden];
        let mut residual = vec![0.0 as f16; num_tokens * hidden];
        let mut output = vec![0.0 as f16; num_tokens * hidden];
        let mut out_ref = vec![0.0f32; num_tokens * hidden];

        for t in 0..num_tokens {
            for h in 0..hidden {
                let r_val = ((t + h) % 100) as f32 * 0.01 - 0.5;
                residual[t * hidden + h] = r_val as f16;

                let mut sum_k = 0.0f32;
                for s in 0..k {
                    let val = ((t * k + s + h) % 100) as f32 * 0.01 - 0.5;
                    input[t * (k * hidden) + s * hidden + h] = val as f16;
                    sum_k += val;
                }
                out_ref[t * hidden + h] = r_val + sum_k;
            }
        }

        let mut experts_indicator = vec![true; num_experts];
        let mut indice_ptr = vec![true; num_experts * num_tokens];

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
                true,
                false,
            );

            for tid in 0..num_threads {
                op.run(batch_size, 0, num_threads, tid);
            }
        }

        verify_output(&output, &out_ref, 5e-2, "multithreaded");

        for (i, &val) in experts_indicator.iter().enumerate() {
            assert!(!val, "experts_indicator[{}] was not reset to false", i);
        }
        for (i, &val) in indice_ptr.iter().enumerate() {
            assert!(!val, "indice_ptr[{}] was not reset to false", i);
        }
    }
    #[test]
fn test_merge_add_respects_run_batch_smaller_than_capacity() {
    if !std::arch::is_x86_feature_detected!("avx512fp16") {
        eprintln!("skip: avx512fp16 not detected");
        return;
    }

    use crate::common::num_traits::FromNumber;

    #[inline]
    fn f16_from_f32(x: f32) -> f16 {
        <f16 as FromNumber>::from_f32(x)
    }

    #[inline]
    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    let seq = 2usize;
    let batch_cap = 5usize; // new() capacity
    let batch_run = 3usize; // run() 实际 batch（更小）

    let num_tokens_cap = seq * batch_cap; // 10
    let num_tokens_run = seq * batch_run; // 6

    let k = 2usize;
    let h = 32usize; // 用 32，避免你 moe_merge_add 的 tail 行为影响判断

    // input: [num_tokens_cap, K, H]
    let mut input = vec![f16_from_f32(0.0); num_tokens_cap * k * h];
    // residual: [num_tokens_cap, H]
    let mut residual = vec![f16_from_f32(0.0); num_tokens_cap * h];

    // output 先填 sentinel，要求 run() 不要写到 run tokens 之外
    let sentinel = f16_from_f32(7.75);
    let mut output = vec![sentinel; num_tokens_cap * h];

    // 只给前 num_tokens_run 填真实数据；后面的保持 0（或者随便）
    for t in 0..num_tokens_run {
        for j in 0..h {
            let r = 0.1 * t as f32 + 0.001 * j as f32;
            residual[t * h + j] = f16_from_f32(r);

            for s in 0..k {
                let v = 0.01 * (s as f32 + 1.0) + 0.002 * t as f32 + 0.0003 * j as f32;
                input[t * (k * h) + s * h + j] = f16_from_f32(v);
            }
        }
    }

    // gating 相关这里无所谓（merge_add 本身不依赖），随便给点空间
    let num_experts = 4usize;
    let mut experts_indicator = vec![false; num_experts];
    let mut indice = vec![false; num_experts * num_tokens_cap];

    let op = ExpertsMergeAdd::<f16>::new(
        input.as_ptr(),
        residual.as_ptr(),
        experts_indicator.as_mut_ptr(),
        indice.as_mut_ptr(),
        output.as_mut_ptr(),
        seq,
        batch_cap,     // capacity batch
        num_experts,
        k,
        h,
        false,         // reset_gating 不参与本测试
        false,
    );

    // 单线程运行：关键是把 batch_run 传进去
    // 你的 run() 现在把 _batch_size 忽略了，所以这个测试会失败（会写满 10 tokens）
    op.run(batch_run, 0, 1, 0);

    // 1) 检查 run 范围内：out == residual + input0 + input1
    for t in 0..num_tokens_run {
        for j in 0..h {
            let r = residual[t * h + j] as f32;
            let a0 = input[t * (k * h) + 0 * h + j] as f32;
            let a1 = input[t * (k * h) + 1 * h + j] as f32;
            let exp = r + a0 + a1;

            let got = output[t * h + j] as f32;
            assert!(
                approx(got, exp, 5e-2),
                "run-range mismatch at t={}, j={}, got={}, exp={}",
                t, j, got, exp
            );
        }
    }

    // 2) 检查 run 范围外：output 必须保持 sentinel（绝不能被写）
    for t in num_tokens_run..num_tokens_cap {
        for j in 0..h {
            let got = output[t * h + j] as f32;
            let exp = sentinel as f32;
            assert!(
                approx(got, exp, 1e-3),
                "should-not-touch mismatch at t={}, j={}, got={}, exp(sentinel)={}",
                t, j, got, exp
            );
        }
    }
}
}







