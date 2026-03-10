use crate::common::num_traits::Sigmoid;
use crate::common::num_traits::Sqrt;
use crate::common::num_traits::{exp::Exp, neg_infinity::NegInfinity};
use crate::common::sequence_slice::SequenceSlice;
use crate::runtime::inference::state::SequenceState;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use crate::operators::routing::ExpertsSoftmaxNorm;
use crate::operators::transform::LookupRMSMap;

use crate::operators::routing::TopKSoftmax;
// Add missing imports for zip map operations
use crate::operators::linear::{Attention, MatMul, MatMul3, MatMulAdd};
// use super::mul::matmul_silu_mul_matmul::MatMulSilu;
use crate::operators::expert::{ExpertsMatMulDown, ExpertsMatMulSilu, ExpertsMergeAdd};
use crate::operators::movement::LiftVector;
use crate::operators::routing::MatMulTopK;
use crate::operators::transform::AddZipMap;
use crate::operators::transform::{AddRMSZipMap, RMSMap};
// use super::zip_map::complex_zip::ComplexZipMap;
// use super::zip_map::silu_mul_zip::SiluMulZipMap;
// use crate::common::matmul_params::MatMulParams;
// use crate::common::send_sync_ptr::{ConstPtr, MutPtr};
// use super::map::softmax_map::SoftmaxMap;
// use super::reduce::argmax_reduce::ArgmaxReduce;

#[inline]
fn thread_slices<'a>(list: &'a [Vec<SequenceSlice>], thread_id: usize) -> &'a [SequenceSlice] {
    list.get(thread_id).map(Vec::as_slice).unwrap_or(&[])
}

#[derive(Clone)]
pub enum Operator<T>
// where
//    T: PartialOrd + Copy,
{
    AddRMSZipMap(AddRMSZipMap<T>),
    AddZipMap(AddZipMap<T>),
    Attention(Attention<T>),
    // ComplexZipMap(ComplexZipMap<T>),
    ExpertsMatMulDown(ExpertsMatMulDown<T>),
    ExpertsMatMulSilu(ExpertsMatMulSilu<T>),
    ExpertsMergeAdd(ExpertsMergeAdd<T>),
    ExpertsSoftmaxNorm(ExpertsSoftmaxNorm<T>),
    LiftVector(LiftVector<T>),
    LookupRMSMap(LookupRMSMap<T>),
    MatMul(MatMul<T>),
    MatMul3(MatMul3<T>),
    MatMulAdd(MatMulAdd<T>),
    // MatMulSiluMulMatMul(MatMulSilu<T>),
    MatMulTopK(MatMulTopK<T>),
    RMSMap(RMSMap<T>),
    // SiluMulZipMap(SiluMulZipMap<T>),
    // SoftmaxMap(SoftmaxMap<T>),
    TopKSoftmax(TopKSoftmax<T>),
    // ArgmaxReduce(ArgmaxReduce<T>),
}

impl<T> Operator<T>
where
    T: PartialOrd
        + Copy
        + Default
        + Sub<Output = T>
        + Neg<Output = T>
        + Exp
        + NegInfinity
        + Sigmoid
        + Sqrt
        + AddAssign,
{
    pub fn run(
        &self,
        prefill_size: usize,
        decode_size: usize,
        cpu_num: usize,
        thread_id: usize,
        prefill_list: &[Vec<SequenceSlice>],
        decode_list: &[Vec<SequenceSlice>],
        batch_list: &mut Vec<SequenceState>,
    ) {
        let prefill_slices = thread_slices(prefill_list, thread_id);
        let decode_slices = thread_slices(decode_list, thread_id);

        macro_rules! run_simple {
            ($op:expr) => {
                $op.run(prefill_size, decode_size, cpu_num, thread_id)
            };
        }

        match self {
            Self::AddRMSZipMap(operator) => {
                run_simple!(operator);
            }
            Self::AddZipMap(operator) => {
                run_simple!(operator);
            }
            Self::Attention(operator) => {
                let attention_list = if prefill_size > 0 {
                    prefill_slices
                } else if decode_size > 0 {
                    decode_slices
                } else {
                    &[]
                };
                operator.run(
                    prefill_size,
                    decode_size,
                    attention_list,
                    cpu_num,
                    thread_id,
                );
            }

            Self::ExpertsMatMulDown(operator) => {
                run_simple!(operator);
            }

            Self::ExpertsMatMulSilu(operator) => {
                run_simple!(operator);
            }
            Self::ExpertsMergeAdd(operator) => {
                run_simple!(operator);
            }
            Self::ExpertsSoftmaxNorm(operator) => {
                run_simple!(operator);
            }
            Self::LiftVector(operator) => {
                operator.run(prefill_size, decode_size, decode_slices, cpu_num, thread_id);
            }
            Self::LookupRMSMap(operator) => {
                operator.run(
                    prefill_size,
                    decode_size,
                    cpu_num,
                    thread_id,
                    prefill_slices,
                    decode_slices,
                );
            }
            Self::MatMul(operator) => {
                run_simple!(operator);
            }

            Self::MatMul3(operator) => {
                run_simple!(operator);
            }
            Self::MatMulAdd(operator) => {
                run_simple!(operator);
            }
            /*
            Self::MatMulSiluMulMatMul(operator) => {
                operator.run(
                    position_index,
                    position_interval,
                    batch_size,
                    cpu_num,
                    thread_id,
                );
            }*/
            Self::MatMulTopK(operator) => {
                run_simple!(operator);
            }

            Self::TopKSoftmax(operator) => {
                operator.run(
                    prefill_size,
                    decode_size,
                    cpu_num,
                    thread_id,
                    decode_slices,
                    batch_list,
                );
            }
            Self::RMSMap(operator) => {
                run_simple!(operator);
            } /*
              Self::SiluMulZipMap(operator) => {
                  operator.run(prefill_size, cpu_num, thread_id);
              }
              Self::ComplexZipMap(operator) => {
                  operator.run(prefill_size, cpu_num, thread_id);
              }*/
        }
    }
}

// Many Operator variants contain raw pointers to buffers that are
// partitioned per-thread and used in a thread-safe manner by the
// runtime. Marking `Operator<T>` as `Send`/`Sync` is safe when `T`
// is a plain POD-like type (here constrained to `PartialOrd + Copy`).
// Use `unsafe impl` because raw pointers are not auto-`Send`.
unsafe impl<T> Send for Operator<T> where T: PartialOrd + Copy {}
unsafe impl<T> Sync for Operator<T> where T: PartialOrd + Copy {}

#[cfg(test)]
mod test {
    use super::*;
    use crate::common::sequence_slice::SequenceSlice;
    use crate::runtime::inference::state::{Phase, SequenceState};
    use approx::assert_ulps_eq;
    // use crate::ptensor::tensor_utils::{get_aligned_strides, get_broadcast_shape, get_strides};
    // use std::sync::{Arc, Barrier};
    // use std::thread;

    #[test]
    fn test_experts_softmax_norm() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            println!("AVX2 not supported, skipping test.");
            return;
        }

        let sequence_chunk_size = 1;
        let batch_size = 2;
        let num_experts = 16;
        let num_topk = 4;
        let num_tokens = sequence_chunk_size * batch_size;
        let prefill_size = num_tokens;
        let decode_size = sequence_chunk_size;

        let input_data1: Vec<f32> = vec![
            0.5, -1.0, 2.5, 3.0, 7.5, 6.5, -2.0, 10.0, 4.0, 8.0, 1.0, 9.5, -3.5, 5.5, 11.0, -0.25,
        ];
        let input_data2: Vec<f32> = vec![
            -0.5, 0.25, 3.75, -2.0, 6.0, 1.75, -4.25, 2.5, 0.0, 5.25, -1.25, 4.0, 3.0, -3.5, 7.5,
            2.25,
        ];
        let mut input_data = Vec::new();
        input_data.extend_from_slice(&input_data1);
        input_data.extend_from_slice(&input_data2);

        let mut experts_indicator = vec![false; num_experts];
        let mut indice_ptr = vec![false; num_experts * num_tokens];
        let mut weight_ptr = vec![0.0f32; num_experts * num_tokens];
        let mut topk_indices_ptr = vec![0usize; num_topk * num_tokens];

        let operator = Operator::ExpertsSoftmaxNorm(ExpertsSoftmaxNorm::<f32>::new(
            input_data.as_ptr(),
            experts_indicator.as_mut_ptr(),
            indice_ptr.as_mut_ptr(),
            weight_ptr.as_mut_ptr(),
            topk_indices_ptr.as_mut_ptr(),
            batch_size,
            num_experts,
            num_topk,
            false,
        ));

        let thread_num = 1;
        let thread_id = 0;
        operator.run(
            prefill_size,
            decode_size,
            thread_num,
            thread_id,
            &[],
            &[],
            &mut Vec::new(),
        );

        // Verification for token 0
        let mut expected1: Vec<(usize, f32)> = input_data1.iter().copied().enumerate().collect();
        expected1.sort_by(|a, b| b.1.total_cmp(&a.1));
        let max_val1 = input_data1
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let denom1: f32 = input_data1.iter().map(|v| (v - max_val1).exp()).sum();

        for i in 0..num_topk {
            let (idx, val) = expected1[i];
            let prob = (val - max_val1).exp() / denom1;
            assert!(experts_indicator[idx]);
            let offset = idx * num_tokens + 0;
            assert!(indice_ptr[offset]);
            assert_ulps_eq!(weight_ptr[offset], prob, max_ulps = 4);
        }

        // Verification for token 1
        let mut expected2: Vec<(usize, f32)> = input_data2.iter().copied().enumerate().collect();
        expected2.sort_by(|a, b| b.1.total_cmp(&a.1));
        let max_val2 = input_data2
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let denom2: f32 = input_data2.iter().map(|v| (v - max_val2).exp()).sum();

        for i in 0..num_topk {
            let (idx, val) = expected2[i];
            let prob = (val - max_val2).exp() / denom2;
            assert!(experts_indicator[idx]);
            let offset = idx * num_tokens + 1;
            assert!(indice_ptr[offset]);
            assert_ulps_eq!(weight_ptr[offset], prob, max_ulps = 4);
        }
    }

    #[test]
    fn test_topk_softmax() {
        let batch_size = 2;
        let topk_size = 8;
        let thread_num = 4;
        let prefill_size = batch_size;
        let decode_size = 1;

        let total_candidates_per_item = topk_size * thread_num;
        let input_len = batch_size * total_candidates_per_item;

        let mut input_values = Vec::<f32>::with_capacity(input_len);
        let mut input_indices = Vec::<usize>::with_capacity(input_len);

        for i in 0..batch_size {
            for j in 0..total_candidates_per_item {
                input_values.push(5.0 - (j as f32 * 0.1) - (i as f32));
                input_indices.push(i * 1000 + j);
            }
        }

        let mut output_values = vec![0.0f32; batch_size * topk_size];
        let mut output_indices = vec![0usize; batch_size * topk_size];
        let mut output_sequences = vec![0usize; batch_size];
        let eos_id = 0usize;

        let batch_records: Vec<SequenceState> = (0..batch_size)
            .map(|_| SequenceState {
                sequence_index: 0,
                kv_index: 0,
                phase: Phase::Decode,
                // prompt_length: 0,
                notify: std::sync::Arc::new(tokio::sync::Notify::new()),
            })
            .collect();
        let mut batch_list = batch_records;

        let tokens_per_thread = (batch_size + thread_num - 1) / thread_num;
        let mut decode_lists = Vec::with_capacity(thread_num);
        for tid in 0..thread_num {
            let start = tid * tokens_per_thread;
            let end = (start + tokens_per_thread).min(batch_size);
            let mut slices = Vec::with_capacity(end.saturating_sub(start));
            for batch_index in start..end {
                slices.push(SequenceSlice {
                    batch_index,
                    sequence_index: 0,
                    token_start_index: batch_index,
                    lift_index: 0,
                    length: 1,
                });
            }
            decode_lists.push(slices);
        }

        let operator = Operator::TopKSoftmax(TopKSoftmax::<f32>::new(
            input_indices.as_ptr(),
            input_values.as_ptr(),
            // sums.as_ptr(),
            output_indices.as_mut_ptr(),
            output_values.as_mut_ptr(),
            output_sequences.as_mut_ptr(),
            batch_size,
            topk_size,
            eos_id,
        ));

        for i in 0..thread_num {
            if let Operator::TopKSoftmax(inner) = &operator {
                inner.run(
                    prefill_size,
                    decode_size,
                    thread_num,
                    i,
                    &decode_lists[i],
                    &mut batch_list,
                );
            }
        }

        for i in 0..batch_size {
            let item_input_values =
                &input_values[i * total_candidates_per_item..(i + 1) * total_candidates_per_item];
            let item_input_indices =
                &input_indices[i * total_candidates_per_item..(i + 1) * total_candidates_per_item];

            let mut paired: Vec<_> = item_input_values
                .iter()
                .copied()
                .zip(item_input_indices.iter().copied())
                .collect();
            paired.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            let topk = &paired[..topk_size];
            let max_val = topk[0].0;
            let denom: f32 = topk.iter().map(|(v, _)| (v - max_val).exp()).sum();

            let expected_probs: Vec<f32> = topk
                .iter()
                .map(|(v, _)| (v - max_val).exp() / denom)
                .collect();
            let expected_indices: Vec<usize> = topk.iter().map(|(_, idx)| *idx).collect();

            let output_vals_slice = &output_values[i * topk_size..(i + 1) * topk_size];
            let output_idx_slice = &output_indices[i * topk_size..(i + 1) * topk_size];

            assert_ulps_eq!(output_vals_slice, expected_probs.as_slice(), max_ulps = 4);
            assert_eq!(output_idx_slice, expected_indices.as_slice());
            assert_eq!(output_sequences[i], expected_indices[0]);
        }
    }

    #[test]
    fn test_operator_matmul_f16_dispatch_and_parallel_consistency() {
        use approx::assert_abs_diff_eq;
        use std::f16;

        // MR=3 => M % 3 == 0
        // NR=32 => N % 32 == 0
        // KC=64 => K % 64 == 0
        const M: usize = 6;
        const K: usize = 64;
        const N: usize = 32;

        // A[M×K], B_nt[N×K], C[M×N]
        let mut a = vec![0.0f16; M * K];
        let mut b_nt = vec![0.0f16; N * K];

        for i in 0..M {
            for kk in 0..K {
                let v = 0.01f32 * (i as f32) + 0.001f32 * (kk as f32);
                a[i * K + kk] = v as f16;
            }
        }

        // 注意：现在 RHS 是 NT：[N×K] row-major（每行 K）
        // 让 b_nt[j, kk] = 0.02*kk + 0.003*j
        for j in 0..N {
            for kk in 0..K {
                let v = 0.02f32 * (kk as f32) + 0.003f32 * (j as f32);
                b_nt[j * K + kk] = v as f16;
            }
        }

        let params = crate::common::matmul_params::MatMulParams {
            a_row_step_macro: M,  // MB
            b_row_step_macro: N,  // NB
            column_step_macro: K, // KC
            a_row_step_micro: 3,  // MR
            b_row_step_micro: 32, // NR
        };

        // cpu_num = 1
        let mut c1 = vec![0.0f16; M * N];
        let matmul1 = unsafe {
            crate::operators::linear::MatMul::<f16>::new(
                a.as_ptr(),
                b_nt.as_ptr(), // ✅ 传 NT
                c1.as_mut_ptr(),
                false,
                params,
                M,
                N,
                K,
                false,
            )
        };
        let op1 = super::Operator::MatMul(matmul1.clone());

        let batch_size = M;
        let decode_size = 1;

        op1.run(batch_size, decode_size, 1, 0, &[], &[], &mut Vec::new());

        // cpu_num = thread_num
        let mut c2 = vec![0.0f16; M * N];
        let matmul2 = unsafe {
            crate::operators::linear::MatMul::<f16>::new(
                a.as_ptr(),
                b_nt.as_ptr(), // ✅ 传 NT
                c2.as_mut_ptr(),
                false,
                params,
                M,
                N,
                K,
                false,
            )
        };
        let op2 = super::Operator::MatMul(matmul2.clone());

        let max_threads = matmul2.panel_threads().max(1);
        let thread_num = num_cpus::get().min(max_threads).min(16);

        for tid in 0..thread_num {
            op2.run(
                batch_size,
                decode_size,
                thread_num,
                tid,
                &[],
                &[],
                &mut Vec::new(),
            );
        }

        // 1) 并行一致性
        for idx in 0..(M * N) {
            let x = c1[idx] as f32;
            let y = c2[idx] as f32;
            assert_abs_diff_eq!(x, y, epsilon = 1e-1);
        }

        // 2) 正确性：reference 用 NT 读法
        for i in 0..M {
            for j in 0..N {
                let mut sum = 0.0f32;
                for kk in 0..K {
                    sum += (a[i * K + kk] as f32) * (b_nt[j * K + kk] as f32);
                }
                let got = c2[i * N + j] as f32;
                assert_abs_diff_eq!(got, sum, epsilon = 1e-1);
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn assert_close_with_msg(got: f32, exp: f32, eps: f32, row: usize, rank: usize) {
        let diff = (got - exp).abs();
        if diff > eps {
            panic!(
                "val mismatch row {}, rank {}: got {}, exp {}, diff {}, eps {}",
                row, rank, got, exp, diff, eps
            );
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_operator_matmul_topk_f16_avx512_dispatch_small_no_ties() {
        use std::f16;

        const M: usize = 12;
        const K: usize = 64;
        const N: usize = 128;
        const TOPK: usize = 10;

        const MB: usize = 6;
        const NB: usize = 128;
        const KC: usize = 64;
        const MR: usize = 3;
        const NR: usize = 32;

        let mut a = vec![0.0f16; M * K];
        let mut b_nt = vec![0.0f16; N * K];

        for x in &mut a {
            *x = 1.0f32 as f16;
        }

        // 现在 RHS 是 NT：[N×K] row-major
        // b_nt[j,kk] = base(kk) + col_bias(j)
        for j in 0..N {
            let col_bias = (j as f32) * 1e-3;
            for kk in 0..K {
                let base = (kk as f32) * 1e-6;
                b_nt[j * K + kk] = (base + col_bias) as f16;
            }
        }

        let expected_indices: Vec<usize> = (0..TOPK).map(|r| N - 1 - r).collect();
        let sum_base: f32 = (0..K).map(|kk| (kk as f32) * 1e-6).sum();

        unsafe {
            let thread_max = crate::operators::routing::MatMulTopK::<f16>::detect_threads();
            let buf_len = M * thread_max * TOPK;
            let mut indices_buf = vec![0usize; buf_len];
            let mut values_buf = vec![0.0f16; buf_len];

            let runner = crate::operators::routing::MatMulTopK::<f16>::new(
                a.as_ptr(),
                b_nt.as_ptr(), // ✅ 传 NT
                indices_buf.as_mut_ptr(),
                values_buf.as_mut_ptr(),
                M,
                N,
                K,
                MB,
                NB,
                KC,
                MR,
                NR,
                M,
                TOPK,
            );

            let used_cpu = num_cpus::get().min(runner.thread_max()).min(8).max(1);
            let op = Operator::MatMulTopK(runner);

            for tid in 0..used_cpu {
                op.run(M, 1, used_cpu, tid, &[], &[], &mut Vec::new());
            }

            for row in 0..M {
                let mut merged: Vec<(usize, f32)> = Vec::with_capacity(used_cpu * TOPK);
                for tid in 0..used_cpu {
                    let off = row * (thread_max * TOPK) + tid * TOPK;
                    for r in 0..TOPK {
                        merged.push((indices_buf[off + r], values_buf[off + r] as f32));
                    }
                }
                merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                merged.truncate(TOPK);

                for r in 0..TOPK {
                    let got_idx = merged[r].0;
                    if got_idx != expected_indices[r] {
                        panic!(
                            "idx mismatch row {}, rank {}: got {}, exp {}",
                            row, r, got_idx, expected_indices[r]
                        );
                    }
                }

                for r in 0..TOPK {
                    let j = expected_indices[r];
                    let expected_val = sum_base + (K as f32) * ((j as f32) * 1e-3);
                    let got_val = merged[r].1;
                    assert_close_with_msg(got_val, expected_val, 0.25, row, r);
                }
            }
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_operator_matmul_topk_f16_avx512_dispatch_kc_split_multi_tile_no_ties() {
        use std::f16;

        const M: usize = 24;
        const K: usize = 128;
        const N: usize = 256;
        const TOPK: usize = 10;

        const MB: usize = 12;
        const NB: usize = 128;
        const KC: usize = 64;
        const MR: usize = 3;
        const NR: usize = 32;

        let mut a = vec![0.0f16; M * K];
        let mut b_nt = vec![0.0f16; N * K];

        for x in &mut a {
            *x = 1.0f32 as f16;
        }

        // NT：[N×K]
        for j in 0..N {
            let col_bias = (j as f32) * 5e-4;
            for kk in 0..K {
                let base = (kk as f32) * 5e-7;
                b_nt[j * K + kk] = (base + col_bias) as f16;
            }
        }

        let expected_indices: Vec<usize> = (0..TOPK).map(|r| N - 1 - r).collect();
        let sum_base: f32 = (0..K).map(|kk| (kk as f32) * 5e-7).sum();

        unsafe {
            let thread_max = crate::operators::routing::MatMulTopK::<f16>::detect_threads();
            let buf_len = M * thread_max * TOPK;
            let mut indices_buf = vec![0usize; buf_len];
            let mut values_buf = vec![0.0f16; buf_len];

            let runner = crate::operators::routing::MatMulTopK::<f16>::new(
                a.as_ptr(),
                b_nt.as_ptr(), // ✅ 传 NT
                indices_buf.as_mut_ptr(),
                values_buf.as_mut_ptr(),
                M,
                N,
                K,
                MB,
                NB,
                KC,
                MR,
                NR,
                M,
                TOPK,
            );

            let used_cpu = num_cpus::get().min(runner.thread_max()).min(16).max(1);
            let op = Operator::MatMulTopK(runner);

            for tid in 0..used_cpu {
                op.run(M, 1, used_cpu, tid, &[], &[], &mut Vec::new());
            }

            for row in 0..M {
                let mut merged: Vec<(usize, f32)> = Vec::with_capacity(used_cpu * TOPK);
                for tid in 0..used_cpu {
                    let off = row * (thread_max * TOPK) + tid * TOPK;
                    for r in 0..TOPK {
                        merged.push((indices_buf[off + r], values_buf[off + r] as f32));
                    }
                }
                merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                merged.truncate(TOPK);

                for r in 0..TOPK {
                    let got_idx = merged[r].0;
                    if got_idx != expected_indices[r] {
                        panic!(
                            "idx mismatch row {}, rank {}: got {}, exp {}",
                            row, r, got_idx, expected_indices[r]
                        );
                    }
                }

                for r in 0..TOPK {
                    let j = expected_indices[r];
                    let expected_val = sum_base + (K as f32) * ((j as f32) * 5e-4);
                    let got_val = merged[r].1;
                    assert_close_with_msg(got_val, expected_val, 0.45, row, r);
                }
            }
        }
    }

    #[inline]
    fn silu_f32(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// reference (NT weights):
    /// out[e,b,i] = silu(sum_k a[b,k]*w_gate_nt[e,i,k]) * (sum_k a[b,k]*w_up_nt[e,i,k])
    fn ref_experts_silu_f32(
        a: &[f16],                  // [B,H]
        w_gate_nt: &[f16],          // [E,I,H] row-major (NT)
        w_up_nt: &[f16],            // [E,I,H] row-major (NT)
        experts_indicator: &[bool], // [E]
        indice: &[bool],            // [E,B] expert-major
        out: &mut [f32],            // [E,B,I]
        b: usize,
        h: usize,
        i: usize,
        e: usize,
    ) {
        for v in out.iter_mut() {
            *v = 0.0;
        }

        for ex in 0..e {
            if !experts_indicator[ex] {
                continue;
            }
            for bb in 0..b {
                if !indice[ex * b + bb] {
                    continue;
                }
                for ii in 0..i {
                    let mut g = 0.0f32;
                    let mut u = 0.0f32;

                    // w_nt index: ex*(I*H) + ii*H + kk
                    let wg_base = ex * (i * h) + ii * h;
                    let wu_base = ex * (i * h) + ii * h;

                    for kk in 0..h {
                        let a_v = a[bb * h + kk] as f32;
                        let wg = w_gate_nt[wg_base + kk] as f32;
                        let wu = w_up_nt[wu_base + kk] as f32;
                        g += a_v * wg;
                        u += a_v * wu;
                    }
                    out[ex * (b * i) + bb * i + ii] = silu_f32(g) * u;
                }
            }
        }
    }

    fn run_operator_all_threads(op: &Operator<f16>, batch: usize, cpu_num: usize) {
        for tid in 0..cpu_num {
            op.run(batch, 1, cpu_num, tid, &[], &[], &mut Vec::new());
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_operator_experts_matmul_silu_f16_dispatch_single_expert_basic() {
        use std::f16;

        const B: usize = 6;
        const H: usize = 64;
        const I: usize = 64;
        const E: usize = 1;

        let mb = 3;
        let nb = 32;
        let kc = 64;
        let mr = 3;
        let nr = 32;

        let cpu_num = 2usize.min(num_cpus::get()).max(1);

        let mut a = vec![0.0f16; B * H];

        // ✅ 现在权重改成 NT：[E,I,H]
        let mut w_gate_nt = vec![0.0f16; E * I * H];
        let mut w_up_nt = vec![0.0f16; E * I * H];

        let mut out = vec![0.0f16; E * B * I];

        let experts_indicator = vec![true; E];
        let mut indice = vec![false; E * B];
        for bb in 0..B {
            indice[0 * B + bb] = true;
        }

        for bb in 0..B {
            for kk in 0..H {
                a[bb * H + kk] = ((bb as f32) * 0.01 + (kk as f32) * 0.001) as f16;
            }
        }

        // 填 NT：w[ex, ii, kk]
        for kk in 0..H {
            for ii in 0..I {
                let wg = (kk as f32) * 0.002 + (ii as f32) * 0.0003;
                let wu = (kk as f32) * 0.0017 + (ii as f32) * 0.0002;
                w_gate_nt[0 * (I * H) + ii * H + kk] = wg as f16;
                w_up_nt[0 * (I * H) + ii * H + kk] = wu as f16;
            }
        }

        unsafe {
            let runner = crate::operators::expert::ExpertsMatMulSilu::<f16>::new(
                a.as_ptr(),
                w_gate_nt.as_ptr(), // ✅ 传 NT
                w_up_nt.as_ptr(),   // ✅ 传 NT
                experts_indicator.as_ptr(),
                indice.as_ptr(),
                out.as_mut_ptr(),
                B,
                I,
                H,
                E,
                mb,
                nb,
                kc,
                mr,
                nr,
                false,
            );

            let op = Operator::ExpertsMatMulSilu(runner);
            run_operator_all_threads(&op, B, cpu_num);
        }

        let mut ref_out = vec![0.0f32; E * B * I];
        ref_experts_silu_f32(
            &a,
            &w_gate_nt,
            &w_up_nt,
            &experts_indicator,
            &indice,
            &mut ref_out,
            B,
            H,
            I,
            E,
        );

        let eps = 7e-1;
        for idx in 0..(E * B * I) {
            let got = out[idx] as f32;
            let exp = ref_out[idx];
            let diff = (got - exp).abs();
            if diff > eps {
                let e = idx / (B * I);
                let rem = idx % (B * I);
                let b = rem / I;
                let ii = rem % I;
                panic!(
                    "mismatch e={}, b={}, i={} : got={}, exp={}, diff={}, eps={}",
                    e, b, ii, got, exp, diff, eps
                );
            }
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_operator_experts_matmul_silu_f16_dispatch_multi_expert_sparse_routing_kc_split() {
        use std::f16;

        const B: usize = 12;
        const H: usize = 128;
        const I: usize = 96;
        const E: usize = 3;

        let mb = 6;
        let nb = 64;
        let kc = 64;
        let mr = 3;
        let nr = 32;

        let cpu_num = 4usize.min(num_cpus::get()).max(1);

        let mut a = vec![0.0f16; B * H];

        // ✅ NT 权重：[E,I,H]
        let mut w_gate_nt = vec![0.0f16; E * I * H];
        let mut w_up_nt = vec![0.0f16; E * I * H];

        let mut out = vec![0.0f16; E * B * I];

        let experts_indicator = vec![true, false, true];

        let mut indice = vec![false; E * B];
        for bb in (0..B).step_by(2) {
            indice[0 * B + bb] = true;
        }
        for bb in 3..11 {
            indice[2 * B + bb] = true;
        }

        for bb in 0..B {
            for kk in 0..H {
                a[bb * H + kk] = (((bb * 7 + kk * 3) % 31) as f32 * 0.01) as f16;
            }
        }

        // 填 NT：w[ex, ii, kk]
        for ex in 0..E {
            for kk in 0..H {
                for ii in 0..I {
                    let wg = (((ex * 13 + kk * 5 + ii * 7) % 29) as f32 * 0.01) as f32;
                    let wu = (((ex * 11 + kk * 3 + ii * 9) % 37) as f32 * 0.01) as f32;
                    w_gate_nt[ex * (I * H) + ii * H + kk] = wg as f16;
                    w_up_nt[ex * (I * H) + ii * H + kk] = wu as f16;
                }
            }
        }

        unsafe {
            let runner = crate::operators::expert::ExpertsMatMulSilu::<f16>::new(
                a.as_ptr(),
                w_gate_nt.as_ptr(), // ✅ NT
                w_up_nt.as_ptr(),   // ✅ NT
                experts_indicator.as_ptr(),
                indice.as_ptr(),
                out.as_mut_ptr(),
                B,
                I,
                H,
                E,
                mb,
                nb,
                kc,
                mr,
                nr,
                false,
            );

            let op = Operator::ExpertsMatMulSilu(runner);
            run_operator_all_threads(&op, B, cpu_num);
        }

        let mut ref_out = vec![0.0f32; E * B * I];
        ref_experts_silu_f32(
            &a,
            &w_gate_nt,
            &w_up_nt,
            &experts_indicator,
            &indice,
            &mut ref_out,
            B,
            H,
            I,
            E,
        );

        let eps_active = 9e-1;
        let eps_inactive = 1e-3;

        for ex in 0..E {
            for bb in 0..B {
                let active = experts_indicator[ex] && indice[ex * B + bb];
                for ii in 0..I {
                    let idx = ex * (B * I) + bb * I + ii;
                    let got = out[idx] as f32;
                    let exp = ref_out[idx];

                    if active {
                        let diff = (got - exp).abs();
                        if diff > eps_active {
                            panic!(
                            "ACTIVE mismatch ex={}, b={}, i={} : got={}, exp={}, diff={}, eps={}",
                            ex, bb, ii, got, exp, diff, eps_active
                        );
                        }
                    } else {
                        let diff = got.abs();
                        if diff > eps_inactive {
                            panic!(
                                "INACTIVE should be 0 ex={}, b={}, i={} : got={}, diff={}",
                                ex, bb, ii, got, diff
                            );
                        }
                    }
                }
            }
        }
    }

    use std::mem;

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
    fn slot_of(topk: &[usize], b: usize, ktop: usize, e: usize) -> usize {
        let row = &topk[b * ktop..b * ktop + ktop];
        row.iter().position(|&x| x == e).unwrap_or(0)
    }

    /// ref_down with NT wdown:
    /// out_ref[b,slot,j] += weight[e,b] * sum_kk nonlin[e,b,kk] * wdown_nt[e,j,kk]
    fn ref_down_f32(
        nonlin: &[f16],   // [E,B,Hmid]
        wdown_nt: &[f16], // [E,H,Hmid] row-major (NT)
        experts_indicator: &[bool],
        indice: &[bool],     // [E,B]
        weight: &[f16],      // [E,B]
        topk: &[usize],      // [B,Ktop]
        out_ref: &mut [f32], // [B,Ktop,H]
        e: usize,
        b: usize,
        hmid: usize,
        h: usize,
        ktop: usize,
    ) {
        for v in out_ref.iter_mut() {
            *v = 0.0;
        }

        for ex in 0..e {
            if !experts_indicator[ex] {
                continue;
            }
            for bb in 0..b {
                if !indice[ex * b + bb] {
                    continue;
                }
                let s = slot_of(topk, bb, ktop, ex);
                let w = f32_from_f16(weight[ex * b + bb]);

                for j in 0..h {
                    let mut acc = 0.0f32;

                    // wdown_nt index base: ex*(H*Hmid) + j*Hmid
                    let wj_base = ex * (h * hmid) + j * hmid;

                    for kk in 0..hmid {
                        let a = f32_from_f16(nonlin[(ex * b + bb) * hmid + kk]);
                        let bj = f32_from_f16(wdown_nt[wj_base + kk]);
                        acc += a * bj;
                    }
                    out_ref[(bb * ktop + s) * h + j] += w * acc;
                }
            }
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_operator_experts_down_f16_dispatch_basic_mb_gt_mr() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }
        use std::f16;

        let e = 2usize;
        let b = 6usize;
        let hmid = 64usize;
        let h = 64usize;
        let ktop = 2usize;

        let params = crate::common::matmul_params::MatMulParams {
            a_row_step_macro: 3,
            b_row_step_macro: 32,
            column_step_macro: 32,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let mut nonlin = vec![0.0f16; e * b * hmid];

        // ✅ NT wdown：[E,H,Hmid]
        let mut wdown_nt = vec![0.0f16; e * h * hmid];

        let mut out = vec![0.0f16; b * ktop * h];

        let experts_indicator = vec![true, true];
        let mut indice = vec![false; e * b];
        let mut weight = vec![0.0f16; e * b];
        let mut topk = vec![0usize; b * ktop];

        for bb in 0..b {
            topk[bb * ktop + 0] = 0;
            topk[bb * ktop + 1] = 1;
        }

        for bb in 0..b {
            indice[0 * b + bb] = true;
            if bb % 2 == 1 {
                indice[1 * b + bb] = true;
            }
        }

        for ex in 0..e {
            for bb in 0..b {
                weight[ex * b + bb] =
                    (0.5f32 + 0.01f32 * (ex as f32) + 0.02f32 * (bb as f32)) as f16;
            }
        }

        for ex in 0..e {
            for bb in 0..b {
                for kk in 0..hmid {
                    nonlin[(ex * b + bb) * hmid + kk] =
                        (0.001f32 * (ex as f32) + 0.01f32 * (bb as f32) + 0.0003f32 * (kk as f32))
                            as f16;
                }
            }
        }

        // 填 NT：wdown_nt[ex, j, kk]
        for ex in 0..e {
            for kk in 0..hmid {
                for j in 0..h {
                    let v = (0.0007f32 * (ex as f32)
                        + 0.0009f32 * (kk as f32)
                        + 0.0002f32 * (j as f32)) as f16;
                    wdown_nt[ex * (h * hmid) + j * hmid + kk] = v;
                }
            }
        }

        unsafe {
            let runner = crate::operators::expert::ExpertsMatMulDown::<f16>::new(
                nonlin.as_ptr(),
                wdown_nt.as_ptr(), // ✅ NT
                experts_indicator.as_ptr(),
                indice.as_ptr(),
                weight.as_ptr(),
                topk.as_ptr(),
                out.as_mut_ptr(),
                e,
                b,
                hmid,
                h,
                ktop,
                params,
            );

            let op = Operator::ExpertsMatMulDown(runner);
            run_operator_all_threads(&op, b, 1);
        }

        let mut out_ref = vec![0.0f32; b * ktop * h];
        ref_down_f32(
            &nonlin,
            &wdown_nt,
            &experts_indicator,
            &indice,
            &weight,
            &topk,
            &mut out_ref,
            e,
            b,
            hmid,
            h,
            ktop,
        );

        let eps = 8e-1f32;
        for i in 0..out.len() {
            let got = f32_from_f16(out[i]);
            let exp = out_ref[i];
            let diff = (got - exp).abs();
            if diff > eps {
                panic!(
                    "down basic mismatch at {}: got={}, exp={}, diff={}, eps={}",
                    i, got, exp, diff, eps
                );
            }
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_operator_experts_down_f16_dispatch_tail_cols() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }
        use std::f16;

        let e = 1usize;
        let b = 3usize;
        let hmid = 32usize;
        let h = 48usize;
        let ktop = 1usize;

        let params = crate::common::matmul_params::MatMulParams {
            a_row_step_macro: 3,
            b_row_step_macro: 48,
            column_step_macro: 16,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let mut nonlin = vec![0.0f16; e * b * hmid];

        // ✅ NT wdown：[E,H,Hmid]
        let mut wdown_nt = vec![0.0f16; e * h * hmid];

        let mut out = vec![0.0f16; b * ktop * h];

        let experts_indicator = vec![true];
        let indice = vec![true; e * b];
        let mut weight = vec![0.0f16; e * b];
        let mut topk = vec![0usize; b * ktop];

        for bb in 0..b {
            weight[bb] = (0.7f32 + 0.02f32 * (bb as f32)) as f16;
            topk[bb] = 0;
        }

        for bb in 0..b {
            for kk in 0..hmid {
                nonlin[bb * hmid + kk] = (0.01f32 * (bb as f32) + 0.0003f32 * (kk as f32)) as f16;
            }
        }

        // 填 NT：wdown_nt[0, j, kk]
        for kk in 0..hmid {
            for j in 0..h {
                let v = (0.001f32 * (kk as f32) + 0.0009f32 * (j as f32)) as f16;
                wdown_nt[0 * (h * hmid) + j * hmid + kk] = v;
            }
        }

        unsafe {
            let runner = crate::operators::expert::ExpertsMatMulDown::<f16>::new(
                nonlin.as_ptr(),
                wdown_nt.as_ptr(), // ✅ NT
                experts_indicator.as_ptr(),
                indice.as_ptr(),
                weight.as_ptr(),
                topk.as_ptr(),
                out.as_mut_ptr(),
                e,
                b,
                hmid,
                h,
                ktop,
                params,
            );

            let op = Operator::ExpertsMatMulDown(runner);
            run_operator_all_threads(&op, b, 1);
        }

        let mut out_ref = vec![0.0f32; b * ktop * h];
        ref_down_f32(
            &nonlin,
            &wdown_nt,
            &experts_indicator,
            &indice,
            &weight,
            &topk,
            &mut out_ref,
            e,
            b,
            hmid,
            h,
            ktop,
        );

        let eps = 8e-1f32;
        for i in 0..out.len() {
            let got = f32_from_f16(out[i]);
            let exp = out_ref[i];
            let diff = (got - exp).abs();
            if diff > eps {
                panic!(
                    "down tail mismatch at {}: got={}, exp={}, diff={}, eps={}",
                    i, got, exp, diff, eps
                );
            }
        }
    }

    #[inline]
    fn assert_close(got: f32, exp: f32, tol: f32, msg: &str) {
        let diff = (got - exp).abs();
        if diff > tol {
            panic!(
                "{} got={}, exp={}, diff={}, tol={}",
                msg, got, exp, diff, tol
            );
        }
    }

    fn ref_merge_add_f32(
        input: &[f16],       // [T,K,H]
        residual: &[f16],    // [T,H]
        out_ref: &mut [f32], // [T,H]
        num_tokens: usize,
        k: usize,
        h: usize,
    ) {
        for t in 0..num_tokens {
            for hh in 0..h {
                let mut v = f32_from_f16(residual[t * h + hh]);
                let base = t * (k * h);
                for s in 0..k {
                    v += f32_from_f16(input[base + s * h + hh]);
                }
                out_ref[t * h + hh] = v;
            }
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_operator_merge_add_f16_dispatch_k1_basic() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }
        use std::f16;

        let seq = 2usize;
        let batch = 3usize;
        let num_tokens = seq * batch;

        let k = 1usize;
        let h = 64usize;
        let num_experts = 4usize;

        let mut input = vec![0.0f16; num_tokens * k * h];
        let mut residual = vec![0.0f16; num_tokens * h];
        let mut out = vec![0.0f16; num_tokens * h];

        let mut experts_indicator = vec![false; num_experts];
        let mut indice_ptr = vec![false; num_experts * num_tokens];

        for t in 0..num_tokens {
            for hh in 0..h {
                let r_val = 0.1f32 * (t as f32) + 0.001f32 * (hh as f32);
                let i_val = 0.05f32 * (t as f32) + 0.0007f32 * (hh as f32);
                residual[t * h + hh] = r_val as f16;
                input[(t * k + 0) * h + hh] = i_val as f16;
            }
        }

        unsafe {
            let runner = crate::operators::expert::ExpertsMergeAdd::<f16>::new(
                input.as_ptr(),
                residual.as_ptr(),
                experts_indicator.as_mut_ptr(),
                indice_ptr.as_mut_ptr(),
                out.as_mut_ptr(),
                seq,
                batch,
                num_experts,
                k,
                h,
                false,
                false,
            );

            let op = Operator::ExpertsMergeAdd(runner);
            run_operator_all_threads(&op, batch, 1);
        }

        let mut out_ref = vec![0.0f32; num_tokens * h];
        ref_merge_add_f32(&input, &residual, &mut out_ref, num_tokens, k, h);

        let tol = 5e-2f32;
        for i in 0..out.len() {
            let got = f32_from_f16(out[i]);
            let exp = out_ref[i];
            assert_close(got, exp, tol, "k1_basic");
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_operator_merge_add_f16_dispatch_k3_and_tail_h48() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }
        use std::f16;

        let num_tokens = 5usize;
        let seq = 1usize;
        let batch = num_tokens;

        let k = 3usize;
        let h = 48usize;
        let num_experts = 8usize;

        let mut input = vec![0.0f16; num_tokens * k * h];
        let mut residual = vec![0.0f16; num_tokens * h];
        let mut out = vec![0.0f16; num_tokens * h];

        let mut experts_indicator = vec![false; num_experts];
        let mut indice_ptr = vec![false; num_experts * num_tokens];

        for t in 0..num_tokens {
            for hh in 0..h {
                let r_val = 0.03f32 * (t as f32) + 0.0009f32 * (hh as f32);
                residual[t * h + hh] = r_val as f16;

                for s in 0..k {
                    let val = 0.01f32 * ((s as f32) + 1.0)
                        + 0.002f32 * (t as f32)
                        + 0.0002f32 * (hh as f32);
                    input[t * (k * h) + s * h + hh] = val as f16;
                }
            }
        }

        unsafe {
            let runner = crate::operators::expert::ExpertsMergeAdd::<f16>::new(
                input.as_ptr(),
                residual.as_ptr(),
                experts_indicator.as_mut_ptr(),
                indice_ptr.as_mut_ptr(),
                out.as_mut_ptr(),
                seq,
                batch,
                num_experts,
                k,
                h,
                false,
                false,
            );

            let op = Operator::ExpertsMergeAdd(runner);
            run_operator_all_threads(&op, batch, 1);
        }

        let mut out_ref = vec![0.0f32; num_tokens * h];
        ref_merge_add_f32(&input, &residual, &mut out_ref, num_tokens, k, h);

        let tol = 5e-2f32;
        for i in 0..out.len() {
            let got = f32_from_f16(out[i]);
            let exp = out_ref[i];
            assert_close(got, exp, tol, "k3_tail_h48");
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_operator_merge_add_f16_dispatch_reset_gating_multithread() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }
        use std::f16;

        let sequence_chunk_size = 16usize;
        let batch_size = 4usize;
        let num_tokens = sequence_chunk_size * batch_size;

        let k = 2usize;
        let h = 64usize;
        let num_experts = 8usize;

        let num_threads = 4usize;

        let mut input = vec![0.0f16; num_tokens * k * h];
        let mut residual = vec![0.0f16; num_tokens * h];
        let mut output = vec![0.0f16; num_tokens * h];

        let mut out_ref = vec![0.0f32; num_tokens * h];

        for t in 0..num_tokens {
            for hh in 0..h {
                let r_val = (((t + hh) % 100) as f32) * 0.01 - 0.5;
                residual[t * h + hh] = r_val as f16;

                for s in 0..k {
                    let v = (((t * k + s + hh) % 100) as f32) * 0.01 - 0.5;
                    input[t * (k * h) + s * h + hh] = v as f16;
                }
            }
        }
        ref_merge_add_f32(&input, &residual, &mut out_ref, num_tokens, k, h);

        let mut experts_indicator = vec![true; num_experts];
        let mut indice_ptr = vec![true; num_experts * num_tokens];

        unsafe {
            let runner = crate::operators::expert::ExpertsMergeAdd::<f16>::new(
                input.as_ptr(),
                residual.as_ptr(),
                experts_indicator.as_mut_ptr(),
                indice_ptr.as_mut_ptr(),
                output.as_mut_ptr(),
                sequence_chunk_size,
                batch_size,
                num_experts,
                k,
                h,
                true,
                false,
            );

            let op = Operator::ExpertsMergeAdd(runner);

            for tid in 0..num_threads {
                op.run(batch_size, 0, num_threads, tid, &[], &[], &mut Vec::new());
            }
        }

        let tol = 5e-2f32;
        for i in 0..output.len() {
            let got = f32_from_f16(output[i]);
            let exp = out_ref[i];
            assert_close(got, exp, tol, "multithreaded_merge");
        }

        for (i, &v) in experts_indicator.iter().enumerate() {
            if v {
                panic!("experts_indicator[{}] not reset to false", i);
            }
        }
        for (i, &v) in indice_ptr.iter().enumerate() {
            if v {
                panic!("indice_ptr[{}] not reset to false", i);
            }
        }
    }
    #[test]
    fn test_operator_matmul_add_f16_dispatch_and_parallel_consistency() {
        use approx::assert_abs_diff_eq;
        use std::f16;

        // MR=3 => M % 3 == 0
        // NR=32 => N % 32 == 0
        // KC=64 => K % 64 == 0
        const M: usize = 6;
        const K: usize = 64;
        const N: usize = 32;

        // A[M×K], B_nt[N×K], residual[M×N], C[M×N]
        let mut a = vec![0.0f16; M * K];
        let mut b_nt = vec![0.0f16; N * K];
        let mut residual = vec![0.0f16; M * N];

        // init A: a[i,kk]
        for i in 0..M {
            for kk in 0..K {
                let v = 0.01f32 * (i as f32) + 0.001f32 * (kk as f32);
                a[i * K + kk] = v as f16;
            }
        }

        // init B_nt: NT layout [N×K] row-major (each row stride = K)
        // b_nt[j,kk] = 0.02*kk + 0.003*j
        for j in 0..N {
            for kk in 0..K {
                let v = 0.02f32 * (kk as f32) + 0.003f32 * (j as f32);
                b_nt[j * K + kk] = v as f16;
            }
        }

        // init residual: residual[i,j]
        for i in 0..M {
            for j in 0..N {
                let v = 0.05f32 * (i as f32) + 0.0007f32 * (j as f32);
                residual[i * N + j] = v as f16;
            }
        }

        let params = crate::common::matmul_params::MatMulParams {
            a_row_step_macro: M,  // MB
            b_row_step_macro: N,  // NB
            column_step_macro: K, // KC
            a_row_step_micro: 3,  // MR
            b_row_step_micro: 32, // NR
        };

        // ===== cpu_num = 1 =====
        let mut c1 = vec![0.0f16; M * N];
        let runner1 = unsafe {
            crate::operators::linear::MatMulAdd::<f16>::new(
                a.as_ptr(),
                b_nt.as_ptr(),     // ✅ 传 NT
                residual.as_ptr(), // ✅ residual
                c1.as_mut_ptr(),
                params,
                M,
                N,
                K,
            )
        };
        let op1 = super::Operator::MatMulAdd(runner1.clone());

        let batch_size = M;
        let decode_size = 1;

        op1.run(batch_size, decode_size, 1, 0, &[], &[], &mut Vec::new());

        // ===== cpu_num = thread_num =====
        let mut c2 = vec![0.0f16; M * N];
        let runner2 = unsafe {
            crate::operators::linear::MatMulAdd::<f16>::new(
                a.as_ptr(),
                b_nt.as_ptr(),     // ✅ 传 NT
                residual.as_ptr(), // ✅ residual
                c2.as_mut_ptr(),
                params,
                M,
                N,
                K,
            )
        };
        let op2 = super::Operator::MatMulAdd(runner2.clone());

        // 若 MatMulAdd 没有 panel_threads()，就用一个保守策略：最多 16
        //（你其它测试也经常 min(16)）
        let thread_num = num_cpus::get().min(16).max(1);

        for tid in 0..thread_num {
            op2.run(
                batch_size,
                decode_size,
                thread_num,
                tid,
                &[],
                &[],
                &mut Vec::new(),
            );
        }

        // 1) 并行一致性
        for idx in 0..(M * N) {
            let x = c1[idx] as f32;
            let y = c2[idx] as f32;
            assert_abs_diff_eq!(x, y, epsilon = 1e-1);
        }

        // 2) 正确性：reference 用 NT 读法 + residual
        for i in 0..M {
            for j in 0..N {
                let mut sum = residual[i * N + j] as f32;
                for kk in 0..K {
                    // A[i,kk] * B_nt[j,kk]
                    sum += (a[i * K + kk] as f32) * (b_nt[j * K + kk] as f32);
                }
                let got = c2[i * N + j] as f32;
                assert_abs_diff_eq!(got, sum, epsilon = 2e-1);
            }
        }
    }
}

/*
 #[test]
    fn test_rms() {
        let sequence_chunk_size = 1;
        let batch_size = 10;
        let hidden_size = 18;
        let cpu_num = num_cpus::get();

        let prefill_size = sequence_chunk_size * batch_size;
        let decode_size = sequence_chunk_size;

        let shapes = vec![sequence_chunk_size, batch_size, hidden_size];
        let length = shapes.iter().product();
        let input_data: Vec<f32> = (1..=hidden_size)
            .cycle()
            .take(length)
            .map(|x| x as f32)
            .collect();
        let eps = 1e-6;
        let mut output_data: Vec<f32> = vec![0.0; length];

        let operator = Operator::RMSMap(RMSMap::new(
            input_data.as_ptr(),
            output_data.as_mut_ptr(),
            hidden_size,
            eps,
            false,
        ));

        let result = [
            0.09238425642251968,
            0.18476851284503937,
            0.27715277671813965,
            0.36953702569007874,
            0.4619212746620178,
            0.5543055534362793,
            0.646689772605896,
            0.7390740513801575,
            0.831458330154419,
            0.9238425493240356,
            1.0162267684936523,
            1.1086111068725586,
            1.2009953260421753,
            1.293379545211792,
            1.3857638835906982,
            1.478148102760315,
            1.5705323219299316,
            1.662916660308838,
        ];
        let thread_num: usize = cpu_num;
        for i in 0..thread_num {
            operator.run(
                prefill_size,
                decode_size,
                cpu_num,
                i,
                &[],
                &[],
                &mut Vec::new(),
            );
        }
        assert_ulps_eq!(output_data[18..36], result, max_ulps = 4);
        println!("{:?}", output_data);
    }



#[test]
    fn test_add_zip() {
        let sequence_chunk_size = 1;
        let batch_size = 10;
        let head_num = 3;
        let head_size = 6;

        let prefill_size = sequence_chunk_size * batch_size;
        let decode_size = sequence_chunk_size;

        let shapes = vec![sequence_chunk_size, batch_size, head_num, head_size];
        let length = shapes.iter().product();

        let input_data1: Vec<f32> = (0..=17).cycle().take(length).map(|x| x as f32).collect();
        let input_data2: Vec<f32> = vec![1.0; length];
        let results: Vec<f32> = (1..=18).cycle().take(length).map(|x| x as f32).collect();
        let mut output_data: Vec<f32> = vec![0.0; length];

        let thread_num: usize = num_cpus::get();
        let operator = Operator::AddZipMap(AddZipMap::new(
            input_data1.as_ptr(),
            input_data2.as_ptr(),
            output_data.as_mut_ptr(),
            batch_size,
            head_num,
            head_size,
        ));

        for i in 0..thread_num {
            operator.run(prefill_size, decode_size, thread_num, i, &[], &[], &mut Vec::new());
        }

        assert_ulps_eq!(output_data[0..180], results[0..180], max_ulps = 4);
        println!("{:?}", output_data);
    }
#[test]
    fn test_complexmul() {
        let sequence_length = 10;
        let sequence_chunk_size = 8;
        let batch_size = 10;
        let head_num = 10;
        let head_size = 34;

        let prefill_size = sequence_chunk_size * batch_size;
        let decode_size = sequence_chunk_size;

        let shape1 = vec![sequence_chunk_size, batch_size, head_num, head_size];
        let shape2 = vec![sequence_length, head_size];

        let length1: usize = shape1.iter().product();
        let length2: usize = shape2.iter().product();
        let length = length1;
        let input_data1: Vec<f32> = (1..=head_size)
            .cycle()
            .take(length1)
            .map(|x| x as f32)
            .collect();
        let input_data2: Vec<f32> = (1..=head_size)
            .cycle()
            .take(length2)
            .map(|x| x as f32)
            .collect();
        let mut output_data: Vec<f32> = vec![0.0; length];

        let expected: Vec<f32> = vec![
            -3.0, 4.0, -7.0, 24.0, -11.0, 60.0, -15.0, 112.0, -19.0, 180.0, -23.0, 264.0, -27.0,
            364.0, -31.0, 480.0, -35.0, 612.0, -39.0, 760.0, -43.0, 924.0, -47.0, 1104.0, -51.0,
            1300.0, -55.0, 1512.0, -59.0, 1740.0, -63.0, 1984.0, -67.0, 2244.0,
        ];

        let thread_num: usize = num_cpus::get();
        let mut operator = Operator::ComplexZipMap(ComplexZipMap::<f32>::new(
            input_data1.as_ptr(),
            input_data2.as_ptr(),
            output_data.as_mut_ptr(),
            // sequence_chunk_size,
            batch_size,
            head_num,
            head_size,
            false,
        ));

        for i in 0..thread_num {
            operator.run(prefill_size, decode_size, thread_num, i, &[], &[], &mut Vec::new());
        }

        assert_eq!(output_data[0..34], expected);
    }

    #[test]
    fn test_silu() {
        let sequence_chunk_size = 8;
        let batch_size = 10;
        // let hidden_size = 19;
        let head_num = 1;
        let head_size = 19;

        let prefill_size = sequence_chunk_size * batch_size;
        let decode_size = sequence_chunk_size;

        let shapes = vec![sequence_chunk_size, batch_size, head_num, head_size];

        let length = shapes.iter().product();
        let input_data1: Vec<f32> = vec![
            2.1671206951141357,
            1.4490455389022827,
            -2.002431631088257,
            0.5662149786949158,
            0.3909946382045746,
            0.9437483549118042,
            -0.37030690908432007,
            0.7542704939842224,
            0.5875813961029053,
            1.6026240587234497,
            2.2485475540161133,
            -0.6622593402862549,
            -0.0015666020335629582,
            -0.5069465041160583,
            -0.37254711985588074,
            0.4420417249202728,
            -0.9305257201194763,
            0.5145581364631653,
            0.6260590553283691,
        ]
        .repeat(sequence_chunk_size * batch_size);
        // let input_data2: [f32; 190] = [1.0; 190];

        let mut input_data2: Vec<f32> = vec![1.0; length];
        let mut output_data: Vec<f32> = vec![0.0; length];

        let thread_num: usize = num_cpus::get();
        let mut operator = Operator::SiluMulZipMap(SiluMulZipMap::new(
            input_data1.as_ptr(),
            input_data2.as_ptr(),
            output_data.as_mut_ptr(),
            batch_size,
            head_num,
            head_size,
        ));

        for i in 0..thread_num {
            operator.run(prefill_size, decode_size, thread_num, i, &[], &[], &mut Vec::new());
        }
        let result = vec![
            1.9444659948349,
            1.1735117435455322,
            -0.23818494379520416,
            0.36118248105049133,
            0.23323695361614227,
            0.6793630719184875,
            -0.15125809609889984,
            0.5129857659339905,
            0.3777032196521759,
            1.3339999914169312,
            2.033867835998535,
            -0.22532200813293457,
            -0.0007826874498277903,
            -0.1905660629272461,
            -0.15197153389453888,
            0.269090861082077,
            -0.2631694972515106,
            0.32204875349998474,
            0.4079371392726898,
        ]
        .repeat(sequence_chunk_size * batch_size);
        assert_ulps_eq!(output_data[..], result, max_ulps = 4);
    }

*/
