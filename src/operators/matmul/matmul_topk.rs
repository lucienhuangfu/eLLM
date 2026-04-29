// === compiler/mul/matmul_topk.rs ===
#![allow(non_snake_case)]

use std::f16;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

use crate::common::heap::FixedMinHeap;
use crate::common::{
    matmul_params::MatMulParams,
    send_sync_ptr::{ConstPtr, MutPtr},
};
use crate::kernel;
use crate::operators::assign::assign;
use crate::operators::traits::MatMulTopKTrait;

#[derive(Clone)]
pub struct MatMulTopK<T> {
    // A / B / 输出 top-k
    ptr1: ConstPtr<T>,         // A[M×K]
    ptr2: ConstPtr<T>,         // ✅ B_nt[N×K]
    indice_ptr: MutPtr<usize>, // indices buffer: [batch_max][thread_max][TOPK]
    value_ptr: MutPtr<T>,      // values buffer : [batch_max][thread_max][TOPK]

    // 维度
    a_row: usize,  // M_max
    b_row: usize,  // N_max
    column: usize, // K_max

    pub params: MatMulParams,

    topk: usize,
    batch_max: usize,

    // ✅ 内部 thread_max（用于 scratch/heaps 绑定）
    thread_max: usize,

    // 预打包的 B panel：[panels_k][panels_n][kc*nr]
    packed_b: Box<[T]>,
    packed_panel_stride: usize, // = kc * nr

    // 每线程的 C_tile 池：[thread_max][mr×nr]
    c_tile_pool: Box<[T]>,
    c_tile_stride_elems: usize, // = mr * nr

    // 每 (batch, thread) 一棵 heap
    heaps: Box<[FixedMinHeap<T>]>,

    _marker: PhantomData<T>,
}

impl<T> MatMulTopK<T>
where
    T: Copy + Default + PartialOrd + Add<Output = T> + Mul<Output = T>,
{
    #[inline]
    pub fn detect_threads() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        ptr1: *const T,          // A[M×K]
        ptr2_b_nt_nxk: *const T, // ✅ B_nt[N×K]（按行连续），不再转置
        indice_ptr: *mut usize,  // indices 输出 buffer（至少 batch_max*thread_max*topk）
        value_ptr: *mut T,       // values 输出 buffer（至少 batch_max*thread_max*topk）
        a_row: usize,            // M_max
        b_row: usize,            // N_max
        column: usize,           // K_max
        a_row_step_macro: usize,
        b_row_step_macro: usize,
        column_step_macro: usize,
        a_row_step_micro: usize,
        b_row_step_micro: usize,
        batch_max: usize,
        topk: usize,
    ) -> Self {
        let params = MatMulParams {
            a_row_step_macro,
            b_row_step_macro,
            column_step_macro,
            a_row_step_micro,
            b_row_step_micro,
        };

        let m_max = a_row;
        let n_max = b_row;
        let k_max = column;

        // ✅ 内部决定 thread_max
        let thread_max = Self::detect_threads();

        let kc = params.column_step_macro.max(1);
        let nr = params.b_row_step_micro.max(1);
        let mr = params.a_row_step_micro.max(1);

        let packed_panel_stride = kc * nr;
        let packed_b = Self::pack_b_panels(ptr2_b_nt_nxk, n_max, k_max, kc, nr);

        let c_tile_stride_elems = mr * nr;
        let c_tile_pool_len = thread_max * c_tile_stride_elems;
        let c_tile_pool: Vec<T> = vec![T::default(); c_tile_pool_len];

        // === (2) heaps: [batch_max][thread_max]，绑定到输出 buffer 上 ===
        let stride_thread = topk;
        let stride_batch = thread_max * topk;

        let mut heaps_vec: Vec<FixedMinHeap<T>> = Vec::with_capacity(batch_max * thread_max);

        for b in 0..batch_max {
            for tid in 0..thread_max {
                let values_base = value_ptr.add(b * stride_batch + tid * stride_thread);
                let indices_base = indice_ptr.add(b * stride_batch + tid * stride_thread);
                heaps_vec.push(FixedMinHeap::new(values_base, indices_base, topk));
            }
        }

        Self {
            ptr1: ConstPtr { ptr: ptr1 },
            ptr2: ConstPtr { ptr: ptr2_b_nt_nxk },
            indice_ptr: MutPtr { ptr: indice_ptr },
            value_ptr: MutPtr { ptr: value_ptr },

            a_row: m_max,
            b_row: n_max,
            column: k_max,

            params,
            topk,
            batch_max,
            thread_max,

            packed_b,
            packed_panel_stride,
            c_tile_pool: c_tile_pool.into_boxed_slice(),
            c_tile_stride_elems,

            heaps: heaps_vec.into_boxed_slice(),
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    fn thread_c_tile_ptr(&self, thread_id: usize) -> *mut T {
        debug_assert!(thread_id < self.thread_max);
        unsafe {
            self.c_tile_pool
                .as_ptr()
                .add(thread_id * self.c_tile_stride_elems) as *mut T
        }
    }

    /// ✅ 不改你风格：返回 *mut FixedMinHeap<T>
    #[inline(always)]
    fn heap_for(&self, batch: usize, thread_id: usize) -> *mut FixedMinHeap<T> {
        debug_assert!(batch < self.batch_max);
        debug_assert!(thread_id < self.thread_max);
        let idx = batch * self.thread_max + thread_id;
        debug_assert!(idx < self.heaps.len());
        unsafe { self.heaps.as_ptr().add(idx) as *mut FixedMinHeap<T> }
    }

    #[inline(always)]
    fn pack_b_panels(b_nt: *const T, n: usize, k: usize, kc: usize, nr: usize) -> Box<[T]> {
        let panels_k = k.div_ceil(kc);
        let panels_n = n.div_ceil(nr);
        let panel_stride = kc * nr;
        let mut packed = vec![T::default(); panels_k * panels_n * panel_stride];

        unsafe {
            for kb in 0..panels_k {
                let k0 = kb * kc;
                let kc_cur = (k - k0).min(kc);
                for nb in 0..panels_n {
                    let n0 = nb * nr;
                    let nr_cur = (n - n0).min(nr);
                    let panel = packed.as_mut_ptr().add((kb * panels_n + nb) * panel_stride);
                    for p in 0..kc_cur {
                        let dst_row = panel.add(p * nr);
                        for lane in 0..nr_cur {
                            *dst_row.add(lane) = *b_nt.add((n0 + lane) * k + (k0 + p));
                        }
                    }
                }
            }
        }

        packed.into_boxed_slice()
    }

    #[inline(always)]
    fn packed_panel_ptr(&self, n0: usize, k0: usize) -> *const T {
        let kc = self.params.column_step_macro.max(1);
        let nr = self.params.b_row_step_micro.max(1);
        let panels_n = self.b_row.div_ceil(nr);
        let panel_idx = (k0 / kc) * panels_n + (n0 / nr);
        unsafe {
            self.packed_b
                .as_ptr()
                .add(panel_idx * self.packed_panel_stride)
        }
    }

    pub fn run(
        &self,
        prefill_size: usize,
        _decode_size: usize,
        thread_num: usize,
        thread_id: usize,
    ) {
        unsafe {
            assert!(prefill_size <= self.batch_max);

            // ✅ cpu_num/thread_id 合法，且 cpu_num <= thread_max
            assert!(thread_num <= self.thread_max);
            assert!(thread_id < thread_num);

            let m_run = prefill_size;
            let n = self.b_row;
            let k = self.column;

            let mb = self.params.a_row_step_macro.max(1);
            let nb = self.params.b_row_step_macro.max(1);
            let kc = self.params.column_step_macro.max(1);
            let mr = self.params.a_row_step_micro.max(1);
            let nr = self.params.b_row_step_micro.max(1);

            // === 关键：M 向上 pad 到 MR 的倍数（空算），保证固定 MR 微核/循环不炸 ===
            let m_pad = ((m_run + mr - 1) / mr) * mr;

            // new() 预留的 A 行容量必须覆盖到 m_pad
            debug_assert!(m_pad <= self.a_row);

            // 你当前的块内循环仍要求 m_blk%mr==0；
            // 这轮不做 tile 内 tail，要求 MB 是 MR 的倍数
            debug_assert!(mb % mr == 0);

            debug_assert!(n % nr == 0);
            debug_assert!(k % kc == 0);

            let a_base = self.ptr1.ptr;
            let lda = k;
            let c_tile_ptr = self.thread_c_tile_ptr(thread_id);

            // 清本线程 heap：只清真实 batch
            for b in 0..m_run {
                let heap_ptr = self.heap_for(b, thread_id);
                (*heap_ptr).clear();
            }

            // === tiling 走 m_pad 而不是 m_run ===
            let tiles_m = (m_pad + mb - 1) / mb;
            let tiles_n = (n + nb - 1) / nb;
            let tiles_total = tiles_m * tiles_n;

            if let Some((tb, te)) = assign(tiles_total, thread_num, thread_id) {
                for t in tb..te {
                    let tm = t / tiles_n;
                    let tn = t % tiles_n;

                    let m0 = tm * mb;
                    let n0 = tn * nb;

                    let m_blk = (m_pad - m0).min(mb);
                    let n_blk = (n - n0).min(nb);

                    debug_assert!(m_blk % mr == 0);
                    debug_assert!(n_blk % nr == 0);

                    let mut mi = 0usize;
                    while mi < m_blk {
                        let global_m_start = m0 + mi;

                        let mut nt = 0usize;
                        while nt < n_blk {
                            let global_n_start = n0 + nt;

                            // 清 tile
                            for i in 0..(mr * nr) {
                                *c_tile_ptr.add(i) = T::default();
                            }

                            let mut k0 = 0usize;
                            while k0 < k {
                                let a_tile = a_base.add(global_m_start * lda + k0);
                                let b_panel_ptr = self.packed_panel_ptr(global_n_start, k0);

                                self.compute(a_tile, b_panel_ptr, c_tile_ptr);

                                k0 += kc;
                            }

                            // 写 heap：仍然只写 batch_size 范围内的行（你原本就 guard 了）
                            for r in 0..mr {
                                let batch_idx = global_m_start + r;
                                if batch_idx >= m_run {
                                    continue;
                                }
                                let heap_ptr = self.heap_for(batch_idx, thread_id);
                                let heap = &mut *heap_ptr;

                                for c in 0..nr {
                                    let col_idx = global_n_start + c;
                                    let v = *c_tile_ptr.add(r * nr + c);
                                    heap.push(v, col_idx);
                                }
                            }

                            nt += nr;
                        }

                        mi += mr;
                    }
                }
            }

            // sort：只对真实 batch 做
            for b in 0..m_run {
                let heap_ptr = self.heap_for(b, thread_id);
                (*heap_ptr).sort_desc();
            }
        }
    }

    #[inline]
    pub fn thread_max(&self) -> usize {
        self.thread_max
    }
}

/* ------------------ 微核 compute：保持你的调用风格 ------------------ */

impl<T> MatMulTopKTrait<T> for MatMulTopK<T>
where
    T: Copy + Default + PartialOrd + Add<Output = T> + Mul<Output = T>,
{
    default fn compute(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T) {
        let mr = self.params.a_row_step_micro.max(1);
        let nr = self.params.b_row_step_micro.max(1);

        let call_param = MatMulParams {
            a_row_step_macro: self.column,
            b_row_step_macro: nr,
            column_step_macro: self.params.column_step_macro,
            a_row_step_micro: mr,
            b_row_step_micro: nr,
        };

        kernel::scalar::matmul_block::matmul_block(input_ptr1, input_ptr2, output_ptr, &call_param);
    }
}

impl MatMulTopKTrait<f16> for MatMulTopK<f16> {
    fn compute(&self, input_ptr1: *const f16, input_ptr2: *const f16, output_ptr: *mut f16) {
        let mr = self.params.a_row_step_micro.max(1);
        let nr = self.params.b_row_step_micro.max(1);

        let call_param = MatMulParams {
            a_row_step_macro: self.column,
            b_row_step_macro: nr,
            column_step_macro: self.params.column_step_macro,
            a_row_step_micro: mr,
            b_row_step_micro: nr,
        };

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::matmul_block::matmul_block(
                input_ptr1,
                input_ptr2,
                output_ptr,
                &call_param,
            );
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        {
            kernel::scalar::matmul_block::matmul_block(
                input_ptr1,
                input_ptr2,
                output_ptr,
                &call_param,
            );
        }
    }
}

impl MatMulTopKTrait<f32> for MatMulTopK<f32> {
    fn compute(&self, input_ptr1: *const f32, input_ptr2: *const f32, output_ptr: *mut f32) {
        let mr = self.params.a_row_step_micro.max(1);
        let nr = self.params.b_row_step_micro.max(1);

        let call_param = MatMulParams {
            a_row_step_macro: self.column,
            b_row_step_macro: nr,
            column_step_macro: self.params.column_step_macro,
            a_row_step_micro: mr,
            b_row_step_micro: nr,
        };

        kernel::scalar::matmul_block::matmul_block(input_ptr1, input_ptr2, output_ptr, &call_param);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn verify_topk_result_from_bnt(
        m: usize,
        k: usize,
        n: usize,
        topk: usize,
        cpu_num: usize,
        thread_max: usize,
        a: &[f16],
        b_nt: &[f16], // ✅ N×K
        indices_buf: &[usize],
        values_buf: &[f16],
        epsilon: f32,
    ) {
        for i in 0..m {
            // (1) 参考全量：row_c[j] = dot(a_i, b_nt[j])
            let mut row_c = vec![0.0f32; n];
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += (a[i * k + kk] as f32) * (b_nt[j * k + kk] as f32);
                }
                row_c[j] = sum;
            }

            // (2) 参考 topk
            let mut indexed_row: Vec<(usize, f32)> = row_c.into_iter().enumerate().collect();
            indexed_row.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let expected_topk = &indexed_row[0..topk];

            // (3) 合并所有线程局部 topk
            let mut merged: Vec<(usize, f32)> = Vec::with_capacity(cpu_num * topk);
            for tid in 0..cpu_num {
                let offset = i * (thread_max * topk) + tid * topk;
                for r in 0..topk {
                    merged.push((indices_buf[offset + r], values_buf[offset + r] as f32));
                }
            }

            merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let final_topk = &merged[0..topk];

            // (4) 对比
            for r in 0..topk {
                let (exp_idx, exp_val) = expected_topk[r];
                let (got_idx, got_val) = final_topk[r];

                assert_abs_diff_eq!(got_val, exp_val, epsilon = epsilon);

                if (got_val - exp_val).abs() < 1e-5 {
                    assert_eq!(got_idx, exp_idx, "Mismatch at row {}, rank {}", i, r);
                }
            }
        }
    }

    #[test]
    fn test_matmul_topk_f16_3x64x32() {
        const M: usize = 3;
        const K: usize = 64;
        const N: usize = 32;
        const TOPK: usize = 10;

        let cpu_num = 4usize;

        let mut a = vec![0.0 as f16; M * K];
        let mut b_nt = vec![0.0 as f16; N * K]; // ✅ N×K

        for i in 0..M {
            for kk in 0..K {
                a[i * K + kk] = ((i + kk) as f32 * 0.01) as f16;
            }
        }
        for j in 0..N {
            for kk in 0..K {
                b_nt[j * K + kk] = ((kk + j) as f32 * 0.001) as f16;
            }
        }

        unsafe {
            let thread_max = MatMulTopK::<f16>::detect_threads();
            let buf_len = M * thread_max * TOPK;
            let mut indices_buf = vec![0usize; buf_len];
            let mut values_buf = vec![0.0 as f16; buf_len];

            let runner = MatMulTopK::<f16>::new(
                a.as_ptr(),
                b_nt.as_ptr(), // ✅ 直接传 B_nt
                indices_buf.as_mut_ptr(),
                values_buf.as_mut_ptr(),
                M,
                N,
                K,
                3,
                32,
                64,
                3,
                32,
                M,
                TOPK,
            );

            let used = cpu_num.min(runner.thread_max());
            for tid in 0..used {
                runner.run(M, 0, used, tid);
            }

            verify_topk_result_from_bnt(
                M,
                K,
                N,
                TOPK,
                used,
                runner.thread_max(),
                &a,
                &b_nt,
                &indices_buf,
                &values_buf,
                0.01,
            );
        }
    }

    #[test]
    fn test_matmul_topk_f16_24x256x512() {
        const M: usize = 24;
        const K: usize = 256;
        const N: usize = 512;
        const TOPK: usize = 10;

        let cpu_num = 8usize;

        let mut a = vec![0.0 as f16; M * K];
        let mut b_nt = vec![0.0 as f16; N * K]; // ✅ N×K

        for i in 0..M {
            for kk in 0..K {
                let v = ((i * 131 + kk * 17) % 97) as f32 * 0.01;
                a[i * K + kk] = v as f16;
            }
        }
        for j in 0..N {
            for kk in 0..K {
                let v = ((kk * 73 + j * 11) % 101) as f32 * 0.01;
                b_nt[j * K + kk] = v as f16;
            }
        }

        unsafe {
            let thread_max = MatMulTopK::<f16>::detect_threads();
            let buf_len = M * thread_max * TOPK;
            let mut indices_buf = vec![0usize; buf_len];
            let mut values_buf = vec![0.0 as f16; buf_len];

            let runner = MatMulTopK::<f16>::new(
                a.as_ptr(),
                b_nt.as_ptr(), // ✅
                indices_buf.as_mut_ptr(),
                values_buf.as_mut_ptr(),
                M,
                N,
                K,
                24,
                128,
                64,
                3,
                32,
                M,
                TOPK,
            );

            let used = cpu_num.min(runner.thread_max());
            for tid in 0..used {
                runner.run(M, 0, used, tid);
            }

            verify_topk_result_from_bnt(
                M,
                K,
                N,
                TOPK,
                used,
                runner.thread_max(),
                &a,
                &b_nt,
                &indices_buf,
                &values_buf,
                0.5,
            );
        }
    }

    #[test]
    fn test_matmul_topk_f16_large_like_144x2048x2048_smoke() {
        const M: usize = 144;
        const K: usize = 2048;
        const N: usize = 2048;
        const TOPK: usize = 10;

        let cpu_num = 8usize;

        let mut a = vec![0.0 as f16; M * K];
        let mut b_nt = vec![0.0 as f16; N * K]; // ✅ N×K

        for i in 0..M {
            for kk in 0..K {
                a[i * K + kk] = (((i + kk) % 7) as f32 * 0.01) as f16;
            }
        }
        for j in 0..N {
            for kk in 0..K {
                b_nt[j * K + kk] = (((kk + j) % 11) as f32 * 0.01) as f16;
            }
        }

        unsafe {
            let thread_max = MatMulTopK::<f16>::detect_threads();
            let buf_len = M * thread_max * TOPK;
            let mut indices_buf = vec![0usize; buf_len];
            let mut values_buf = vec![0.0 as f16; buf_len];

            let runner = MatMulTopK::<f16>::new(
                a.as_ptr(),
                b_nt.as_ptr(), // ✅
                indices_buf.as_mut_ptr(),
                values_buf.as_mut_ptr(),
                M,
                N,
                K,
                24,
                128,
                64,
                3,
                32,
                M,
                TOPK,
            );

            let used = cpu_num.min(runner.thread_max());
            for tid in 0..used {
                runner.run(M, 0, used, tid);
            }

            verify_topk_result_from_bnt(
                M,
                K,
                N,
                TOPK,
                used,
                runner.thread_max(),
                &a,
                &b_nt,
                &indices_buf,
                &values_buf,
                0.8,
            );
        }
    }
    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_matmul_topk_f16_batch7_pad_to9_no_ties() {
        const M_RUN: usize = 7; // 非 3 倍数
        const M_MAX: usize = 9; // pad 后
        const K: usize = 64;
        const N: usize = 64; // n 必须是 32 的倍数
        const TOPK: usize = 8;

        let cpu_num = 4usize;

        // A: [M_MAX×K]，前 7 行全 1，pad 行全 0
        let mut a = vec![0.0f16; M_MAX * K];
        for i in 0..M_RUN {
            for kk in 0..K {
                a[i * K + kk] = 1.0f32 as f16;
            }
        }

        // B_nt: [N×K]
        // 让每一行 j 的值是常数 bias=j（严格递增），这样 dot = K * bias，严格递增，无 ties
        let mut b_nt = vec![0.0f16; N * K];
        for j in 0..N {
            let bias = (j as f32) * 0.01;
            for kk in 0..K {
                b_nt[j * K + kk] = bias as f16;
            }
        }

        unsafe {
            let thread_max = MatMulTopK::<f16>::detect_threads();

            // indices/values buffer 按 batch_max=M_MAX 分配（capacity），但这次 run 只会写前 M_RUN
            let buf_len = M_MAX * thread_max * TOPK;
            let mut indices_buf = vec![0usize; buf_len];
            let mut values_buf = vec![0.0f16; buf_len];

            let runner = MatMulTopK::<f16>::new(
                a.as_ptr(),
                b_nt.as_ptr(),
                indices_buf.as_mut_ptr(),
                values_buf.as_mut_ptr(),
                M_MAX, // a_row (capacity)
                N,
                K,
                6,     // MB（3 的倍数）
                64,    // NB
                64,    // KC
                3,     // MR
                32,    // NR
                M_MAX, // batch_max (capacity)
                TOPK,
            );

            let used = cpu_num.min(runner.thread_max());
            for tid in 0..used {
                runner.run(M_RUN, 0, used, tid); // batch_size=7
            }

            // 期望 topk：因为输出随 j 单调递增，topk 就是最大的 TOPK 个列索引
            // 即 [N-1, N-2, ..., N-TOPK]
            let expected: Vec<usize> = (0..TOPK).map(|r| N - 1 - r).collect();

            // 合并所有线程的局部 topk 后再取最终 topk（和你 verify 的方式一致）
            for i in 0..M_RUN {
                let mut merged: Vec<(usize, f32)> = Vec::with_capacity(used * TOPK);
                for tid in 0..used {
                    let offset = i * (thread_max * TOPK) + tid * TOPK;
                    for r in 0..TOPK {
                        merged.push((indices_buf[offset + r], values_buf[offset + r] as f32));
                    }
                }
                merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let final_topk: Vec<usize> = merged[..TOPK].iter().map(|x| x.0).collect();

                // 只检查索引集合即可（顺序应该也是降序）
                assert_eq!(final_topk, expected, "row {} topk mismatch", i);
            }
        }
    }
}
