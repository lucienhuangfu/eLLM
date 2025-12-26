// === runner/matmul_topk.rs ===
#![allow(non_snake_case)]

use std::f16;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

use super::super::super::init::{
    matmul_params::MatMulParams,
    send_sync_ptr::{ConstPtr, MutPtr},
};
use super::super::super::kernel;
use super::super::super::kernel::common::heap::FixedMinHeap;
use super::super::assign::assign;
use super::mul_trait::MatMulTopKTrait;

#[derive(Clone)]
pub struct MatMulTopK<T>
where
    T: PartialOrd + Copy,
{
    // A / B / 输出 top-k
    ptr1: ConstPtr<T>,         // A[M×K]
    ptr2: ConstPtr<T>,         // 指向 B_nt[N×K]
    indice_ptr: MutPtr<usize>, // indices buffer: [batch_max][thread_max][TOPK]
    value_ptr: MutPtr<T>,      // values buffer : [batch_max][thread_max][TOPK]

    // 维度
    a_row: usize,  // M_max
    b_row: usize,  // N_max
    column: usize, // K_max

    pub params: MatMulParams,

    topk: usize,
    batch_max: usize,

    // ✅ 不再存 cpu_max_for_scratch，改为内部 thread_max
    thread_max: usize,

    // 构造期转置出来的 B_nt（N×K，行主；行距=K）
    b_nt_buf: Box<[T]>,

    // 每线程的 B 面板池：[thread_max][kc×nr]
    b_panel_pool: Box<[T]>,
    b_panel_stride_elems: usize, // = kc * nr

    // 每线程的 C_tile 池：[thread_max][mr×nr]
    c_tile_pool: Box<[T]>,
    c_tile_stride_elems: usize, // = mr * nr

    // ✅ 不改：每 (batch, thread) 一棵 heap
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
        ptr1: *const T,         // A[M×K]
        ptr2_b_kxn: *const T,   // B[K×N]，构造期使用一次
        indice_ptr: *mut usize, // indices 输出 buffer（必须至少 batch_max*thread_max*topk）
        value_ptr: *mut T,      // values 输出 buffer（必须至少 batch_max*thread_max*topk）
        a_row: usize,           // M_max
        b_row: usize,           // N_max
        column: usize,          // K_max
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

        // === (1) 构造期：B[K×N] → B_nt[N×K] ===
        let mut b_nt_vec: Vec<T> = vec![T::default(); n_max * k_max];
        let b_nt_ptr = b_nt_vec.as_mut_ptr();

        for kk in 0..k_max {
            let b_row_src = ptr2_b_kxn.add(kk * n_max);
            for jj in 0..n_max {
                *b_nt_ptr.add(jj * k_max + kk) = *b_row_src.add(jj);
            }
        }
        let b_nt_box = b_nt_vec.into_boxed_slice();
        let b_nt_base = b_nt_box.as_ptr();

        // === (2) scratch pools: [thread_max][...] ===
        let kc = params.column_step_macro.max(1);
        let nr = params.b_row_step_micro.max(1);
        let mr = params.a_row_step_micro.max(1);

        let b_panel_stride_elems = kc * nr;
        let b_panel_pool_len = thread_max * b_panel_stride_elems;
        let b_panel_pool: Vec<T> = vec![T::default(); b_panel_pool_len];

        let c_tile_stride_elems = mr * nr;
        let c_tile_pool_len = thread_max * c_tile_stride_elems;
        let c_tile_pool: Vec<T> = vec![T::default(); c_tile_pool_len];

        // === (3) heaps: [batch_max][thread_max]，绑定到输出 buffer 上 ===
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
            ptr2: ConstPtr { ptr: b_nt_base },
            indice_ptr: MutPtr { ptr: indice_ptr },
            value_ptr: MutPtr { ptr: value_ptr },

            a_row: m_max,
            b_row: n_max,
            column: k_max,

            params,
            topk,
            batch_max,
            thread_max,

            b_nt_buf: b_nt_box,
            b_panel_pool: b_panel_pool.into_boxed_slice(),
            b_panel_stride_elems,
            c_tile_pool: c_tile_pool.into_boxed_slice(),
            c_tile_stride_elems,

            heaps: heaps_vec.into_boxed_slice(),
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    fn thread_b_panel_ptr(&self, thread_id: usize) -> *mut T {
        debug_assert!(thread_id < self.thread_max);
        unsafe {
            self.b_panel_pool
                .as_ptr()
                .add(thread_id * self.b_panel_stride_elems) as *mut T
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
    unsafe fn pack_b_panel(
        &self,
        b_nt: *const T,
        ldb_row: usize,
        n0: usize,
        k0: usize,
        kc: usize,
        nr: usize,
        out: *mut T,
    ) {
        for p in 0..kc {
            let src_col = k0 + p;
            let dst_row = out.add(p * nr);
            for lane in 0..nr {
                let j = n0 + lane;
                let src = b_nt.add(j * ldb_row + src_col);
                *dst_row.add(lane) = *src;
            }
        }
    }

    pub fn run(
        &self,
        _position_index: usize,
        _position_interval: usize,
        batch_size: usize,
        cpu_num: usize,
        thread_id: usize,
    ) {
        unsafe {
            assert!(batch_size <= self.batch_max);

            // ✅ cpu 相关：只要求 cpu_num/thread_id 合法，且 cpu_num <= thread_max
            assert!(cpu_num <= self.thread_max);
            assert!(thread_id < cpu_num);

            let m = batch_size;
            let n = self.b_row;
            let k = self.column;

            let mb = self.params.a_row_step_macro.max(1);
            let nb = self.params.b_row_step_macro.max(1);
            let kc = self.params.column_step_macro.max(1);
            let mr = self.params.a_row_step_micro.max(1);
            let nr = self.params.b_row_step_micro.max(1);

            // 打印 n 和 nr 的值用于调试
            println!("[matmul_topk] n = {}, nr = {}", n, nr);

            // 你说：先假设整除
            debug_assert!(m % mr == 0);
            debug_assert!(n % nr == 0);
            debug_assert!(k % kc == 0);

            let a_base = self.ptr1.ptr;
            let lda = k;

            let b_nt_ptr = self.ptr2.ptr;
            let ldb_row = k;

            let b_panel_ptr = self.thread_b_panel_ptr(thread_id);
            let c_tile_ptr = self.thread_c_tile_ptr(thread_id);

            // 清本线程 heap
            for b in 0..batch_size {
                let heap_ptr = self.heap_for(b, thread_id);
                (*heap_ptr).clear();
            }

            let tiles_m = (m + mb - 1) / mb;
            let tiles_n = (n + nb - 1) / nb;
            let tiles_total = tiles_m * tiles_n;

            if let Some((tb, te)) = assign(tiles_total, cpu_num, thread_id) {
                for t in tb..te {
                    let tm = t / tiles_n;
                    let tn = t % tiles_n;

                    let m0 = tm * mb;
                    let n0 = tn * nb;

                    let m_blk = (m - m0).min(mb);
                    let n_blk = (n - n0).min(nb);

                    debug_assert!(m_blk % mr == 0);
                    debug_assert!(n_blk % nr == 0);

                    let mut mi = 0usize;
                    while mi < m_blk {
                        let global_m_start = m0 + mi;

                        let mut nt = 0usize;
                        while nt < n_blk {
                            let global_n_start = n0 + nt;

                            for i in 0..(mr * nr) {
                                *c_tile_ptr.add(i) = T::default();
                            }

                            let mut k0 = 0usize;
                            while k0 < k {
                                self.pack_b_panel(
                                    b_nt_ptr,
                                    ldb_row,
                                    global_n_start,
                                    k0,
                                    kc,
                                    nr,
                                    b_panel_ptr,
                                );

                                let a_tile = a_base.add(global_m_start * lda + k0);

                                self.compute(a_tile, b_panel_ptr as *const T, c_tile_ptr);

                                k0 += kc;
                            }

                            for r in 0..mr {
                                let batch_idx = global_m_start + r;
                                if batch_idx >= batch_size {
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

            for b in 0..batch_size {
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

        kernel::generic::matmul_block::matmul_block(
            input_ptr1,
            input_ptr2,
            output_ptr,
            &call_param,
        );
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
            kernel::generic::matmul_block::matmul_block(
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

        kernel::generic::matmul_block::matmul_block(
            input_ptr1,
            input_ptr2,
            output_ptr,
            &call_param,
        );
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn verify_topk_result(
        m: usize,
        k: usize,
        n: usize,
        topk: usize,
        cpu_num: usize,    // 实际跑的线程数
        thread_max: usize, // runner 内部 stride（heaps 绑定用这个）
        a: &[f16],
        b: &[f16],
        indices_buf: &[usize],
        values_buf: &[f16],
        epsilon: f32,
    ) {
        for i in 0..m {
            // (1) 参考全量 matmul
            let mut row_c = vec![0.0f32; n];
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += (a[i * k + kk] as f32) * (b[kk * n + j] as f32);
                }
                row_c[j] = sum;
            }

            // (2) 参考 topk
            let mut indexed_row: Vec<(usize, f32)> = row_c.into_iter().enumerate().collect();
            indexed_row.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let expected_topk = &indexed_row[0..topk];

            // (3) 合并所有线程的局部 topk（你原来的 reduce 思路）
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

                // 数值完全相等/极近时，索引也要求一致
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

        // 你想模拟多线程：随便取个 <= thread_max 的数
        let cpu_num = 4usize;

        let mut a = vec![0.0 as f16; M * K];
        let mut b = vec![0.0 as f16; K * N];

        for i in 0..M {
            for kk in 0..K {
                a[i * K + kk] = ((i + kk) as f32 * 0.01) as f16;
            }
        }
        for kk in 0..K {
            for j in 0..N {
                b[kk * N + j] = ((kk + j) as f32 * 0.001) as f16;
            }
        }

        unsafe {
            // 先构造 runner（内部决定 thread_max）
            // 注意：因为 runner 在 new() 里会把 heap 绑到输出 buffer，
            // 输出 buffer 必须按 thread_max 分配
            let thread_max = MatMulTopK::<f16>::detect_threads();
            let buf_len = M * thread_max * TOPK;
            let mut indices_buf = vec![0usize; buf_len];
            let mut values_buf = vec![0.0 as f16; buf_len];

            let runner = MatMulTopK::<f16>::new(
                a.as_ptr(),
                b.as_ptr(),
                indices_buf.as_mut_ptr(),
                values_buf.as_mut_ptr(),
                M,
                N,
                K,
                3,  // MB
                32, // NB
                64, // KC
                3,  // MR
                32, // NR
                M,  // batch_max
                TOPK,
            );

            let used = cpu_num.min(runner.thread_max());
            for tid in 0..used {
                runner.run(0, 0, M, used, tid);
            }

            verify_topk_result(
                M,
                K,
                N,
                TOPK,
                used,
                runner.thread_max(),
                &a,
                &b,
                &indices_buf,
                &values_buf,
                0.01,
            );
        }
    }

    #[test]
    fn test_matmul_topk_f16_24x256x512() {
        // 中等尺寸：能覆盖多 tile、多 heap push，但不会太慢
        const M: usize = 24;
        const K: usize = 256;
        const N: usize = 512;
        const TOPK: usize = 10;

        let cpu_num = 8usize;

        let mut a = vec![0.0 as f16; M * K];
        let mut b = vec![0.0 as f16; K * N];

        for i in 0..M {
            for kk in 0..K {
                let v = ((i * 131 + kk * 17) % 97) as f32 * 0.01;
                a[i * K + kk] = v as f16;
            }
        }
        for kk in 0..K {
            for j in 0..N {
                let v = ((kk * 73 + j * 11) % 101) as f32 * 0.01;
                b[kk * N + j] = v as f16;
            }
        }

        unsafe {
            let thread_max = MatMulTopK::<f16>::detect_threads();
            let buf_len = M * thread_max * TOPK;
            let mut indices_buf = vec![0usize; buf_len];
            let mut values_buf = vec![0.0 as f16; buf_len];

            let runner = MatMulTopK::<f16>::new(
                a.as_ptr(),
                b.as_ptr(),
                indices_buf.as_mut_ptr(),
                values_buf.as_mut_ptr(),
                M,
                N,
                K,
                24,  // MB（整除 M）
                128, // NB（整除 N）
                64,  // KC（整除 K）
                3,   // MR
                32,  // NR
                M,
                TOPK,
            );

            let used = cpu_num.min(runner.thread_max());
            for tid in 0..used {
                runner.run(0, 0, M, used, tid);
            }

            // 中等 K 累加误差：给个稍微宽松一点
            verify_topk_result(
                M,
                K,
                N,
                TOPK,
                used,
                runner.thread_max(),
                &a,
                &b,
                &indices_buf,
                &values_buf,
                0.5,
            );
        }
    }

    #[test]
    #[ignore]
    fn test_matmul_topk_f16_large_like_144x2048x2048_smoke() {
        // 大尺寸：默认 ignore，想跑手动打开
        const M: usize = 144;
        const K: usize = 2048;
        const N: usize = 2048;
        const TOPK: usize = 10;

        let cpu_num = 8usize;

        let mut a = vec![0.0 as f16; M * K];
        let mut b = vec![0.0 as f16; K * N];

        for i in 0..M {
            for kk in 0..K {
                a[i * K + kk] = (((i + kk) % 7) as f32 * 0.01) as f16;
            }
        }
        for kk in 0..K {
            for j in 0..N {
                b[kk * N + j] = (((kk + j) % 11) as f32 * 0.01) as f16;
            }
        }

        unsafe {
            let thread_max = MatMulTopK::<f16>::detect_threads();
            let buf_len = M * thread_max * TOPK;
            let mut indices_buf = vec![0usize; buf_len];
            let mut values_buf = vec![0.0 as f16; buf_len];

            let runner = MatMulTopK::<f16>::new(
                a.as_ptr(),
                b.as_ptr(),
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
                runner.run(0, 0, M, used, tid);
            }

            // 大 K 累加误差：放宽
            verify_topk_result(
                M,
                K,
                N,
                TOPK,
                used,
                runner.thread_max(),
                &a,
                &b,
                &indices_buf,
                &values_buf,
                0.8,
            );
        }
    }
}
