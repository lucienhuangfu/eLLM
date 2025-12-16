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
use super::mul_trait::MatMulTopKTrait; // 只在这里 use 一次

/// Top-K 版本的 MatMul：
/// - A: [batch_max, K]
/// - B: [K, N] (构造期转成 B_nt [N, K])
/// - 输出：values/indices 的逻辑布局为 [batch_max][cpu_max_for_scratch][TOPK]
///
/// 运行时：
/// - 对每个 batch、每个线程维护一个 FixedMinHeap<T>
/// - 用 2D 分块 + 3×32 微核计算 C 的每个小 tile
/// - tile 完成后，把 tile 中的标量丢进对应 heap 里
#[derive(Clone)]
pub struct MatMulTopK<T>
where
    T: PartialOrd + Copy,
{
    // A / B / 输出 top-k
    ptr1: ConstPtr<T>,         // A[M×K]
    ptr2: ConstPtr<T>,         // 指向 B_nt[N×K]
    indice_ptr: MutPtr<usize>, // indices buffer: [batch_max][cpu_max_for_scratch][TOPK]
    value_ptr: MutPtr<T>,      // values buffer : [batch_max][cpu_max_for_scratch][TOPK]

    // 维度
    a_row: usize,  // M_max
    b_row: usize,  // N_max
    column: usize, // K_max

    // blocking 参数（mb/nb/kc/mr/nr）
    pub params: MatMulParams,

    // top-k / 线程、batch 上限
    topk: usize,
    batch_max: usize,
    cpu_max_for_scratch: usize,

    // 构造期转置出来的 B_nt（N×K，行主；行距=K）
    b_nt_buf: Box<[T]>,

    // 每线程的 B 面板池：[cpu_max_for_scratch][kc×nr]
    b_panel_pool: Box<[T]>,
    b_panel_stride_elems: usize, // = kc * nr

    // 每线程的 C_tile 池：[cpu_max_for_scratch][mr×nr]
    c_tile_pool: Box<[T]>,
    c_tile_stride_elems: usize, // = mr * nr

    // 每 (batch, thread) 一棵 heap，长度 = batch_max * cpu_max_for_scratch
    // heap 自身只存指针和值/len，不再分配数组
    heaps: Box<[FixedMinHeap<T>]>,

    _marker: PhantomData<T>,
}

impl<T> MatMulTopK<T>
where
    T: Copy + Default + PartialOrd + Add<Output = T> + Mul<Output = T>,
{
    /// 构造函数：
    /// - ptr1: A[M×K]
    /// - ptr2_b_kxn: B[K×N]（构造期转成 B_nt[N×K]）
    /// - indice_ptr/value_ptr: 预先分配好的 top-k 输出 buffer
    ///
    /// 要求：
    /// - A 行数 = a_row
    /// - B 行数 = b_row (= N_max)
    /// - 列数 = column (= K_max)
    /// - 外部输出 buffer 大小至少为 [batch_max][cpu_max_for_scratch][topk]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        ptr1: *const T,         // A[M×K]
        ptr2_b_kxn: *const T,   // B[K×N]，构造期使用一次
        indice_ptr: *mut usize, // indices 输出 buffer
        value_ptr: *mut T,      // values 输出 buffer
        a_row: usize,           // M_max
        b_row: usize,           // N_max
        column: usize,          // K_max
        a_row_step_macro: usize,
        b_row_step_macro: usize,
        column_step_macro: usize,
        a_row_step_micro: usize,
        b_row_step_micro: usize,
        batch_max: usize,
        cpu_max_for_scratch: usize,
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

        // === (1) 构造期：B[K×N] → B_nt[N×K] ===
        let mut b_nt_vec: Vec<T> = vec![T::default(); n_max * k_max];
        let b_nt_ptr = b_nt_vec.as_mut_ptr();

        for kk in 0..k_max {
            let b_row_src = ptr2_b_kxn.add(kk * n_max); // 原 B[kk, :]
            for jj in 0..n_max {
                // B_nt[jj, kk] = B[kk, jj]
                *b_nt_ptr.add(jj * k_max + kk) = *b_row_src.add(jj);
            }
        }
        let b_nt_box = b_nt_vec.into_boxed_slice();
        let b_nt_base = b_nt_box.as_ptr();

        // === (2) 预分配 B 面板池：[cpu_max_for_scratch][kc×nr] ===
        let kc = params.column_step_macro.max(1);
        let nr = params.b_row_step_micro.max(1);
        let mr = params.a_row_step_micro.max(1);

        let b_panel_stride_elems = kc * nr;
        let b_panel_pool_len = cpu_max_for_scratch * b_panel_stride_elems;
        let b_panel_pool: Vec<T> = vec![T::default(); b_panel_pool_len];

        // === (3) 预分配 C_tile 池：[cpu_max_for_scratch][mr×nr] ===
        let c_tile_stride_elems = mr * nr;
        let c_tile_pool_len = cpu_max_for_scratch * c_tile_stride_elems;
        let c_tile_pool: Vec<T> = vec![T::default(); c_tile_pool_len];

        // === (4) 为每个 (batch, thread) 构造 FixedMinHeap，附着到输出 buffer 上 ===
        // layout: [batch_max][cpu_max_for_scratch][topk]
        let stride_thread = topk;
        let stride_batch = cpu_max_for_scratch * topk;

        let mut heaps_vec: Vec<FixedMinHeap<T>> =
            Vec::with_capacity(batch_max * cpu_max_for_scratch);

        for b in 0..batch_max {
            for tid in 0..cpu_max_for_scratch {
                let values_base = value_ptr.add(b * stride_batch + tid * stride_thread);
                let indices_base = indice_ptr.add(b * stride_batch + tid * stride_thread);

                heaps_vec.push(FixedMinHeap::new(values_base, indices_base, topk));
            }
        }

        Self {
            ptr1: ConstPtr { ptr: ptr1 },
            ptr2: ConstPtr { ptr: b_nt_base }, // 指向 B_nt
            indice_ptr: MutPtr { ptr: indice_ptr },
            value_ptr: MutPtr { ptr: value_ptr },

            a_row: m_max,
            b_row: n_max,
            column: k_max,

            params,
            topk,
            batch_max,
            cpu_max_for_scratch,

            b_nt_buf: b_nt_box,
            b_panel_pool: b_panel_pool.into_boxed_slice(),
            b_panel_stride_elems,
            c_tile_pool: c_tile_pool.into_boxed_slice(),
            c_tile_stride_elems,
            heaps: heaps_vec.into_boxed_slice(),

            _marker: PhantomData,
        }
    }

    /// 当前线程的 B_panel 指针（不分配）
    #[inline(always)]
    fn thread_b_panel_ptr(&self, thread_id: usize) -> *mut T {
        debug_assert!(thread_id < self.cpu_max_for_scratch);
        unsafe {
            self.b_panel_pool
                .as_ptr()
                .add(thread_id * self.b_panel_stride_elems) as *mut T
        }
    }

    /// 当前线程的 C_tile 指针（不分配）
    #[inline(always)]
    fn thread_c_tile_ptr(&self, thread_id: usize) -> *mut T {
        debug_assert!(thread_id < self.cpu_max_for_scratch);
        unsafe {
            self.c_tile_pool
                .as_ptr()
                .add(thread_id * self.c_tile_stride_elems) as *mut T
        }
    }

    /// 获取 (batch, thread) 对应的 heap 的可变裸指针
    #[inline(always)]
    fn heap_for(&self, batch: usize, thread_id: usize) -> *mut FixedMinHeap<T> {
        debug_assert!(batch < self.batch_max);
        debug_assert!(thread_id < self.cpu_max_for_scratch);
        let idx = batch * self.cpu_max_for_scratch + thread_id;
        debug_assert!(idx < self.heaps.len());
        unsafe { self.heaps.as_ptr().add(idx) as *mut FixedMinHeap<T> }
    }

    /// pack B_nt 的一个 (kc × nr) 小块到 panel：
    /// - B_nt: [N×K] 行主，行距 = ldb_row (=K)
    /// - 输入块范围：N 方向从 n0 开始，K 方向从 k0 开始
    /// - 输出：panel[kc×nr] 行主
    #[inline(always)]
    unsafe fn pack_b_panel(
        &self,
        b_nt: *const T,
        ldb_row: usize, // = K
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

    /// 运行：忽略 sequence 维度，只看 [batch_size, K] × [K, N]
    /// position_index / position_interval 先不用，你可以后面按需要扩展。
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
            assert!(cpu_num <= self.cpu_max_for_scratch);
            assert!(thread_id < cpu_num);

            let m = batch_size; // 行：batch
            let n = self.b_row; // 列：N_max (比如 vocab_size)
            let k = self.column; // K

            let mb = self.params.a_row_step_macro.max(1);
            let nb = self.params.b_row_step_macro.max(1);
            let kc = self.params.column_step_macro.max(1);
            let mr = self.params.a_row_step_micro.max(1); // 3
            let nr = self.params.b_row_step_micro.max(1); // 32

            debug_assert!(m % mr == 0);
            debug_assert!(n % nr == 0);
            debug_assert!(k % kc == 0);

            // 行距（元素计）
            let a_base = self.ptr1.ptr; // A[M×K]
            let lda = k; // A 每行跨度 = K

            let b_nt_ptr = self.ptr2.ptr; // B_nt[N×K]
            let ldb_row = k; // B_nt 每行跨度 = K

            // 输出 heap 对应的 layout stride
            let stride_thread = self.topk;
            let stride_batch = self.cpu_max_for_scratch * self.topk;

            // 当前线程的 panel / c_tile 缓冲区
            let b_panel_ptr = self.thread_b_panel_ptr(thread_id); // kc×nr
            let c_tile_ptr = self.thread_c_tile_ptr(thread_id); // mr×nr

            // run 之前，先清空本次要用到的 (batch, thread) 的 heap len
            for b in 0..batch_size {
                let heap_ptr = self.heap_for(b, thread_id);
                (*heap_ptr).clear();
            }

            // 2D tile 划分 (只在 M×N 平面上)
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

                    // 沿 M 方向按 mr（3）走，沿 N 方向按 nr（32）走
                    let mut mi = 0;
                    while mi < m_blk {
                        let global_m_start = m0 + mi;
                        let mr_here = mr.min(m_blk - mi);

                        let mut nt = 0;
                        while nt < n_blk {
                            let global_n_start = n0 + nt;
                            let nr_here = nr.min(n_blk - nt);

                            // === 为当前 (global_m_start..+mr, global_n_start..+nr) 的 micro-tile 清零 C_tile ===
                            let c_elems = mr * nr;
                            for i in 0..c_elems {
                                *c_tile_ptr.add(i) = T::default();
                            }

                            // === Kc 循环：对这个 micro-tile 累加 KC×NR 的块 ===
                            let mut k0 = 0;
                            while k0 < k {
                                // (1) pack B panel (kc×nr)
                                self.pack_b_panel(
                                    b_nt_ptr,
                                    ldb_row,
                                    global_n_start,
                                    k0,
                                    kc,
                                    nr_here,
                                    b_panel_ptr,
                                );

                                // (2) A 子块起点：从 global_m_start 开始的 mr 行
                                let a_tile = a_base.add(global_m_start * lda + k0);

                                // (3) 调用 3×32 微核：C_tile += A_tile × B_panel
                                self.compute(a_tile, b_panel_ptr as *const T, c_tile_ptr);

                                k0 += kc;
                            }

                            // === K 累加结束：把 C_tile 中的值丢进对应 batch 的 heap ===
                            for r in 0..mr_here {
                                let batch_idx = global_m_start + r;
                                if batch_idx >= batch_size {
                                    continue;
                                }

                                let heap_ptr = self.heap_for(batch_idx, thread_id);
                                let heap = &mut *heap_ptr;

                                for c in 0..nr_here {
                                    let col_idx = global_n_start + c;
                                    if col_idx >= n {
                                        continue;
                                    }
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

            // 所有 tile 结束后，对本线程负责的所有 batch 做一次降序排序
            for b in 0..batch_size {
                let heap_ptr = self.heap_for(b, thread_id);
                (*heap_ptr).sort_desc();
            }

            // 此时：
            // - value_ptr:  [batch_max][cpu_max_for_scratch][topk]
            // - indice_ptr: [batch_max][cpu_max_for_scratch][topk]
            // 中的 [0..batch_size][thread_id][0..topk] 就是本线程的 top-k 结果
        }
    }
}

/* ------------------ 微核 compute：保持你的调用风格 ------------------ */

impl<T> MatMulTopKTrait<T> for MatMulTopK<T>
where
    T: Copy + Default + PartialOrd + Add<Output = T> + Mul<Output = T>,
{
    default fn compute(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T) {
        // 通用版本：可以用 generic matmul_block，注意这里的 ldc 要等于 nr（局部 C_tile 的列数）
        let mr = self.params.a_row_step_micro.max(1);
        let nr = self.params.b_row_step_micro.max(1);

        let call_param = MatMulParams {
            // 这里 a_row_step_macro 作为 lda = K
            a_row_step_macro: self.column,
            // b_row_step_macro 作为 ldc = nr（C_tile 的列数）
            b_row_step_macro: nr,
            column_step_macro: self.params.column_step_macro, // kc
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
            a_row_step_macro: self.column,                    // lda = K
            b_row_step_macro: nr,                             // ldc = nr (C_tile 的列数)
            column_step_macro: self.params.column_step_macro, // kc
            a_row_step_micro: mr,                             // mr (=3)
            b_row_step_micro: nr,                             // nr (=32)
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
        kernel::generic::matmul_block::matmul_block(
            input_ptr1,
            input_ptr2,
            output_ptr,
            &call_param,
        );
    }
}

impl MatMulTopKTrait<f32> for MatMulTopK<f32> {
    fn compute(&self, input_ptr1: *const f32, input_ptr2: *const f32, output_ptr: *mut f32) {
        let mr = self.params.a_row_step_micro.max(1);
        let nr = self.params.b_row_step_micro.max(1);

        let call_param = MatMulParams {
            a_row_step_macro: self.column,                    // lda = K
            b_row_step_macro: nr,                             // ldc = nr (C_tile 的列数)
            column_step_macro: self.params.column_step_macro, // kc
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

    // 辅助验证函数
    fn verify_topk_result(
        m: usize,
        k: usize,
        n: usize,
        topk: usize,
        cpu_num: usize,
        cpu_max_for_scratch: usize,
        a: &[f16],
        b: &[f16],
        indices_buf: &[usize],
        values_buf: &[f16],
        epsilon: f32,
    ) {
        for i in 0..m {
            // (1) 计算参考结果 (Full MatMul Ground Truth)
            let mut row_c = vec![0.0f32; n];
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += (a[i * k + kk] as f32) * (b[kk * n + j] as f32);
                }
                row_c[j] = sum;
            }

            // (2) 排序找 Ground Truth TopK (降序)
            let mut indexed_row: Vec<(usize, f32)> = row_c.into_iter().enumerate().collect();
            indexed_row.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let expected_topk = &indexed_row[0..topk];

            // (3) 收集所有“线程”产生的局部 TopK，并进行 Reduce (Merge)
            let mut merged_candidates: Vec<(usize, f32)> = Vec::new();

            for tid in 0..cpu_num {
                // Offset = batch_idx * (cpu_max * topk) + thread_idx * topk
                let offset = i * (cpu_max_for_scratch * topk) + tid * topk;
                let result_indices = &indices_buf[offset..offset + topk];
                let result_values = &values_buf[offset..offset + topk];

                for k_idx in 0..topk {
                    merged_candidates.push((result_indices[k_idx], result_values[k_idx] as f32));
                }
            }

            // (4) 对合并后的结果再次排序取全局 TopK
            merged_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let final_topk = &merged_candidates[0..topk];

            // (5) 逐个比对
            for k_idx in 0..topk {
                let (exp_idx, exp_val) = expected_topk[k_idx];
                let (got_idx, got_val) = final_topk[k_idx];

                // 验证数值
                assert_abs_diff_eq!(got_val, exp_val, epsilon = epsilon);

                // 验证索引
                // 如果数值非常接近，索引可能因为浮点误差而不同
                // 这里我们要求：如果数值相等（或极接近），索引必须匹配
                if (got_val - exp_val).abs() < 1e-5 {
                    assert_eq!(got_idx, exp_idx, "Mismatch at batch {}, rank {}", i, k_idx);
                }
            }
        }
    }

    #[test]
    fn test_matmul_topk_f16_3x64x32() {
        // 1. 维度定义 (仿照 matmul.rs test_matmul_runner_f16_3x64x32)
        // M=3 (divisible by MR=3), K=64, N=32
        const M: usize = 3;
        const K: usize = 64;
        const N: usize = 32;
        const TOPK: usize = 10;

        // 2. 模拟多线程参数
        let cpu_num = 4;
        let cpu_max_for_scratch = 4;

        // 3. 数据准备
        let mut a = vec![0.0f16; M * K];
        let mut b = vec![0.0f16; K * N];

        // 初始化 A
        for i in 0..M {
            for k in 0..K {
                let val = (i + k) as f32 * 0.01;
                a[i * K + k] = val as f16;
            }
        }
        // 初始化 B
        for k in 0..K {
            for j in 0..N {
                let val = (k + j) as f32 * 0.001;
                b[k * N + j] = val as f16;
            }
        }

        // 4. 输出 Buffer
        let buf_len = M * cpu_max_for_scratch * TOPK;
        let mut indices_buf = vec![0usize; buf_len];
        let mut values_buf = vec![0.0f16; buf_len];

        // 5. 构造 Runner
        unsafe {
            let runner = MatMulTopK::<f16>::new(
                a.as_ptr(),
                b.as_ptr(),
                indices_buf.as_mut_ptr(),
                values_buf.as_mut_ptr(),
                M,  // a_row
                N,  // b_row
                K,  // column
                3,  // MB (Macro Block M)
                32, // NB (Macro Block N)
                64, // KC
                3,  // MR (必须整除 M)
                32, // NR
                M,  // batch_max
                cpu_max_for_scratch,
                TOPK,
            );

            // 6. Fake Multi-threading
            for tid in 0..cpu_num {
                runner.run(0, 0, M, cpu_num, tid);
            }
        }

        // 7. 验证结果
        verify_topk_result(
            M,
            K,
            N,
            TOPK,
            cpu_num,
            cpu_max_for_scratch,
            &a,
            &b,
            &indices_buf,
            &values_buf,
            0.005,
        );
    }

    #[test]
    fn test_matmul_topk_f16_144x2048x2048() {
        // 1. 维度定义 (仿照 matmul.rs test_matmul_runner_f16_144x2048x2048)
        const M: usize = 144;
        const K: usize = 2048;
        const N: usize = 2048;
        const TOPK: usize = 10;

        let cpu_num = 8;
        let cpu_max_for_scratch = 8;

        let mut a = vec![0.0f16; M * K];
        let mut b = vec![0.0f16; K * N];

        // 初始化 A, B
        for i in 0..M {
            for k in 0..K {
                let val = ((i + k) % 7) as f32 * 0.01;
                a[i * K + k] = val as f16;
            }
        }
        for k in 0..K {
            for j in 0..N {
                let val = ((k + j) % 11) as f32 * 0.01;
                b[k * N + j] = val as f16;
            }
        }

        let buf_len = M * cpu_max_for_scratch * TOPK;
        let mut indices_buf = vec![0usize; buf_len];
        let mut values_buf = vec![0.0f16; buf_len];

        unsafe {
            let runner = MatMulTopK::<f16>::new(
                a.as_ptr(),
                b.as_ptr(),
                indices_buf.as_mut_ptr(),
                values_buf.as_mut_ptr(),
                M,
                N,
                K,
                24,  // MB
                128, // NB
                64,  // KC
                3,   // MR
                32,  // NR
                M,
                cpu_max_for_scratch,
                TOPK,
            );

            for tid in 0..cpu_num {
                runner.run(0, 0, M, cpu_num, tid);
            }
        }

        // 验证结果 (K较大时累加误差较大，epsilon 放宽到 0.5)
        verify_topk_result(
            M,
            K,
            N,
            TOPK,
            cpu_num,
            cpu_max_for_scratch,
            &a,
            &b,
            &indices_buf,
            &values_buf,
            0.5,
        );
    }
}
