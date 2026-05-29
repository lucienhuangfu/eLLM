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

// Variable naming used in this operator:
// - input_rows / input_row_start: rows from the input matrix A and batch rows.
// - output_cols / output_col_start: candidate columns scored for top-k.
// - reduction_cols / reduction_col_start: the K dimension reduced by GEMM.
// - input_block_rows / output_block_cols / reduction_block_cols: macro tile sizes.
// - micro_tile_rows / micro_tile_cols: micro-kernel tile size.
// - batch_row: a real, non-padded input row whose heap receives top-k candidates.
// 本算子的变量命名约定：
// - input_rows / input_row_start：输入矩阵 A 的行维度，也对应 batch 行。
// - output_cols / output_col_start：参与 top-k 打分的候选列。
// - reduction_cols / reduction_col_start：GEMM 中被规约的 K 维度。
// - input_block_rows / output_block_cols / reduction_block_cols：宏块大小。
// - micro_tile_rows / micro_tile_cols：微内核 tile 大小。
// - batch_row：真实的非 padding 输入行，对应一个 top-k heap。

#[derive(Clone)]
pub struct MatMulTopK<T> {
    // Input, weight, and per-thread top-k output buffers.
    // 输入、权重以及每线程 top-k 输出缓存。
    ptr1: ConstPtr<T>,         // A[input_rows, reduction_cols]
    ptr2: ConstPtr<T>,         // B_nt[output_cols, reduction_cols]
    indice_ptr: MutPtr<usize>, // Top-k index buffer. top-k 下标缓存：[batch_max][thread_max][TOPK]
    value_ptr: MutPtr<T>,      // Top-k value buffer. top-k 分数缓存：[batch_max][thread_max][TOPK]

    // Maximum dimensions.
    // 最大维度。
    a_row: usize,  // Maximum input rows. 最大输入行数，M_max。
    b_row: usize,  // Maximum output columns. 最大输出列数，N_max。
    column: usize, // Maximum reduction columns. 最大规约列数，K_max。

    pub params: MatMulParams,

    topk: usize,
    batch_max: usize,

    // Internal thread capacity used by scratch tiles and heaps.
    // 内部线程容量，用于绑定 scratch tile 和 heap。
    thread_max: usize,

    // Weight panels packed once in new().
    // 权重 panel 在 new() 中提前 pack。
    packed_b: Box<[T]>,
    packed_panel_stride: usize,

    // Per-thread output micro tile pool.
    // 每线程一份输出微内核 tile 缓存。
    c_tile_pool: Box<[T]>,
    c_tile_stride_elems: usize,

    // One heap for each (batch, thread) pair.
    // 每个 (batch, thread) 对应一棵 heap。
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
        ptr1: *const T,          // A[input_rows, reduction_cols]
        ptr2_b_nt_nxk: *const T, // B_nt[output_cols, reduction_cols]
        indice_ptr: *mut usize,  // indices output buffer. 下标输出缓存。
        value_ptr: *mut T,       // values output buffer. 分数输出缓存。
        a_row: usize,            // maximum input rows. 最大输入行数。
        b_row: usize,            // maximum output columns. 最大输出列数。
        column: usize,           // maximum reduction columns. 最大规约列数。
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

        let micro_tile_rows_tmp = a_row_step_micro.max(1);
        let m_max = a_row.div_ceil(micro_tile_rows_tmp) * micro_tile_rows_tmp;
        let n_max = b_row;
        let k_max = column;

        // Detect thread capacity once; run() only validates against it.
        // 只在 new() 中确定线程容量；run() 中只做校验。
        let thread_max = Self::detect_threads();

        let reduction_block_cols = params.column_step_macro.max(1);
        let micro_tile_cols = b_row_step_micro.max(1);
        let micro_tile_rows = micro_tile_rows_tmp;

        let packed_panel_stride = reduction_block_cols * micro_tile_cols;
        let packed_b = Self::pack_b_panels(
            ptr2_b_nt_nxk,
            n_max,
            k_max,
            reduction_block_cols,
            micro_tile_cols,
        );

        let c_tile_stride_elems = micro_tile_rows * micro_tile_cols;
        let c_tile_pool_len = thread_max * c_tile_stride_elems;
        let c_tile_pool: Vec<T> = vec![T::default(); c_tile_pool_len];

        // Bind heaps directly to caller-provided output buffers.
        // heap 直接绑定到外部传入的输出 buffer。
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

    /// Return this thread's heap for one batch row.
    /// 返回当前线程在某个 batch row 上的 heap。
    #[inline(always)]
    fn heap_for(&self, batch: usize, thread_id: usize) -> *mut FixedMinHeap<T> {
        debug_assert!(batch < self.batch_max);
        debug_assert!(thread_id < self.thread_max);
        let idx = batch * self.thread_max + thread_id;
        debug_assert!(idx < self.heaps.len());
        unsafe { self.heaps.as_ptr().add(idx) as *mut FixedMinHeap<T> }
    }

    #[inline(always)]
    fn pack_b_panels(
        weight_nt: *const T,
        output_cols: usize,
        reduction_cols: usize,
        reduction_block_cols: usize,
        micro_tile_cols: usize,
    ) -> Box<[T]> {
        let reduction_panel_count = reduction_cols.div_ceil(reduction_block_cols);
        let output_panel_count = output_cols.div_ceil(micro_tile_cols);
        let panel_stride = reduction_block_cols * micro_tile_cols;
        let mut packed =
            vec![T::default(); reduction_panel_count * output_panel_count * panel_stride];

        unsafe {
            for reduction_panel_index in 0..reduction_panel_count {
                let reduction_start = reduction_panel_index * reduction_block_cols;
                let reduction_cols_this =
                    (reduction_cols - reduction_start).min(reduction_block_cols);
                for output_panel_index in 0..output_panel_count {
                    let output_start = output_panel_index * micro_tile_cols;
                    let output_cols_this = (output_cols - output_start).min(micro_tile_cols);
                    let panel = packed.as_mut_ptr().add(
                        (reduction_panel_index * output_panel_count + output_panel_index)
                            * panel_stride,
                    );
                    for reduction_lane in 0..reduction_cols_this {
                        let packed_row = panel.add(reduction_lane * micro_tile_cols);
                        for output_lane in 0..output_cols_this {
                            *packed_row.add(output_lane) = *weight_nt.add(
                                (output_start + output_lane) * reduction_cols
                                    + (reduction_start + reduction_lane),
                            );
                        }
                    }
                }
            }
        }

        packed.into_boxed_slice()
    }

    #[inline(always)]
    fn packed_panel_ptr(&self, output_col_start: usize, reduction_col_start: usize) -> *const T {
        let reduction_block_cols = self.params.column_step_macro.max(1);
        let micro_tile_cols = self.params.b_row_step_micro.max(1);
        let output_panel_count = self.b_row.div_ceil(micro_tile_cols);
        let panel_index = (reduction_col_start / reduction_block_cols) * output_panel_count
            + (output_col_start / micro_tile_cols);
        unsafe {
            self.packed_b
                .as_ptr()
                .add(panel_index * self.packed_panel_stride)
        }
    }

    pub fn run(
        &self,
        prefill_size: usize,
        decode_size: usize,
        thread_num: usize,
        thread_id: usize,
    ) {
        unsafe {
            let active_input_rows = if prefill_size > 0 {
                prefill_size
            } else {
                decode_size
            };

            assert!(active_input_rows <= self.batch_max);

            assert!(thread_num <= self.thread_max);
            assert!(thread_id < thread_num);

            let output_cols = self.b_row;
            let reduction_cols = self.column;

            let input_block_rows = self.params.a_row_step_macro.max(1);
            let output_block_cols = self.params.b_row_step_macro.max(1);
            let reduction_block_cols = self.params.column_step_macro.max(1);
            let micro_tile_rows = self.params.a_row_step_micro.max(1);
            let micro_tile_cols = self.params.b_row_step_micro.max(1);

            // Pad rows so the fixed micro-kernel never sees a partial input tile.
            // 补齐行数，避免固定微内核处理输入行 tail。
            let padded_input_rows = active_input_rows.div_ceil(micro_tile_rows) * micro_tile_rows;

            debug_assert!(padded_input_rows <= self.a_row);

            debug_assert!(input_block_rows % micro_tile_rows == 0);

            debug_assert!(output_cols % micro_tile_cols == 0);
            debug_assert!(reduction_cols % reduction_block_cols == 0);

            let input_base = self.ptr1.ptr;
            let input_row_stride = reduction_cols;
            let c_tile_ptr = self.thread_c_tile_ptr(thread_id);

            // Clear heaps only for real batch rows, not padded rows.
            // 只清真实 batch 行的 heap，不处理 padding 行。
            for batch_row in 0..active_input_rows {
                let heap_ptr = self.heap_for(batch_row, thread_id);
                (*heap_ptr).clear();
            }

            let input_tile_count = padded_input_rows.div_ceil(input_block_rows);
            let output_tile_count = output_cols.div_ceil(output_block_cols);
            let total_tiles = input_tile_count * output_tile_count;

            if let Some((task_begin, task_end)) = assign(total_tiles, thread_num, thread_id) {
                for task_id in task_begin..task_end {
                    let input_tile_id = task_id / output_tile_count;
                    let output_tile_id = task_id % output_tile_count;

                    let input_row_start = input_tile_id * input_block_rows;
                    let output_col_start = output_tile_id * output_block_cols;

                    let input_rows_in_block =
                        (padded_input_rows - input_row_start).min(input_block_rows);
                    let output_cols_in_block =
                        (output_cols - output_col_start).min(output_block_cols);

                    debug_assert!(input_rows_in_block % micro_tile_rows == 0);
                    debug_assert!(output_cols_in_block % micro_tile_cols == 0);

                    let mut input_row_offset = 0usize;
                    while input_row_offset < input_rows_in_block {
                        let global_input_row_start = input_row_start + input_row_offset;

                        let mut output_col_offset = 0usize;
                        while output_col_offset < output_cols_in_block {
                            let global_output_col_start = output_col_start + output_col_offset;

                            for tile_index in 0..(micro_tile_rows * micro_tile_cols) {
                                *c_tile_ptr.add(tile_index) = T::default();
                            }

                            let mut reduction_col_start = 0usize;
                            while reduction_col_start < reduction_cols {
                                let input_tile = input_base.add(
                                    global_input_row_start * input_row_stride + reduction_col_start,
                                );
                                let weight_panel_ptr = self
                                    .packed_panel_ptr(global_output_col_start, reduction_col_start);

                                self.compute(input_tile, weight_panel_ptr, c_tile_ptr);

                                reduction_col_start += reduction_block_cols;
                            }

                            // Push only real rows into top-k heaps; padded rows are ignored.
                            // 只把真实行写入 top-k heap，忽略 padding 行。
                            for row_in_tile in 0..micro_tile_rows {
                                let batch_row = global_input_row_start + row_in_tile;
                                if batch_row >= active_input_rows {
                                    continue;
                                }
                                let heap_ptr = self.heap_for(batch_row, thread_id);
                                let heap = &mut *heap_ptr;

                                for col_in_tile in 0..micro_tile_cols {
                                    let output_col = global_output_col_start + col_in_tile;
                                    let value = *c_tile_ptr
                                        .add(row_in_tile * micro_tile_cols + col_in_tile);
                                    heap.push(value, output_col);
                                }
                            }

                            output_col_offset += micro_tile_cols;
                        }

                        input_row_offset += micro_tile_rows;
                    }
                }
            }

            // Sort only real batch rows.
            // 只排序真实 batch 行。
            for batch_row in 0..active_input_rows {
                let heap_ptr = self.heap_for(batch_row, thread_id);
                (*heap_ptr).sort_desc();
            }
        }
    }

    #[inline]
    pub fn thread_max(&self) -> usize {
        self.thread_max
    }
}

/* ------------------ compute micro-kernel entry ------------------ */
/* ------------------ compute 微内核入口 ------------------ */

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
