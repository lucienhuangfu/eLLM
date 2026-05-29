// === compiler/mul/matmul3.rs ===
#![allow(non_snake_case)]

use std::f16;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Sub};

use crate::kernel::common::matmul_params::MatMulParams;
use crate::num_traits::{FromNumber, Sqrt};
use crate::operators::send_sync_ptr::{ConstPtr, MutPtr};

use crate::runtime::sequence_slice::SequenceSlice;
use crate::operators::assign::{assign, KqvPath};
use crate::operators::traits::MatMulkqvTrait;

// Generic scalar helpers used by fallback paths.
// fallback 路径使用的通用标量 helper。
use crate::kernel::scalar::complex_mul::complex_mul;
use crate::kernel::scalar::rms_norm::{rms_norm, rms_norm_unit};

#[inline(always)]
fn rotate_half_rope<T>(head: *mut T, rope: *const T, length: usize)
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    let half = length / 2;
    let mut rotated = Vec::with_capacity(length);
    unsafe {
        for i in 0..half {
            let x1 = *head.add(i);
            let x2 = *head.add(i + half);
            let cos = *rope.add(2 * i);
            let sin = *rope.add(2 * i + 1);
            rotated.push(x1 * cos - x2 * sin);
        }
        for i in 0..half {
            let x1 = *head.add(i);
            let x2 = *head.add(i + half);
            let cos = *rope.add(2 * i);
            let sin = *rope.add(2 * i + 1);
            rotated.push(x2 * cos + x1 * sin);
        }
        head.copy_from_nonoverlapping(rotated.as_ptr(), length);
    }
}

// Variable naming used in this operator:
// - input_rows / input_row_start: token rows from hidden states.
// - query_output_cols: Q projection columns, equal to query_head_count * head_dim.
// - key_value_output_cols: K/V projection columns, equal to kv_head_num * head_dim.
// - reduction_cols / reduction_col_start: hidden-size K dimension reduced by GEMM.
// - input_block_rows / output_block_cols / reduction_block_cols: macro tile sizes.
// - micro_tile_rows / micro_tile_cols: micro-kernel tile size.
// - head_index / head_col: head-level offsets used by Q/K/V and RoPE.
// 本算子的变量命名约定：
// - input_rows / input_row_start：hidden states 的 token 行。
// - query_output_cols：Q 投影列数，等于 query_head_count * head_dim。
// - key_value_output_cols：K/V 投影列数，等于 kv_head_num * head_dim。
// - reduction_cols / reduction_col_start：GEMM 中被规约的 hidden-size K 维度。
// - input_block_rows / output_block_cols / reduction_block_cols：宏块大小。
// - micro_tile_rows / micro_tile_cols：微内核 tile 大小。
// - head_index / head_col：Q/K/V 和 RoPE 使用的 head 级偏移。

/// Three GEMMs for Q/K/V projection.
/// Q/K/V 三路投影 GEMM。
///
/// Layout contract:
/// 布局约定：
/// - A:      [M×K]
/// - Wq_nt:  [Nq×K]
/// - Wk_nt:  [Nkv×K]
/// - Wv_nt:  [Nkv×K]
/// - Cq:     [M×Nq]
/// - Ck:     [M×Nkv]
/// - Cv:     [M×Nkv]
///
/// Weights must already be NT layout; new() packs panels but does not transpose.
/// 权重必须已经是 NT 布局；new() 只 pack panel，不做转置。
#[derive(Clone)]
pub struct MatMul3<T> {
    // Input and Q/K/V output states.
    // 输入以及 Q/K/V 输出状态。
    hidden_ptr: ConstPtr<T>, // A[input_rows, reduction_cols]
    pub q_state_ptr: MutPtr<T>,  // Query state. query 输出。
    pub k_state_ptr: MutPtr<T>,  // Key cache/state. key cache/state。
    pub v_state_ptr: MutPtr<T>,  // Value cache/state. value cache/state。
    q_norm_weight: ConstPtr<T>,
    k_norm_weight: ConstPtr<T>,

    // RoPE table; caller guarantees layout compatibility.
    // RoPE 表；布局由外部保证一致。
    rope_ptr: ConstPtr<T>,
    sequence_length: usize,
    batch_size: usize,
    kv_head_num: usize,
    group_num: usize,
    head_dim: usize,
    use_qk_norm: bool,
    m_row: usize,
    col: usize,
    // b_q_row: usize,  // Nq
    // b_kv_row: usize, // Nkv

    // Blocking parameters.
    // 分块参数。
    pub params: MatMulParams,
    _marker: PhantomData<T>,

    // Packed Q/K/V weight panels prepared in new().
    // Q/K/V 权重 panel 在 new() 中提前准备。
    packed_q: Box<[T]>,
    packed_k: Box<[T]>,
    packed_v: Box<[T]>,
    packed_panel_stride: usize,
}

impl<T> MatMul3<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Sub<Output = T> + Default + Sqrt + FromNumber,
{
    #[inline]
    pub fn new(
        hidden_ptr: *const T,
        q_weight_ptr_nt: *const T, // Wq_nt[query_cols, reduction_cols]
        q_state_ptr: *mut T,
        k_weight_ptr_nt: *const T, // Wk_nt[key_value_cols, reduction_cols]
        k_state_ptr: *mut T,
        v_weight_ptr_nt: *const T, // Wv_nt[key_value_cols, reduction_cols]
        v_state_ptr: *mut T,
        q_norm_weight: *const T,
        k_norm_weight: *const T,
        rope_ptr: *const T,
        sequence_length: usize,
        batch_size: usize,
        // GQA dimensions.
        // GQA 维度信息。
        kv_head_num: usize,
        group_num: usize,
        head_dim: usize,
        use_qk_norm: bool,
        m_row: usize,
        col: usize,
        // b_q_row: usize,
        // b_kv_row: usize,
        a_row_step_macro: usize,
        b_row_step_macro: usize,
        column_step_macro: usize,
        a_row_step_micro: usize,
        b_row_step_micro: usize,
    ) -> Self {
        let reduction_block_cols = column_step_macro.max(1);
        let micro_tile_cols = b_row_step_micro.max(1);
        let packed_panel_stride = reduction_block_cols * micro_tile_cols;
        let packed_q = Self::pack_b_panels(
            q_weight_ptr_nt,
            kv_head_num * group_num * head_dim,
            col,
            reduction_block_cols,
            micro_tile_cols,
        );
        let packed_k = Self::pack_b_panels(
            k_weight_ptr_nt,
            kv_head_num * head_dim,
            col,
            reduction_block_cols,
            micro_tile_cols,
        );
        let packed_v = Self::pack_b_panels(
            v_weight_ptr_nt,
            kv_head_num * head_dim,
            col,
            reduction_block_cols,
            micro_tile_cols,
        );

        Self {
            hidden_ptr: ConstPtr { ptr: hidden_ptr },
            q_state_ptr: MutPtr { ptr: q_state_ptr },
            k_state_ptr: MutPtr { ptr: k_state_ptr },
            v_state_ptr: MutPtr { ptr: v_state_ptr },
            q_norm_weight: ConstPtr { ptr: q_norm_weight },
            k_norm_weight: ConstPtr { ptr: k_norm_weight },

            rope_ptr: ConstPtr { ptr: rope_ptr },
            sequence_length,
            batch_size,
            kv_head_num,
            group_num,
            head_dim,
            use_qk_norm,
            m_row,
            col,
            params: MatMulParams {
                a_row_step_macro,
                b_row_step_macro,
                column_step_macro,
                a_row_step_micro,
                b_row_step_micro,
            },
            _marker: PhantomData,

            packed_q,
            packed_k,
            packed_v,
            packed_panel_stride,
        }
    }

    #[inline(always)]
    pub fn panel_threads(&self) -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
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
    fn packed_panel_ptr(
        &self,
        packed_b: &[T],
        output_cols: usize,
        output_col_start: usize,
        reduction_col_start: usize,
    ) -> *const T {
        let reduction_block_cols = self.params.column_step_macro.max(1);
        let micro_tile_cols = self.params.b_row_step_micro.max(1);
        let output_panel_count = output_cols.div_ceil(micro_tile_cols);
        let panel_index = (reduction_col_start / reduction_block_cols) * output_panel_count
            + (output_col_start / micro_tile_cols);
        unsafe {
            packed_b
                .as_ptr()
                .add(panel_index * self.packed_panel_stride)
        }
    }

    /// Run tiled GEMM for one Q/K/V path and optionally apply RMSNorm+RoPE.
    /// 对 Q/K/V 中一路执行 tiled GEMM，并可选执行 RMSNorm+RoPE。
    #[inline(always)]
    unsafe fn gemm_one_path_tiles(
        &self,
        input_base: *const T,
        output_base: *mut T,
        packed_weight: &[T],
        input_rows: usize,
        output_cols: usize,
        reduction_cols: usize,
        output_row_stride: usize,
        rope_base: *const T,
        tile_begin: usize,
        tile_end: usize,
        finalize: bool,
    ) where
        Self: MatMulkqvTrait<T>,
    {
        let input_block_rows = self.params.a_row_step_macro;
        let output_block_cols = self.params.b_row_step_macro;
        let reduction_block_cols = self.params.column_step_macro;
        let micro_tile_rows = self.params.a_row_step_micro;
        let micro_tile_cols = self.params.b_row_step_micro;

        debug_assert_eq!(micro_tile_rows, 3);
        debug_assert_eq!(micro_tile_cols, 32);
        debug_assert!(reduction_cols % reduction_block_cols == 0);
        debug_assert!(output_cols % micro_tile_cols == 0);
        debug_assert!(self.head_dim % micro_tile_cols == 0);

        let input_tile_count = input_rows.div_ceil(input_block_rows);
        let output_tile_count = output_cols.div_ceil(output_block_cols);
        let total_tiles = input_tile_count * output_tile_count;

        let head_dim = self.head_dim;

        debug_assert!(tile_begin <= tile_end);
        debug_assert!(tile_end <= total_tiles);

        for task_id in tile_begin..tile_end {
            let input_tile_id = task_id / output_tile_count;
            let output_tile_id = task_id % output_tile_count;

            let input_row_start = input_tile_id * input_block_rows;
            let output_col_start = output_tile_id * output_block_cols;

            let input_rows_in_block = (input_rows - input_row_start).min(input_block_rows);
            let output_cols_in_block = (output_cols - output_col_start).min(output_block_cols);

            debug_assert!(input_rows_in_block % micro_tile_rows == 0);
            debug_assert!(output_cols_in_block % micro_tile_cols == 0);

            let mut reduction_col_start = 0;
            while reduction_col_start < reduction_cols {
                let reduction_cols_this =
                    reduction_block_cols.min(reduction_cols - reduction_col_start);

                let mut output_col_offset = 0;
                while output_col_offset < output_cols_in_block {
                    let weight_panel_ptr = self.packed_panel_ptr(
                        packed_weight,
                        output_cols,
                        output_col_start + output_col_offset,
                        reduction_col_start,
                    );

                    let mut input_row_offset = 0;
                    while input_row_offset < input_rows_in_block {
                        let input_tile = input_base.add(
                            (input_row_start + input_row_offset) * reduction_cols
                                + reduction_col_start,
                        );
                        let output_tile = output_base.add(
                            (input_row_start + input_row_offset) * output_row_stride
                                + (output_col_start + output_col_offset),
                        );

                        self.compute1(
                            input_tile,
                            weight_panel_ptr,
                            output_tile,
                            reduction_cols,
                            output_row_stride,
                            reduction_cols_this,
                        );

                        if finalize && (reduction_col_start + reduction_cols_this == reduction_cols)
                        {
                            let global_col = output_col_start + output_col_offset;
                            let offset_in_head = global_col % head_dim;

                            if offset_in_head + micro_tile_cols == head_dim {
                                let head_col0 = global_col - offset_in_head;

                                let c_head_ptr = output_base.add(
                                    (input_row_start + input_row_offset) * output_row_stride
                                        + head_col0,
                                );
                                let rope_head_ptr = rope_base.add(head_col0);

                                self.compute2(c_head_ptr, rope_head_ptr, output_row_stride);
                            }
                        }

                        input_row_offset += micro_tile_rows;
                    }

                    output_col_offset += micro_tile_cols;
                }
                reduction_col_start += reduction_cols_this;
            }
        }
    }

    #[inline(always)]
    fn task_assign_path(
        v_tiles: usize,
        k_tiles: usize,
        q_tiles: usize,
        task_id: usize,
    ) -> Option<(KqvPath, usize)> {
        let total = v_tiles + k_tiles + q_tiles;
        if task_id >= total {
            return None;
        }
        if task_id < v_tiles {
            return Some((KqvPath::V, task_id));
        }
        let task_id = task_id - v_tiles;
        if task_id < k_tiles {
            return Some((KqvPath::K, task_id));
        }
        Some((KqvPath::Q, task_id - k_tiles))
    }

    #[inline(always)]
    fn build_row_map(
        &self,
        prefill_size: usize,
        decode_size: usize,
        attention_list: &[SequenceSlice],
    ) -> Vec<(usize, usize, usize)> {
        if attention_list.is_empty() {
            let fallback_len = prefill_size.max(decode_size).min(self.m_row);
            return (0..fallback_len)
                .map(|row| (row, 0usize, row.min(self.sequence_length.saturating_sub(1))))
                .collect();
        }

        let mut rows = Vec::with_capacity(attention_list.iter().map(|slice| slice.length).sum());
        for slice in attention_list {
            if slice.batch_index >= self.batch_size {
                continue;
            }

            for offset in 0..slice.length {
                let token_index = slice.token_start_index + offset;
                let sequence_index = slice.sequence_index + offset;
                if token_index >= self.m_row || sequence_index >= self.sequence_length {
                    continue;
                }
                rows.push((token_index, slice.batch_index, sequence_index));
            }
        }
        rows
    }

    #[inline(always)]
    unsafe fn compute_head_from_packed(
        &self,
        a_row: *const T,
        dst_head: *mut T,
        packed_b: &[T],
        n_total: usize,
        head_index: usize,
        apply_rope: bool,
        norm_weight: *const T,
        sequence_index: usize,
    ) where
        Self: MatMulkqvTrait<T>,
    {
        let reduction_cols = self.col;
        let reduction_block_cols = self.params.column_step_macro.max(1);
        let micro_tile_cols = self.params.b_row_step_micro.max(1);
        debug_assert_eq!(micro_tile_cols, 32);
        debug_assert_eq!(self.head_dim % micro_tile_cols, 0);

        for head_col in 0..self.head_dim {
            *dst_head.add(head_col) = T::default();
        }

        let head_col0 = head_index * self.head_dim;
        let head_output_panel0 = head_col0 / micro_tile_cols;
        let output_panel_count = n_total.div_ceil(micro_tile_cols);
        self.compute_head_gemv(
            a_row,
            dst_head,
            packed_b.as_ptr(),
            head_output_panel0,
            output_panel_count,
            reduction_cols,
            reduction_block_cols,
            micro_tile_cols,
            self.head_dim,
        );

        if apply_rope {
            let eps = T::from_f32(1e-6);
            let rope_ptr = self.rope_ptr.ptr.add(sequence_index * self.head_dim);
            if self.use_qk_norm {
                self.compute_norm_rope(dst_head, norm_weight, rope_ptr, self.head_dim, eps);
            } else {
                rotate_half_rope(dst_head, rope_ptr, self.head_dim);
            }
        }
    }

    #[inline(always)]
    fn path_task(
        row_count: usize,
        kv_head_num: usize,
        q_head_num: usize,
        task_id: usize,
    ) -> Option<(KqvPath, usize, usize)> {
        let v_tasks = row_count * kv_head_num;
        let k_tasks = v_tasks;
        let q_tasks = row_count * q_head_num;
        let total = v_tasks + k_tasks + q_tasks;
        if task_id >= total {
            return None;
        }

        if task_id < v_tasks {
            return Some((KqvPath::V, task_id / kv_head_num, task_id % kv_head_num));
        }

        let task_id = task_id - v_tasks;
        if task_id < k_tasks {
            return Some((KqvPath::K, task_id / kv_head_num, task_id % kv_head_num));
        }

        let task_id = task_id - k_tasks;
        Some((KqvPath::Q, task_id / q_head_num, task_id % q_head_num))
    }

    /// 入口：不再有 S 维度，只针对当前 A[M×K] 做一次 K/Q/V。
    pub fn run(
        &self,
        prefill_size: usize,
        decode_size: usize,
        attention_list: &[SequenceSlice],
        thread_num: usize,
        thread_id: usize,
    ) where
        Self: MatMulkqvTrait<T>,
    {
        unsafe {
            let reduction_cols = self.col;
            let query_output_cols = self.kv_head_num * self.group_num * self.head_dim;
            let key_value_output_cols = self.kv_head_num * self.head_dim;
            let query_head_count = self.kv_head_num * self.group_num;
            let row_map = self.build_row_map(prefill_size, decode_size, attention_list);
            let row_count = row_map.len();
            if row_count == 0 || thread_id >= thread_num || thread_num == 0 {
                return;
            }

            let a_base = self.hidden_ptr.ptr;
            let cq_base = self.q_state_ptr.ptr;
            let ck_base = self.k_state_ptr.ptr;
            let cv_base = self.v_state_ptr.ptr;

            let query_row_stride = query_output_cols;
            let key_row_stride = key_value_output_cols;
            let value_row_stride = key_value_output_cols;
            let total_tasks = row_count * (self.kv_head_num * 2 + query_head_count);
            if let Some((task_begin, task_end)) = assign(total_tasks, thread_num, thread_id) {
                for task_id in task_begin..task_end {
                    let Some((path, row_idx, head_index)) =
                        Self::path_task(row_count, self.kv_head_num, query_head_count, task_id)
                    else {
                        continue;
                    };

                    let (token_index, batch_index, sequence_index) = row_map[row_idx];
                    let input_row = a_base.add(token_index * reduction_cols);

                    let rope_sequence_index = if attention_list.is_empty() {
                        0
                    } else {
                        sequence_index
                    };

                    match path {
                        KqvPath::V => {
                            let cache_row =
                                (sequence_index * self.batch_size + batch_index) * value_row_stride;
                            let dst_head = cv_base.add(cache_row + head_index * self.head_dim);
                            self.compute_head_from_packed(
                                input_row,
                                dst_head,
                                &self.packed_v,
                                key_value_output_cols,
                                head_index,
                                false,
                                self.k_norm_weight.ptr,
                                rope_sequence_index,
                            );
                        }
                        KqvPath::K => {
                            let cache_row =
                                (sequence_index * self.batch_size + batch_index) * key_row_stride;
                            let dst_head = ck_base.add(cache_row + head_index * self.head_dim);
                            self.compute_head_from_packed(
                                input_row,
                                dst_head,
                                &self.packed_k,
                                key_value_output_cols,
                                head_index,
                                true,
                                self.k_norm_weight.ptr,
                                rope_sequence_index,
                            );
                        }
                        KqvPath::Q => {
                            let dst_head = cq_base
                                .add(token_index * query_row_stride + head_index * self.head_dim);
                            self.compute_head_from_packed(
                                input_row,
                                dst_head,
                                &self.packed_q,
                                query_output_cols,
                                head_index,
                                true,
                                self.q_norm_weight.ptr,
                                rope_sequence_index,
                            );
                        }
                    }
                }
            }
        }
    }
}

impl<T> MatMulkqvTrait<T> for MatMul3<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Sub<Output = T> + Default + Sqrt,
{
    #[inline]
    default fn compute1(
        &self,
        _a: *const T,
        _b_panel: *const T,
        _c: *mut T,
        _lda: usize,
        _ldc: usize,
        _kc: usize,
    ) {
        // generic 占位
    }

    #[inline]
    default fn compute2(&self, _c_head: *mut T, _rope_head: *const T, _ldc: usize) {
        // generic 占位
    }

    #[inline]
    default fn compute_norm_rope(
        &self,
        c_head: *mut T,
        norm_weight: *const T,
        rope_head: *const T,
        length: usize,
        eps: T,
    ) {
        rms_norm(c_head, norm_weight, c_head, length, eps);
        rotate_half_rope(c_head, rope_head, length);
    }

    #[inline]
    default fn compute_head_gemv(
        &self,
        a_row: *const T,
        dst_head: *mut T,
        packed_b: *const T,
        head_output_panel: usize,
        output_panel_count: usize,
        reduction_cols: usize,
        reduction_block_cols: usize,
        micro_tile_cols: usize,
        head_dim: usize,
    ) {
        unsafe {
            for head_col in (0..head_dim).step_by(micro_tile_cols) {
                let output_panel = head_output_panel + head_col / micro_tile_cols;
                let mut reduction_col_start = 0usize;
                while reduction_col_start < reduction_cols {
                    let reduction_cols_this =
                        reduction_block_cols.min(reduction_cols - reduction_col_start);
                    let reduction_panel = reduction_col_start / reduction_block_cols;
                    let weight_panel = packed_b.add(
                        (reduction_panel * output_panel_count + output_panel)
                            * reduction_block_cols
                            * micro_tile_cols,
                    );
                    for reduction_lane in 0..reduction_cols_this {
                        let input_value = *a_row.add(reduction_col_start + reduction_lane);
                        for output_lane in 0..micro_tile_cols {
                            let dst = dst_head.add(head_col + output_lane);
                            *dst = *dst
                                + input_value
                                    * *weight_panel
                                        .add(reduction_lane * micro_tile_cols + output_lane);
                        }
                    }
                    reduction_col_start += reduction_cols_this;
                }
            }
        }
    }
}

// f16 specialization: call the AVX-512 micro-kernel.
// f16 特化：调用 AVX-512 微内核。
impl MatMulkqvTrait<f16> for MatMul3<f16> {
    #[inline]
    fn compute1(
        &self,
        a: *const f16,
        b_panel: *const f16,
        c: *mut f16,
        lda: usize,
        ldc: usize,
        kc: usize,
    ) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            crate::kernel::x86_64::f16_512::matmul_rms_complex::matmul_update_inplace_3x32_accum(
                a, b_panel, c, lda, ldc, kc,
            );
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        unsafe {
            let nr = self.params.b_row_step_micro;
            for m in 0..self.params.a_row_step_micro {
                for n in 0..nr {
                    let mut sum = *c.add(m * ldc + n) as f32;
                    for k in 0..kc {
                        sum += (*a.add(m * lda + k) as f32) * (*b_panel.add(k * nr + n) as f32);
                    }
                    *c.add(m * ldc + n) = sum as f16;
                }
            }
        }
    }

    #[inline]
    fn compute2(&self, c_head: *mut f16, rope_head: *const f16, ldc: usize) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            let eps: f16 = 1e-6f32 as f16;
            crate::kernel::x86_64::f16_512::matmul_rms_complex::matmul_finalize_rmsnorm_rope_inplace_3x128(
                c_head,
                rope_head,
                ldc,
                eps,
            );
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        unsafe {
            let eps = 1e-6f32 as f16;
            for r in 0..self.params.a_row_step_micro {
                let row_ptr = c_head.add(r * ldc);
                rms_norm_unit(row_ptr, row_ptr, self.head_dim, eps);
                complex_mul(row_ptr, rope_head, row_ptr, self.head_dim);
            }
        }
    }

    #[inline]
    fn compute_norm_rope(
        &self,
        c_head: *mut f16,
        norm_weight: *const f16,
        rope_head: *const f16,
        length: usize,
        eps: f16,
    ) {
        crate::kernel::x86_64::f16_512::rms_norm::rms_norm(
            c_head,
            norm_weight,
            c_head,
            length,
            eps,
        );
        rotate_half_rope(c_head, rope_head, length);
    }

    #[inline]
    fn compute_head_gemv(
        &self,
        a_row: *const f16,
        dst_head: *mut f16,
        packed_b: *const f16,
        head_output_panel: usize,
        output_panel_count: usize,
        reduction_cols: usize,
        reduction_block_cols: usize,
        micro_tile_cols: usize,
        head_dim: usize,
    ) {
        unsafe {
            for head_col in (0..head_dim).step_by(micro_tile_cols) {
                let output_panel = head_output_panel + head_col / micro_tile_cols;
                let mut acc = [0.0f32; 32];
                let mut reduction_col_start = 0usize;
                while reduction_col_start < reduction_cols {
                    let reduction_cols_this =
                        reduction_block_cols.min(reduction_cols - reduction_col_start);
                    let reduction_panel = reduction_col_start / reduction_block_cols;
                    let weight_panel = packed_b.add(
                        (reduction_panel * output_panel_count + output_panel)
                            * reduction_block_cols
                            * micro_tile_cols,
                    );
                    for reduction_lane in 0..reduction_cols_this {
                        let input_value = *a_row.add(reduction_col_start + reduction_lane) as f32;
                        for output_lane in 0..micro_tile_cols {
                            acc[output_lane] += input_value
                                * (*weight_panel.add(reduction_lane * micro_tile_cols + output_lane)
                                    as f32);
                        }
                    }
                    reduction_col_start += reduction_cols_this;
                }

                for output_lane in 0..micro_tile_cols {
                    *dst_head.add(head_col + output_lane) = acc[output_lane] as f16;
                }
            }
        }
    }
}

// f32 fallback implementation; a dedicated micro-kernel can be added later.
// f32 fallback 实现；后续需要时可以补专用微内核。
impl MatMulkqvTrait<f32> for MatMul3<f32> {
    #[inline]
    fn compute1(
        &self,
        a: *const f32,
        b_panel: *const f32,
        c: *mut f32,
        lda: usize,
        ldc: usize,
        kc: usize,
    ) {
        unsafe {
            for m in 0..3 {
                for n in 0..32 {
                    let mut sum = 0.0;
                    for k in 0..kc {
                        let val_a = *a.add(m * lda + k);
                        let val_b = *b_panel.add(k * 32 + n);
                        sum += val_a * val_b;
                    }
                    *c.add(m * ldc + n) += sum;
                }
            }
        }
    }

    #[inline]
    fn compute2(&self, c_head: *mut f32, rope_head: *const f32, ldc: usize) {
        unsafe {
            let eps = 1e-6;
            for r in 0..3 {
                let row_ptr = c_head.add(r * ldc);
                rms_norm_unit(row_ptr, row_ptr, 128, eps);
                complex_mul(row_ptr, rope_head, row_ptr, 128);
            }
        }
    }

    #[inline]
    fn compute_norm_rope(
        &self,
        c_head: *mut f32,
        norm_weight: *const f32,
        rope_head: *const f32,
        length: usize,
        eps: f32,
    ) {
        rms_norm(c_head, norm_weight, c_head, length, eps);
        rotate_half_rope(c_head, rope_head, length);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ========================================================================
    // Helpers for f32 tests
    // ========================================================================

    /// 参考：A[M×K] * W[K×N]，但我们存的是 W_nt[N×K]，所以用 w_nt[j*K + p]
    fn ref_matmul_f32_from_wnt(
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        w_nt: &[f32], // N×K
        c: &mut [f32],
    ) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a[i * k + p] * w_nt[j * k + p];
                }
                c[i * n + j] = sum;
            }
        }
    }

    fn ref_post_process_f32(m: usize, n: usize, c: &mut [f32], rope: &[f32], head_dim: usize) {
        let eps = 1e-6;
        unsafe {
            for i in 0..m {
                for h_base in (0..n).step_by(head_dim) {
                    let ptr = c.as_mut_ptr().add(i * n + h_base);
                    let rope_ptr = rope.as_ptr().add(h_base);
                    rms_norm_unit(ptr, ptr, head_dim, eps);
                    complex_mul(ptr, rope_ptr, ptr, head_dim);
                }
            }
        }
    }

    // ========================================================================
    // Helpers for f16 tests
    // ========================================================================

    fn avail_threads_cap(cap: usize) -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(cap)
            .max(1)
    }

    /// f32 accumulate reference GEMM: out = A[M×K] * W[K×N]，但 W 存的是 W_nt[N×K]
    fn gemm_ref_f16_acc_f32_from_wnt(
        a: &[f16],
        w_nt: &[f16], // N×K
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += (a[i * k + kk] as f32) * (w_nt[j * k + kk] as f32);
                }
                out[i * n + j] = sum;
            }
        }
    }

    fn ref_post_process_f16_acc_f32(
        m: usize,
        n: usize,
        c: &mut [f32],
        rope: &[f16],
        head_dim: usize,
    ) {
        for i in 0..m {
            for h_base in (0..n).step_by(head_dim) {
                let head = &mut c[i * n + h_base..i * n + h_base + head_dim];
                let sum_sq: f32 = head.iter().map(|v| v * v).sum();
                let rrms = 1.0f32 / (sum_sq / head_dim as f32 + 1e-6).sqrt();
                for v in head.iter_mut() {
                    *v *= rrms;
                }
                for j in (0..head_dim).step_by(2) {
                    let a = head[j];
                    let b = head[j + 1];
                    let c = rope[h_base + j] as f32;
                    let d = rope[h_base + j + 1] as f32;
                    head[j] = a * c - b * d;
                    head[j + 1] = a * d + b * c;
                }
            }
        }
    }

    fn run_runner(runner: &MatMul3<f16>, m: usize, thread_num: usize) {
        for tid in 0..thread_num {
            runner.run(m, 0, &[], thread_num, tid);
        }
    }

    // ========================================================================
    // Tests
    // ========================================================================

    #[test]
    fn test_matmul3_qkv_f32_72_rows() {
        let m = 72;
        let k = 256;
        let sequence_length = 1024;
        let head_dim = 128;
        let n_q = 32 * 128; // 4096
        let n_kv = 4 * 128; // 512

        // A
        let mut a = vec![0.0f32; m * k];

        // ✅ 权重改为 N×K（W_nt）
        let mut wq_nt = vec![0.0f32; n_q * k];
        let mut wk_nt = vec![0.0f32; n_kv * k];
        let mut wv_nt = vec![0.0f32; n_kv * k];

        let mut cq = vec![0.0f32; m * n_q];
        let mut cq_ref = vec![0.0f32; m * n_q];

        let mut ck = vec![0.0f32; m * n_kv];
        let mut ck_ref = vec![0.0f32; m * n_kv];

        let mut cv = vec![0.0f32; m * n_kv];
        let mut cv_ref = vec![0.0f32; m * n_kv];

        let q_norm = vec![1.0f32; head_dim];
        let k_norm = vec![1.0f32; head_dim];
        let mut rope = vec![1.0f32; n_q.max(n_kv)];

        for i in 0..m * k {
            a[i] = (i % 100) as f32 * 0.01;
        }

        // 原先是 K×N 的填法，这里改成 N×K
        for j in 0..n_q {
            for kk in 0..k {
                let idx_old = kk * n_q + j;
                let v = ((idx_old + 1) % 7) as f32 * 0.01;
                wq_nt[j * k + kk] = v;
            }
        }
        for j in 0..n_kv {
            for kk in 0..k {
                let idx_old = kk * n_kv + j;
                wk_nt[j * k + kk] = ((idx_old + 2) % 7) as f32 * 0.01;
                wv_nt[j * k + kk] = ((idx_old + 3) % 7) as f32 * 0.01;
            }
        }
        for i in 0..rope.len() {
            rope[i] = 1.0;
        }

        unsafe {
            let matmul = MatMul3::<f32>::new(
                a.as_ptr(),
                wq_nt.as_ptr(),
                cq.as_mut_ptr(),
                wk_nt.as_ptr(),
                ck.as_mut_ptr(),
                wv_nt.as_ptr(),
                cv.as_mut_ptr(),
                q_norm.as_ptr(),
                k_norm.as_ptr(),
                rope.as_ptr(),
                sequence_length, // sequence_length
                1,               // batch_size
                n_kv / head_dim, // kv_head_num
                n_q / n_kv,      // group_num
                head_dim,        // head_dim
                true,            // use_qk_norm
                m,               // m_row
                k,               // col
                24,              // a_row_step_macro
                128,             // b_row_step_macro
                32,              // column_step_macro
                3,               // a_row_step_micro
                32,              // b_row_step_micro
            );

            matmul.run(m, 0, &[], 1, 0);

            // reference（从 W_nt 计算）
            ref_matmul_f32_from_wnt(m, k, n_q, &a, &wq_nt, &mut cq_ref);
            ref_post_process_f32(m, n_q, &mut cq_ref, &rope, head_dim);

            ref_matmul_f32_from_wnt(m, k, n_kv, &a, &wk_nt, &mut ck_ref);
            ref_post_process_f32(m, n_kv, &mut ck_ref, &rope, head_dim);

            ref_matmul_f32_from_wnt(m, k, n_kv, &a, &wv_nt, &mut cv_ref);

            let verify = |name: &str, out: &[f32], reference: &[f32]| {
                let mut max_diff = 0.0f32;
                for (i, (v1, v2)) in out.iter().zip(reference.iter()).enumerate() {
                    let diff = (v1 - v2).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                    if diff > 1e-3 {
                        panic!(
                            "{} mismatch at index {}: got {}, expected {}, diff {}",
                            name, i, v1, v2, diff
                        );
                    }
                }
                println!("{} passed. Max diff: {}", name, max_diff);
            };

            verify("Q Output", &cq, &cq_ref);
            verify("K Output", &ck, &ck_ref);
            verify("V Output", &cv, &cv_ref);
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_kqv_f16_avx512_small_single_tile() {
        const M: usize = 3;
        const K: usize = 64;
        const NQ: usize = 128;
        const NKV: usize = 128;
        const HEAD_DIM: usize = 128;

        let thread_num = avail_threads_cap(8);

        let mut a = vec![0.0f16; M * K];

        // ✅ 权重改为 N×K
        let mut wq_nt = vec![0.0f16; NQ * K];
        let mut wk_nt = vec![0.0f16; NKV * K];
        let mut wv_nt = vec![0.0f16; NKV * K];

        let mut cq = vec![0.0f16; M * NQ];
        let mut ck = vec![0.0f16; M * NKV];
        let mut cv = vec![0.0f16; M * NKV];

        let q_norm = vec![1.0f16; HEAD_DIM];
        let k_norm = vec![1.0f16; HEAD_DIM];
        let mut rope = vec![0.0f16; NQ.max(NKV)];
        for i in (0..rope.len()).step_by(2) {
            rope[i] = 1.0f16;
        }

        for i in 0..M {
            for kk in 0..K {
                a[i * K + kk] = (0.01f32 * (i as f32) + 0.001f32 * (kk as f32)) as f16;
            }
        }

        // 原来是 K×N 的写法，这里转成 N×K
        for j in 0..NQ {
            for kk in 0..K {
                wq_nt[j * K + kk] = (0.02f32 * (kk as f32) + 0.003f32 * (j as f32)) as f16;
            }
        }
        for j in 0..NKV {
            for kk in 0..K {
                wk_nt[j * K + kk] = (0.015f32 * (kk as f32) + 0.002f32 * (j as f32)) as f16;
                wv_nt[j * K + kk] = (0.017f32 * (kk as f32) + 0.0025f32 * (j as f32)) as f16;
            }
        }

        let runner = MatMul3::<f16>::new(
            a.as_ptr(),
            wq_nt.as_ptr(),
            cq.as_mut_ptr(),
            wk_nt.as_ptr(),
            ck.as_mut_ptr(),
            wv_nt.as_ptr(),
            cv.as_mut_ptr(),
            q_norm.as_ptr(),
            k_norm.as_ptr(),
            rope.as_ptr(),
            HEAD_DIM,       // sequence_length
            1,              // batch_size
            NKV / HEAD_DIM, // kv_head_num
            NQ / NKV,       // group_num
            HEAD_DIM,       // head_dim
            true,           // use_qk_norm
            M,              // m_row
            K,              // col
            3,              // a_row_step_macro
            32,             // b_row_step_macro
            64,             // column_step_macro
            3,              // a_row_step_micro
            32,             // b_row_step_micro
        );

        run_runner(&runner, M, thread_num);

        let mut cq_ref = vec![0.0f32; M * NQ];
        let mut ck_ref = vec![0.0f32; M * NKV];
        let mut cv_ref = vec![0.0f32; M * NKV];
        gemm_ref_f16_acc_f32_from_wnt(&a, &wq_nt, &mut cq_ref, M, K, NQ);
        gemm_ref_f16_acc_f32_from_wnt(&a, &wk_nt, &mut ck_ref, M, K, NKV);
        gemm_ref_f16_acc_f32_from_wnt(&a, &wv_nt, &mut cv_ref, M, K, NKV);
        ref_post_process_f16_acc_f32(M, NQ, &mut cq_ref, &rope, HEAD_DIM);
        ref_post_process_f16_acc_f32(M, NKV, &mut ck_ref, &rope, HEAD_DIM);

        for i in 0..M {
            for j in 0..NQ {
                assert_abs_diff_eq!(cq[i * NQ + j] as f32, cq_ref[i * NQ + j], epsilon = 1e-1);
            }
            for j in 0..NKV {
                assert_abs_diff_eq!(ck[i * NKV + j] as f32, ck_ref[i * NKV + j], epsilon = 1e-1);
                assert_abs_diff_eq!(cv[i * NKV + j] as f32, cv_ref[i * NKV + j], epsilon = 1e-1);
            }
        }
    }

    // 下面这些 f16 测试原本都以 K×N 构造权重并 reference 也是 K×N；
    // 现在统一改为 N×K（W_nt），因此全部按同样方式改造：
    //   1) wq/wk/wv 的 Vec 长度从 K*N 变为 N*K
    //   2) 填充索引从 [kk*N + j] 变为 [j*K + kk]
    //   3) reference 从 b[kk*N + j] 变为 w_nt[j*K + kk]
    //
    // 为了避免篇幅爆炸，这里保留你原测试结构，并做同样的 layout 改动。

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_kqv_f16_avx512_multi_tile() {
        const M: usize = 12;
        const K: usize = 64;
        const NQ: usize = 128;
        const NKV: usize = 128;
        const HEAD_DIM: usize = 128;

        let thread_num = avail_threads_cap(16);

        let mut a = vec![0.0f16; M * K];
        let mut wq_nt = vec![0.0f16; NQ * K];
        let mut wk_nt = vec![0.0f16; NKV * K];
        let mut wv_nt = vec![0.0f16; NKV * K];

        let mut cq = vec![0.0f16; M * NQ];
        let mut ck = vec![0.0f16; M * NKV];
        let mut cv = vec![0.0f16; M * NKV];

        let q_norm = vec![1.0f16; HEAD_DIM];
        let k_norm = vec![1.0f16; HEAD_DIM];
        let mut rope = vec![0.0f16; NQ.max(NKV)];
        for i in (0..rope.len()).step_by(2) {
            rope[i] = 1.0f16;
        }

        for i in 0..M {
            for kk in 0..K {
                a[i * K + kk] = (((i * 7 + kk * 3) % 19) as f32 * 0.01f32) as f16;
            }
        }
        for j in 0..NQ {
            for kk in 0..K {
                wq_nt[j * K + kk] = (((kk * 5 + j * 11) % 23) as f32 * 0.01f32) as f16;
            }
        }
        for j in 0..NKV {
            for kk in 0..K {
                wk_nt[j * K + kk] = (((kk * 3 + j * 7) % 29) as f32 * 0.01f32) as f16;
                wv_nt[j * K + kk] = (((kk * 9 + j * 4) % 31) as f32 * 0.01f32) as f16;
            }
        }

        let runner = MatMul3::<f16>::new(
            a.as_ptr(),
            wq_nt.as_ptr(),
            cq.as_mut_ptr(),
            wk_nt.as_ptr(),
            ck.as_mut_ptr(),
            wv_nt.as_ptr(),
            cv.as_mut_ptr(),
            q_norm.as_ptr(),
            k_norm.as_ptr(),
            rope.as_ptr(),
            HEAD_DIM,       // sequence_length
            1,              // batch_size
            NKV / HEAD_DIM, // kv_head_num
            NQ / NKV,       // group_num
            HEAD_DIM,       // head_dim
            true,           // use_qk_norm
            M,              // m_row
            K,              // col
            6,              // a_row_step_macro
            64,             // b_row_step_macro
            64,             // column_step_macro
            3,              // a_row_step_micro
            32,             // b_row_step_micro
        );

        run_runner(&runner, M, thread_num);

        let mut cq_ref = vec![0.0f32; M * NQ];
        let mut ck_ref = vec![0.0f32; M * NKV];
        let mut cv_ref = vec![0.0f32; M * NKV];
        gemm_ref_f16_acc_f32_from_wnt(&a, &wq_nt, &mut cq_ref, M, K, NQ);
        gemm_ref_f16_acc_f32_from_wnt(&a, &wk_nt, &mut ck_ref, M, K, NKV);
        gemm_ref_f16_acc_f32_from_wnt(&a, &wv_nt, &mut cv_ref, M, K, NKV);
        ref_post_process_f16_acc_f32(M, NQ, &mut cq_ref, &rope, HEAD_DIM);
        ref_post_process_f16_acc_f32(M, NKV, &mut ck_ref, &rope, HEAD_DIM);

        for i in 0..M {
            for j in 0..NQ {
                assert_abs_diff_eq!(cq[i * NQ + j] as f32, cq_ref[i * NQ + j], epsilon = 5e-1);
            }
            for j in 0..NKV {
                assert_abs_diff_eq!(ck[i * NKV + j] as f32, ck_ref[i * NKV + j], epsilon = 5e-1);
                assert_abs_diff_eq!(cv[i * NKV + j] as f32, cv_ref[i * NKV + j], epsilon = 5e-1);
            }
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_kqv_f16_avx512_kc_split() {
        const M: usize = 3;
        const K: usize = 128;
        const NQ: usize = 128;
        const NKV: usize = 128;
        const HEAD_DIM: usize = 128;

        let thread_num = avail_threads_cap(8);

        let mut a = vec![0.0f16; M * K];
        let mut wq_nt = vec![0.0f16; NQ * K];
        let mut wk_nt = vec![0.0f16; NKV * K];
        let mut wv_nt = vec![0.0f16; NKV * K];

        let mut cq = vec![0.0f16; M * NQ];
        let mut ck = vec![0.0f16; M * NKV];
        let mut cv = vec![0.0f16; M * NKV];

        let q_norm = vec![1.0f16; HEAD_DIM];
        let k_norm = vec![1.0f16; HEAD_DIM];
        let mut rope = vec![0.0f16; NQ.max(NKV)];
        for i in (0..rope.len()).step_by(2) {
            rope[i] = 1.0f16;
        }

        for i in 0..M {
            for kk in 0..K {
                a[i * K + kk] = (((i + kk) % 17) as f32 * 0.01f32) as f16;
            }
        }
        for j in 0..NQ {
            for kk in 0..K {
                wq_nt[j * K + kk] = (((kk * 2 + j) % 13) as f32 * 0.01f32) as f16;
            }
        }
        for j in 0..NKV {
            for kk in 0..K {
                wk_nt[j * K + kk] = (((kk * 3 + j * 2) % 19) as f32 * 0.01f32) as f16;
                wv_nt[j * K + kk] = (((kk * 5 + j * 3) % 23) as f32 * 0.01f32) as f16;
            }
        }

        let runner = MatMul3::<f16>::new(
            a.as_ptr(),
            wq_nt.as_ptr(),
            cq.as_mut_ptr(),
            wk_nt.as_ptr(),
            ck.as_mut_ptr(),
            wv_nt.as_ptr(),
            cv.as_mut_ptr(),
            q_norm.as_ptr(),
            k_norm.as_ptr(),
            rope.as_ptr(),
            HEAD_DIM,       // sequence_length
            1,              // batch_size
            NKV / HEAD_DIM, // kv_head_num
            NQ / NKV,       // group_num
            HEAD_DIM,       // head_dim
            true,           // use_qk_norm
            M,              // m_row
            K,              // col
            3,              // a_row_step_macro
            32,             // b_row_step_macro
            64,             // column_step_macro
            3,              // a_row_step_micro
            32,             // b_row_step_micro
        );

        run_runner(&runner, M, thread_num);

        let mut cq_ref = vec![0.0f32; M * NQ];
        let mut ck_ref = vec![0.0f32; M * NKV];
        let mut cv_ref = vec![0.0f32; M * NKV];
        gemm_ref_f16_acc_f32_from_wnt(&a, &wq_nt, &mut cq_ref, M, K, NQ);
        gemm_ref_f16_acc_f32_from_wnt(&a, &wk_nt, &mut ck_ref, M, K, NKV);
        gemm_ref_f16_acc_f32_from_wnt(&a, &wv_nt, &mut cv_ref, M, K, NKV);
        ref_post_process_f16_acc_f32(M, NQ, &mut cq_ref, &rope, HEAD_DIM);
        ref_post_process_f16_acc_f32(M, NKV, &mut ck_ref, &rope, HEAD_DIM);

        for i in 0..M {
            for j in 0..NQ {
                assert_abs_diff_eq!(cq[i * NQ + j] as f32, cq_ref[i * NQ + j], epsilon = 5e-1);
            }
            for j in 0..NKV {
                assert_abs_diff_eq!(ck[i * NKV + j] as f32, ck_ref[i * NKV + j], epsilon = 5e-1);
                assert_abs_diff_eq!(cv[i * NKV + j] as f32, cv_ref[i * NKV + j], epsilon = 5e-1);
            }
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_kqv_f16_avx512_medium() {
        const M: usize = 48;
        const K: usize = 256;
        const NQ: usize = 256;
        const NKV: usize = 256;
        const HEAD_DIM: usize = 128;

        let thread_num = avail_threads_cap(16);

        let mut a = vec![0.0f16; M * K];
        let mut wq_nt = vec![0.0f16; NQ * K];
        let mut wk_nt = vec![0.0f16; NKV * K];
        let mut wv_nt = vec![0.0f16; NKV * K];

        let mut cq = vec![0.0f16; M * NQ];
        let mut ck = vec![0.0f16; M * NKV];
        let mut cv = vec![0.0f16; M * NKV];

        let q_norm = vec![1.0f16; HEAD_DIM];
        let k_norm = vec![1.0f16; HEAD_DIM];
        let mut rope = vec![0.0f16; NQ.max(NKV)];
        for i in (0..rope.len()).step_by(2) {
            rope[i] = 1.0f16;
        }

        for i in 0..M {
            for kk in 0..K {
                a[i * K + kk] = (((i * 3 + kk * 5) % 97) as f32 * 0.001) as f16;
            }
        }
        for j in 0..NQ {
            for kk in 0..K {
                wq_nt[j * K + kk] = (((kk * 7 + j * 11) % 101) as f32 * 0.001) as f16;
            }
        }
        for j in 0..NKV {
            for kk in 0..K {
                wk_nt[j * K + kk] = (((kk * 13 + j * 17) % 103) as f32 * 0.001) as f16;
                wv_nt[j * K + kk] = (((kk * 19 + j * 23) % 107) as f32 * 0.001) as f16;
            }
        }

        let runner = MatMul3::<f16>::new(
            a.as_ptr(),
            wq_nt.as_ptr(),
            cq.as_mut_ptr(),
            wk_nt.as_ptr(),
            ck.as_mut_ptr(),
            wv_nt.as_ptr(),
            cv.as_mut_ptr(),
            q_norm.as_ptr(),
            k_norm.as_ptr(),
            rope.as_ptr(),
            HEAD_DIM,       // sequence_length
            1,              // batch_size
            NKV / HEAD_DIM, // kv_head_num
            NQ / NKV,       // group_num
            HEAD_DIM,       // head_dim
            true,           // use_qk_norm
            M,              // m_row
            K,              // col
            24,             // a_row_step_macro
            128,            // b_row_step_macro
            64,             // column_step_macro
            3,              // a_row_step_micro
            32,             // b_row_step_micro
        );

        run_runner(&runner, M, thread_num);

        let mut cq_ref = vec![0.0f32; M * NQ];
        let mut ck_ref = vec![0.0f32; M * NKV];
        let mut cv_ref = vec![0.0f32; M * NKV];
        gemm_ref_f16_acc_f32_from_wnt(&a, &wq_nt, &mut cq_ref, M, K, NQ);
        gemm_ref_f16_acc_f32_from_wnt(&a, &wk_nt, &mut ck_ref, M, K, NKV);
        gemm_ref_f16_acc_f32_from_wnt(&a, &wv_nt, &mut cv_ref, M, K, NKV);
        ref_post_process_f16_acc_f32(M, NQ, &mut cq_ref, &rope, HEAD_DIM);
        ref_post_process_f16_acc_f32(M, NKV, &mut ck_ref, &rope, HEAD_DIM);

        for i in 0..M {
            for j in 0..NQ {
                assert_abs_diff_eq!(cq[i * NQ + j] as f32, cq_ref[i * NQ + j], epsilon = 1.0);
            }
            for j in 0..NKV {
                assert_abs_diff_eq!(ck[i * NKV + j] as f32, ck_ref[i * NKV + j], epsilon = 1.0);
                assert_abs_diff_eq!(cv[i * NKV + j] as f32, cv_ref[i * NKV + j], epsilon = 1.0);
            }
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_kqv_f16_avx512_batch7_pad_to9_no_finalize() {
        use approx::assert_abs_diff_eq;

        const M_RUN: usize = 7; // 非 3 倍数
        const M_MAX: usize = 9; // ceil_div(7,3)*3 = 9
        const K: usize = 64;

        const HEAD_DIM: usize = 128;
        const NQ: usize = 128;
        const NKV: usize = 128;

        let thread_num = avail_threads_cap(8);

        // A 按 M_MAX 分配（capacity），只填前 M_RUN 行，其余保持 0
        let mut a = vec![0.0f16; M_MAX * K];
        for i in 0..M_RUN {
            for kk in 0..K {
                a[i * K + kk] = (0.01f32 * (i as f32) + 0.001f32 * (kk as f32)) as f16;
            }
        }

        // W_nt: N×K
        let mut wq_nt = vec![0.0f16; NQ * K];
        let mut wk_nt = vec![0.0f16; NKV * K];
        let mut wv_nt = vec![0.0f16; NKV * K];

        for j in 0..NQ {
            for kk in 0..K {
                wq_nt[j * K + kk] = (0.02f32 * (kk as f32) + 0.003f32 * (j as f32)) as f16;
            }
        }
        for j in 0..NKV {
            for kk in 0..K {
                wk_nt[j * K + kk] = (0.015f32 * (kk as f32) + 0.002f32 * (j as f32)) as f16;
                wv_nt[j * K + kk] = (0.017f32 * (kk as f32) + 0.0025f32 * (j as f32)) as f16;
            }
        }

        // 输出按 M_MAX 分配（capacity）
        let mut cq = vec![0.0f16; M_MAX * NQ];
        let mut ck = vec![0.0f16; M_MAX * NKV];
        let mut cv = vec![0.0f16; M_MAX * NKV];

        let q_norm = vec![1.0f16; HEAD_DIM];
        let k_norm = vec![1.0f16; HEAD_DIM];
        let mut rope = vec![0.0f16; HEAD_DIM.max(NQ).max(NKV)];
        for i in (0..rope.len()).step_by(2) {
            rope[i] = 1.0f16;
        }

        // MB 要是 MR 的倍数（你现有 gemm_one_path_tiles 要求 m_blk%mr==0）
        let runner = MatMul3::<f16>::new(
            a.as_ptr(),
            wq_nt.as_ptr(),
            cq.as_mut_ptr(),
            wk_nt.as_ptr(),
            ck.as_mut_ptr(),
            wv_nt.as_ptr(),
            cv.as_mut_ptr(),
            q_norm.as_ptr(),
            k_norm.as_ptr(),
            rope.as_ptr(),
            HEAD_DIM,       // sequence_length
            1,              // batch_size
            NKV / HEAD_DIM, // kv_head_num
            NQ / NKV,       // group_num
            HEAD_DIM,       // head_dim
            true,           // use_qk_norm
            M_MAX,          // m_row 当 capacity 用
            K,              // col
            6,              // a_row_step_macro
            64,             // b_row_step_macro
            64,             // column_step_macro
            3,              // a_row_step_micro
            32,             // b_row_step_micro
        );

        // run 传 batch=7（内部 pad 到 9）
        run_runner(&runner, M_RUN, thread_num);

        // reference：只算前 7 行（pad 行不关心）
        let mut cq_ref = vec![0.0f32; M_RUN * NQ];
        let mut ck_ref = vec![0.0f32; M_RUN * NKV];
        let mut cv_ref = vec![0.0f32; M_RUN * NKV];

        gemm_ref_f16_acc_f32_from_wnt(&a, &wq_nt, &mut cq_ref, M_RUN, K, NQ);
        gemm_ref_f16_acc_f32_from_wnt(&a, &wk_nt, &mut ck_ref, M_RUN, K, NKV);
        gemm_ref_f16_acc_f32_from_wnt(&a, &wv_nt, &mut cv_ref, M_RUN, K, NKV);
        ref_post_process_f16_acc_f32(M_RUN, NQ, &mut cq_ref, &rope, HEAD_DIM);
        ref_post_process_f16_acc_f32(M_RUN, NKV, &mut ck_ref, &rope, HEAD_DIM);

        for i in 0..M_RUN {
            for j in 0..NQ {
                assert_abs_diff_eq!(cq[i * NQ + j] as f32, cq_ref[i * NQ + j], epsilon = 5e-1);
            }
            for j in 0..NKV {
                assert_abs_diff_eq!(ck[i * NKV + j] as f32, ck_ref[i * NKV + j], epsilon = 5e-1);
                assert_abs_diff_eq!(cv[i * NKV + j] as f32, cv_ref[i * NKV + j], epsilon = 5e-1);
            }
        }
    }
}
