// === operators/expert/shared_expert/shared_expert_matmul_mul.rs ===
#![allow(non_snake_case)]

use crate::kernel::common::matmul_params::MatMulParams;
use crate::operators::assign::assign;
use crate::operators::expert::expert_routing::{task_assign, ExpertRouting, ExpertTaskMeta};
use crate::operators::send_sync_ptr::{ConstPtr, MutPtr};
use crate::operators::traits::SharedExpertsDownTrait;
use std::marker::PhantomData;
use std::ops::{Add, Mul};
use std::sync::atomic::Ordering;

// Variable naming used in this operator:
// - token_block_rows / token_block_start: routed-token macro block inside one expert (sparse
//   branch) or a dense token macro block (shared branch).
// - output_cols / output_col_start: down-projection output hidden H columns.
// - reduction_cols / reduction_col_start: intermediate Hmid (routed) / Is (shared) reduced by GEMM.
// - micro_tile_rows / micro_tile_cols: micro-kernel tile size.
// - routed_token_begin / token_offset_in_block: positions in the compact expert queue.
// - topk_slot: token-major output slot for one expert route.
// 本算子的变量命名约定：
// - token_block_rows / token_block_start：sparse 分支单个 expert 内的 routed token 宏块，
//   或 shared 分支的稠密 token 宏块。
// - output_cols / output_col_start：down projection 输出 hidden H 列。
// - reduction_cols / reduction_col_start：GEMM 中被规约的 intermediate Hmid（routed）/ Is（shared）。
// - micro_tile_rows / micro_tile_cols：微内核 tile 大小。
// - routed_token_begin / token_offset_in_block：compact expert queue 中的位置。
// - topk_slot：某个 expert route 在 token-major 输出中的 slot。

/// Shared expert + routed experts down projection.
/// shared expert 与 routed experts 的 down 投影。
///
///   shared: SHARED_NONLIN[b, Is] × W_sdown[Is, H] → SHARED_OUT[b, H]      (dense, no score)
///   routed: NONLIN[e, b, Hmid]   × W_down[e,Hmid,H] → OUT[b, slot(b,e), H] (score-scaled scatter)
///
/// run() executes the shared phase first, then the routed phase.
/// run() 先执行 shared 阶段，再执行 routed 阶段。
///
/// Compute is plain scalar Rust; no f16 / AVX-512 specialization yet.
/// compute 为普通标量 Rust；暂不做 f16 / AVX-512 特化。
#[derive(Clone)]
pub struct SharedExpertMatMulDown<T> {
    pub nonlin_ptr: ConstPtr<T>,        // Routed nonlinear input: [E,B,Hmid].
    pub wdown_nt_ptr: ConstPtr<T>,      // Routed down weight NT: [E,H,Hmid].
    pub shared_nonlin_ptr: ConstPtr<T>, // Shared nonlinear input: [B,Is].
    pub shared_wdown_nt_ptr: ConstPtr<T>, // Shared down weight NT: [H,Is].

    pub routing: ExpertRouting<T>,

    pub output_ptr: MutPtr<T>, // Routed token-major output: [B,Ktop,H].
    pub shared_output_ptr: MutPtr<T>, // Shared output: [B,H].

    pub num_experts: usize,
    pub num_token: usize,    // Token capacity.
    pub hmid: usize,         // Routed intermediate hidden size.
    pub shared_inter: usize, // Shared intermediate hidden size (Is).
    pub h: usize,            // Output hidden size.
    pub num_topk: usize,
    pub decode_only_flag: bool,

    pub params: MatMulParams,
    _marker: PhantomData<T>,

    // ---- prepacked weights (packed once in new()) ----
    packed_wdown: Box<[T]>,
    packed_shared_wdown: Box<[T]>,
    packed_panel_stride: usize, // reduction_block_cols * micro_tile_cols

    // Input tile: micro_rows × reduction_block, row-major.
    a_tile_pool: Box<[T]>,
    a_tile_stride: usize, // micro_tile_rows * reduction_block_cols

    // Accumulator tile: micro_rows × micro_cols, row-major.
    acc_pool: Box<[T]>,
    acc_stride: usize, // micro_tile_rows * micro_tile_cols

    // Routed token index buffer, one slice per thread.
    idx_buf_pool: Box<[usize]>,
    idx_stride: usize, // token_block_rows

    // Task-space buffers, one slice per thread.
    task_meta_pool: Box<[ExpertTaskMeta]>,
    task_meta_stride: usize, // num_experts
    routed_tokens_pool: Box<[usize]>,
    routed_slots_pool: Box<[usize]>,
    routed_scores_pool: Box<[T]>,
    routed_stride: usize, // num_experts * capacity_per_expert
}

impl<T> SharedExpertMatMulDown<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    #[inline]
    fn detect_threads() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(16)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        nonlin_ptr: *const T,          // Routed nonlinear input: [E,B,Hmid].
        wdown_nt_ptr: *const T,        // Routed down weight NT: [E,H,Hmid].
        shared_nonlin_ptr: *const T,   // Shared nonlinear input: [B,Is].
        shared_wdown_nt_ptr: *const T, // Shared down weight NT: [H,Is].
        routing: ExpertRouting<T>,
        output_ptr: *mut T,        // Routed token-major output: [B,Ktop,H].
        shared_output_ptr: *mut T, // Shared output: [B,H].
        num_experts: usize,
        num_token: usize,
        hmid: usize,
        shared_inter: usize,
        h: usize,
        num_topk: usize,
        params: MatMulParams,
        decode_only_flag: bool,
    ) -> Self {
        let token_block_rows = params.a_row_step_macro.max(1);
        let reduction_block_cols = params.column_step_macro.max(1);
        let micro_tile_rows = params.a_row_step_micro.max(1);
        let micro_tile_cols = params.b_row_step_micro.max(1);

        let packed_panel_stride = reduction_block_cols * micro_tile_cols;
        let packed_wdown = Self::pack_expert_b_panels(
            wdown_nt_ptr,
            num_experts,
            h,
            hmid,
            reduction_block_cols,
            micro_tile_cols,
        );
        // Shared down = a single dense "expert" (expert_count = 1).
        // shared down 视作单个稠密 "expert"（expert_count = 1）。
        let packed_shared_wdown = Self::pack_expert_b_panels(
            shared_wdown_nt_ptr,
            1,
            h,
            shared_inter,
            reduction_block_cols,
            micro_tile_cols,
        );

        let threads = Self::detect_threads();

        let a_tile_stride = micro_tile_rows * reduction_block_cols;
        let acc_stride = micro_tile_rows * micro_tile_cols;
        let idx_stride = token_block_rows;

        let a_tile_pool = vec![T::default(); threads * a_tile_stride].into_boxed_slice();
        let acc_pool = vec![T::default(); threads * acc_stride].into_boxed_slice();
        let idx_buf_pool = vec![0usize; threads * idx_stride].into_boxed_slice();
        let task_meta_stride = num_experts;
        let routed_stride = num_experts * routing.capacity_per_expert;
        let task_meta_pool =
            vec![ExpertTaskMeta::default(); threads * task_meta_stride].into_boxed_slice();
        let routed_tokens_pool = vec![0usize; threads * routed_stride].into_boxed_slice();
        let routed_slots_pool = vec![0usize; threads * routed_stride].into_boxed_slice();
        let routed_scores_pool = vec![T::default(); threads * routed_stride].into_boxed_slice();

        Self {
            nonlin_ptr: ConstPtr { ptr: nonlin_ptr },
            wdown_nt_ptr: ConstPtr { ptr: wdown_nt_ptr },
            shared_nonlin_ptr: ConstPtr {
                ptr: shared_nonlin_ptr,
            },
            shared_wdown_nt_ptr: ConstPtr {
                ptr: shared_wdown_nt_ptr,
            },

            routing,

            output_ptr: MutPtr { ptr: output_ptr },
            shared_output_ptr: MutPtr {
                ptr: shared_output_ptr,
            },

            num_experts,
            num_token,
            hmid,
            shared_inter,
            h,
            num_topk,
            decode_only_flag,

            params,
            _marker: PhantomData,

            packed_wdown,
            packed_shared_wdown,
            packed_panel_stride,

            a_tile_pool,
            a_tile_stride,

            acc_pool,
            acc_stride,

            idx_buf_pool,
            idx_stride,

            task_meta_pool,
            task_meta_stride,
            routed_tokens_pool,
            routed_slots_pool,
            routed_scores_pool,
            routed_stride,
        }
    }

    #[inline(always)]
    fn thread_slices(&self, tid: usize) -> (*mut T, *mut T, *mut usize) {
        unsafe {
            let a_tile = self.a_tile_pool.as_ptr().add(tid * self.a_tile_stride) as *mut T;
            let acc = self.acc_pool.as_ptr().add(tid * self.acc_stride) as *mut T;
            let idx = self.idx_buf_pool.as_ptr().add(tid * self.idx_stride) as *mut usize;
            (a_tile, acc, idx)
        }
    }

    #[inline(always)]
    fn pack_expert_b_panels(
        weight_nt: *const T, // [expert_count, output_cols, reduction_cols]
        expert_count: usize,
        output_cols: usize,
        reduction_cols: usize,
        reduction_block_cols: usize,
        micro_tile_cols: usize,
    ) -> Box<[T]> {
        let reduction_panel_count = reduction_cols.div_ceil(reduction_block_cols);
        let output_panel_count = output_cols.div_ceil(micro_tile_cols);
        let panel_stride = reduction_block_cols * micro_tile_cols;
        let expert_stride = reduction_panel_count * output_panel_count * panel_stride;
        let mut packed = vec![T::default(); expert_count * expert_stride];

        unsafe {
            for expert_id in 0..expert_count {
                let source_expert = weight_nt.add(expert_id * output_cols * reduction_cols);
                let packed_expert = packed.as_mut_ptr().add(expert_id * expert_stride);
                for reduction_panel_index in 0..reduction_panel_count {
                    let reduction_start = reduction_panel_index * reduction_block_cols;
                    let reduction_cols_this =
                        (reduction_cols - reduction_start).min(reduction_block_cols);
                    for output_panel_index in 0..output_panel_count {
                        let output_start = output_panel_index * micro_tile_cols;
                        let output_cols_this = (output_cols - output_start).min(micro_tile_cols);
                        let packed_panel = packed_expert.add(
                            (reduction_panel_index * output_panel_count + output_panel_index)
                                * panel_stride,
                        );
                        for reduction_lane in 0..reduction_cols_this {
                            let packed_row = packed_panel.add(reduction_lane * micro_tile_cols);
                            for output_lane in 0..output_cols_this {
                                *packed_row.add(output_lane) = *source_expert.add(
                                    (output_start + output_lane) * reduction_cols
                                        + (reduction_start + reduction_lane),
                                );
                            }
                        }
                    }
                }
            }
        }

        packed.into_boxed_slice()
    }

    #[inline(always)]
    fn packed_panel_ptr_dim(
        &self,
        packed: &[T],
        expert_id: usize,
        output_cols: usize,
        reduction_cols: usize,
        output_col_start: usize,
        reduction_col_start: usize,
    ) -> *const T {
        let reduction_block_cols = self.params.column_step_macro.max(1);
        let micro_tile_cols = self.params.b_row_step_micro.max(1);
        let output_panel_count = output_cols.div_ceil(micro_tile_cols);
        let reduction_panel_count = reduction_cols.div_ceil(reduction_block_cols);
        let expert_stride = reduction_panel_count * output_panel_count * self.packed_panel_stride;
        let panel_index = (reduction_col_start / reduction_block_cols) * output_panel_count
            + (output_col_start / micro_tile_cols);
        unsafe {
            packed
                .as_ptr()
                .add(expert_id * expert_stride + panel_index * self.packed_panel_stride)
        }
    }

    /// Pack dense shared tokens into a micro input tile (row-major), zero-padding.
    /// 将稠密 shared token 收集到微内核输入 tile（行主序），未使用行补零。
    #[inline(always)]
    unsafe fn pack_dense_tile(
        shared_base: *const T, // [B,Is]
        row_stride: usize,     // shared_inter
        idx_buf: *const usize,
        valid_rows: usize,
        reduction_col_start: usize,
        reduction_block_cols: usize,
        a_tile: *mut T,
        micro_tile_rows: usize,
    ) {
        for row_in_tile in 0..valid_rows {
            let token_id = *idx_buf.add(row_in_tile);
            let source_row = shared_base.add(token_id * row_stride + reduction_col_start);
            let packed_row = a_tile.add(row_in_tile * reduction_block_cols);
            for reduction_lane in 0..reduction_block_cols {
                *packed_row.add(reduction_lane) = *source_row.add(reduction_lane);
            }
        }
        for row_in_tile in valid_rows..micro_tile_rows {
            let packed_row = a_tile.add(row_in_tile * reduction_block_cols);
            for reduction_lane in 0..reduction_block_cols {
                *packed_row.add(reduction_lane) = T::default();
            }
        }
    }

    /// Pack routed tokens into a micro input tile from the per-expert nonlinear buffer.
    /// 从每个 expert 的非线性 buffer 将 routed token 收集到微内核输入 tile。
    #[inline(always)]
    unsafe fn pack_a_tile(
        &self,
        expert_id: usize,
        reduction_col_start: usize,
        valid_rows: usize,
        idx_buf: *const usize,
        idx_off: usize,
        a_tile: *mut T,
        reduction_block_cols: usize,
        micro_tile_rows: usize,
    ) {
        let expert_input_base = self
            .nonlin_ptr
            .ptr
            .add(expert_id * (self.num_token * self.hmid));

        for row_in_tile in 0..valid_rows {
            let token_id = *idx_buf.add(idx_off + row_in_tile);
            let source_row = expert_input_base.add(token_id * self.hmid + reduction_col_start);
            let packed_row = a_tile.add(row_in_tile * reduction_block_cols);
            for reduction_lane in 0..reduction_block_cols {
                *packed_row.add(reduction_lane) = *source_row.add(reduction_lane);
            }
        }
        for row_in_tile in valid_rows..micro_tile_rows {
            let packed_row = a_tile.add(row_in_tile * reduction_block_cols);
            for reduction_lane in 0..reduction_block_cols {
                *packed_row.add(reduction_lane) = T::default();
            }
        }
    }

    #[inline]
    fn build_task_space(
        &self,
        thread_id: usize,
        batch_size: usize,
        token_block_rows: usize,
        output_column_tile_count: usize,
    ) -> (&[ExpertTaskMeta], &[usize], &[usize], &[T], usize) {
        let expert_tasks_ptr =
            self.task_meta_pool
                .as_ptr()
                .wrapping_add(thread_id * self.task_meta_stride) as *mut ExpertTaskMeta;
        let routed_tokens_ptr =
            self.routed_tokens_pool
                .as_ptr()
                .wrapping_add(thread_id * self.routed_stride) as *mut usize;
        let routed_slots_ptr =
            self.routed_slots_pool
                .as_ptr()
                .wrapping_add(thread_id * self.routed_stride) as *mut usize;
        let routed_scores_ptr = self
            .routed_scores_pool
            .as_ptr()
            .wrapping_add(thread_id * self.routed_stride) as *mut T;
        let mut expert_task_count = 0usize;
        let mut routed_count = 0usize;
        let mut total_tasks = 0usize;

        unsafe {
            for expert_id in 0..self.num_experts {
                let routed_token_count =
                    (&*self.routing.expert_counts.ptr.add(expert_id)).load(Ordering::Acquire);
                if routed_token_count == 0 {
                    continue;
                }

                let token_begin = routed_count;
                let routed_token_count = routed_token_count.min(batch_size);
                for expert_queue_pos in 0..routed_token_count {
                    let route_offset = self.routing.expert_offset(expert_id, expert_queue_pos);
                    let token_id = *self.routing.index_tensor.ptr.add(route_offset);
                    let token_topk_row =
                        self.routing.topk_indices.ptr.add(token_id * self.num_topk);
                    let mut topk_slot = 0usize;
                    for slot_index in 0..self.num_topk {
                        if *token_topk_row.add(slot_index) == expert_id {
                            topk_slot = slot_index;
                            break;
                        }
                    }

                    *routed_tokens_ptr.add(routed_count) = token_id;
                    *routed_slots_ptr.add(routed_count) = topk_slot;
                    *routed_scores_ptr.add(routed_count) =
                        *self.routing.score_tensor.ptr.add(route_offset);
                    routed_count += 1;
                }

                let sequence_length = routed_count - token_begin;
                if sequence_length == 0 {
                    routed_count = token_begin;
                    continue;
                }

                let token_tile_count = sequence_length.div_ceil(token_block_rows);
                let task_count = token_tile_count * output_column_tile_count;
                *expert_tasks_ptr.add(expert_task_count) = ExpertTaskMeta {
                    expert_id,
                    token_begin,
                    sequence_length,
                    task_begin: total_tasks,
                    task_end: total_tasks + task_count,
                };
                expert_task_count += 1;
                total_tasks += task_count;
            }

            (
                std::slice::from_raw_parts(expert_tasks_ptr, expert_task_count),
                std::slice::from_raw_parts(routed_tokens_ptr, routed_count),
                std::slice::from_raw_parts(routed_slots_ptr, routed_count),
                std::slice::from_raw_parts(routed_scores_ptr, routed_count),
                total_tasks,
            )
        }
    }

    pub fn run(
        &self,
        _prefill_size: usize,
        _decode_size: usize,
        _total_size: usize,
        lift_size: usize,
        thread_num: usize,
        thread_id: usize,
    ) {
        unsafe {
            let active_token_count = if self.decode_only_flag {
                lift_size
            } else {
                _total_size
            };
            let output_cols = self.h;

            let token_block_rows = self.params.a_row_step_macro.max(1);
            let output_block_cols = self.params.b_row_step_macro.max(1);
            let reduction_block_cols = self.params.column_step_macro.max(1);
            let micro_tile_rows = self.params.a_row_step_micro.max(1);
            let micro_tile_cols = self.params.b_row_step_micro.max(1);

            debug_assert!(output_cols % micro_tile_cols == 0);
            debug_assert!(self.hmid % reduction_block_cols == 0);
            debug_assert!(self.shared_inter % reduction_block_cols == 0);

            let (a_tile, acc, idx_buf) = self.thread_slices(thread_id);

            /* ---------------- Phase A: shared expert (dense) ---------------- */
            /* ---------------- 阶段 A：shared expert（稠密） ---------------- */
            let shared_reduction_cols = self.shared_inter;
            let shared_nonlin_base = self.shared_nonlin_ptr.ptr;
            let shared_output_base = self.shared_output_ptr.ptr;

            let shared_output_column_tile_count = output_cols.div_ceil(output_block_cols);
            let shared_token_tile_count = active_token_count.div_ceil(token_block_rows);
            let shared_total_tasks = shared_token_tile_count * shared_output_column_tile_count;

            if let Some((task_begin, task_end)) = assign(shared_total_tasks, thread_num, thread_id)
            {
                for task_id in task_begin..task_end {
                    let token_tile_id = task_id / shared_output_column_tile_count;
                    let output_tile_id = task_id % shared_output_column_tile_count;

                    let token_block_start = token_tile_id * token_block_rows;
                    let tokens_in_block =
                        (active_token_count - token_block_start).min(token_block_rows);
                    let output_col_start = output_tile_id * output_block_cols;
                    let output_cols_in_block =
                        (output_cols - output_col_start).min(output_block_cols);
                    if output_cols_in_block == 0 || tokens_in_block == 0 {
                        continue;
                    }

                    let mut output_col_offset = 0usize;
                    while output_col_offset < output_cols_in_block {
                        let output_cols_this =
                            (output_cols_in_block - output_col_offset).min(micro_tile_cols);

                        let mut token_offset_in_block = 0usize;
                        while token_offset_in_block < tokens_in_block {
                            let valid_rows =
                                (tokens_in_block - token_offset_in_block).min(micro_tile_rows);

                            for row_in_tile in 0..valid_rows {
                                *idx_buf.add(row_in_tile) =
                                    token_block_start + token_offset_in_block + row_in_tile;
                            }

                            for accumulator_index in 0..(micro_tile_rows * micro_tile_cols) {
                                *acc.add(accumulator_index) = T::default();
                            }

                            let mut reduction_col_start = 0usize;
                            while reduction_col_start < shared_reduction_cols {
                                let weight_panel = self.packed_panel_ptr_dim(
                                    &self.packed_shared_wdown,
                                    0,
                                    output_cols,
                                    shared_reduction_cols,
                                    output_col_start + output_col_offset,
                                    reduction_col_start,
                                );

                                if valid_rows == 1 {
                                    let token_id = *idx_buf;
                                    let input_row = shared_nonlin_base.add(
                                        token_id * shared_reduction_cols + reduction_col_start,
                                    );
                                    self.compute1_single(
                                        input_row,
                                        weight_panel,
                                        acc,
                                        reduction_block_cols,
                                    );
                                } else if valid_rows < micro_tile_rows {
                                    Self::pack_dense_tile(
                                        shared_nonlin_base,
                                        shared_reduction_cols,
                                        idx_buf,
                                        valid_rows,
                                        reduction_col_start,
                                        reduction_block_cols,
                                        a_tile,
                                        micro_tile_rows,
                                    );
                                    self.compute1_rows(
                                        a_tile as *const T,
                                        weight_panel,
                                        acc,
                                        reduction_block_cols,
                                        valid_rows,
                                    );
                                } else {
                                    Self::pack_dense_tile(
                                        shared_nonlin_base,
                                        shared_reduction_cols,
                                        idx_buf,
                                        valid_rows,
                                        reduction_col_start,
                                        reduction_block_cols,
                                        a_tile,
                                        micro_tile_rows,
                                    );
                                    self.compute1(a_tile as *const T, weight_panel, acc);
                                }

                                reduction_col_start += reduction_block_cols;
                            }

                            // Shared down has no route score: store the accumulator directly.
                            // shared down 无 route score：直接写入累加结果。
                            for row_in_tile in 0..valid_rows {
                                let token_id = *idx_buf.add(row_in_tile);
                                let out_row = shared_output_base
                                    .add(token_id * output_cols)
                                    .add(output_col_start + output_col_offset);
                                let acc_row = acc.add(row_in_tile * micro_tile_cols) as *const T;
                                for col_in_tile in 0..output_cols_this {
                                    *out_row.add(col_in_tile) = *acc_row.add(col_in_tile);
                                }
                            }

                            token_offset_in_block += valid_rows;
                        }

                        output_col_offset += micro_tile_cols;
                    }
                }
            }

            /* ---------------- Phase B: routed experts (sparse) ---------------- */
            /* ---------------- 阶段 B：routed experts（稀疏） ---------------- */
            let reduction_cols = self.hmid;
            let output_column_tile_count = output_cols.div_ceil(output_block_cols);
            let (expert_tasks, routed_tokens, routed_slots, routed_scores, total_tasks) = self
                .build_task_space(
                    thread_id,
                    active_token_count,
                    token_block_rows,
                    output_column_tile_count,
                );

            if let Some((task_begin, task_end)) = assign(total_tasks, thread_num, thread_id) {
                for task_id in task_begin..task_end {
                    let Some((task_meta, token_tile_id, output_tile_id)) =
                        task_assign(&expert_tasks, output_column_tile_count, task_id)
                    else {
                        continue;
                    };

                    let token_block_start = token_tile_id * token_block_rows;
                    let output_col_start = output_tile_id * output_block_cols;
                    let output_cols_in_block =
                        (output_cols - output_col_start).min(output_block_cols);
                    if output_cols_in_block == 0 {
                        continue;
                    }

                    let tokens_in_block =
                        (task_meta.sequence_length - token_block_start).min(token_block_rows);
                    debug_assert!(tokens_in_block > 0);

                    let routed_token_begin = task_meta.token_begin + token_block_start;
                    for token_offset in 0..tokens_in_block {
                        *idx_buf.add(token_offset) =
                            routed_tokens[routed_token_begin + token_offset];
                    }

                    let expert_id = task_meta.expert_id;
                    let mut output_col_offset = 0usize;
                    while output_col_offset < output_cols_in_block {
                        let output_cols_this =
                            (output_cols_in_block - output_col_offset).min(micro_tile_cols);

                        let mut token_offset_in_block = 0usize;
                        while token_offset_in_block < tokens_in_block {
                            let valid_rows =
                                (tokens_in_block - token_offset_in_block).min(micro_tile_rows);

                            for accumulator_index in 0..(micro_tile_rows * micro_tile_cols) {
                                *acc.add(accumulator_index) = T::default();
                            }

                            let mut reduction_col_start = 0usize;
                            debug_assert!(reduction_cols % reduction_block_cols == 0);
                            while reduction_col_start < reduction_cols {
                                let weight_panel = self.packed_panel_ptr_dim(
                                    &self.packed_wdown,
                                    expert_id,
                                    output_cols,
                                    reduction_cols,
                                    output_col_start + output_col_offset,
                                    reduction_col_start,
                                );

                                if valid_rows == 1 {
                                    let token_id = *idx_buf.add(token_offset_in_block);
                                    let input_row = self
                                        .nonlin_ptr
                                        .ptr
                                        .add(expert_id * (self.num_token * self.hmid))
                                        .add(token_id * self.hmid + reduction_col_start);
                                    self.compute1_single(
                                        input_row,
                                        weight_panel,
                                        acc,
                                        reduction_block_cols,
                                    );
                                } else if valid_rows < micro_tile_rows {
                                    Self::pack_a_tile(
                                        self,
                                        expert_id,
                                        reduction_col_start,
                                        valid_rows,
                                        idx_buf,
                                        token_offset_in_block,
                                        a_tile,
                                        reduction_block_cols,
                                        micro_tile_rows,
                                    );
                                    self.compute1_rows(
                                        a_tile as *const T,
                                        weight_panel,
                                        acc,
                                        reduction_block_cols,
                                        valid_rows,
                                    );
                                } else {
                                    Self::pack_a_tile(
                                        self,
                                        expert_id,
                                        reduction_col_start,
                                        valid_rows,
                                        idx_buf,
                                        token_offset_in_block,
                                        a_tile,
                                        reduction_block_cols,
                                        micro_tile_rows,
                                    );
                                    self.compute1(a_tile as *const T, weight_panel, acc);
                                }

                                reduction_col_start += reduction_block_cols;
                            }

                            for row_in_tile in 0..valid_rows {
                                let token_id = *idx_buf.add(token_offset_in_block + row_in_tile);
                                let route_weight = routed_scores
                                    [routed_token_begin + token_offset_in_block + row_in_tile];
                                let topk_slot = routed_slots
                                    [routed_token_begin + token_offset_in_block + row_in_tile];

                                let out_row = self.output_ptr.ptr.add(
                                    token_id * (self.num_topk * output_cols)
                                        + topk_slot * output_cols
                                        + (output_col_start + output_col_offset),
                                );

                                let acc_row = acc.add(row_in_tile * micro_tile_cols) as *const T;
                                for col_in_tile in 0..output_cols_this {
                                    *out_row.add(col_in_tile) = T::default();
                                }

                                self.compute2(
                                    out_row,
                                    acc_row,
                                    &route_weight as *const T,
                                    output_cols_this,
                                );
                            }

                            token_offset_in_block += valid_rows;
                        }

                        output_col_offset += micro_tile_cols;
                    }
                }
            }
        }
    }
}

/* -------------------- SharedExpertsDownTrait scalar implementation -------------------- */
/* -------------------- SharedExpertsDownTrait 标量实现 -------------------- */

impl<T> SharedExpertsDownTrait<T> for SharedExpertMatMulDown<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    /// compute1: acc += A_tile * B_panel over the full micro tile.
    /// compute1：对整个微 tile 执行 acc += A_tile * B_panel。
    fn compute1(&self, a_tile: *const T, b_panel: *const T, acc: *mut T) {
        unsafe {
            let micro_tile_rows = self.params.a_row_step_micro.max(1);
            let micro_tile_cols = self.params.b_row_step_micro.max(1);
            let reduction_block_cols = self.params.column_step_macro.max(1);
            for row_in_tile in 0..micro_tile_rows {
                for col_in_tile in 0..micro_tile_cols {
                    let mut sum = *acc.add(row_in_tile * micro_tile_cols + col_in_tile);
                    for reduction_lane in 0..reduction_block_cols {
                        sum = sum
                            + *a_tile.add(row_in_tile * reduction_block_cols + reduction_lane)
                                * *b_panel.add(reduction_lane * micro_tile_cols + col_in_tile);
                    }
                    *acc.add(row_in_tile * micro_tile_cols + col_in_tile) = sum;
                }
            }
        }
    }

    fn compute1_single(&self, input_row: *const T, b_panel: *const T, acc: *mut T, kc: usize) {
        unsafe {
            let micro_tile_cols = self.params.b_row_step_micro.max(1);
            for col_in_tile in 0..micro_tile_cols {
                let mut sum = *acc.add(col_in_tile);
                for reduction_lane in 0..kc {
                    sum = sum
                        + *input_row.add(reduction_lane)
                            * *b_panel.add(reduction_lane * micro_tile_cols + col_in_tile);
                }
                *acc.add(col_in_tile) = sum;
            }
        }
    }

    fn compute1_rows(
        &self,
        a_tile: *const T,
        b_panel: *const T,
        acc: *mut T,
        kc: usize,
        rows: usize,
    ) {
        unsafe {
            let micro_tile_cols = self.params.b_row_step_micro.max(1);
            for row_in_tile in 0..rows {
                for col_in_tile in 0..micro_tile_cols {
                    let mut sum = *acc.add(row_in_tile * micro_tile_cols + col_in_tile);
                    for reduction_lane in 0..kc {
                        sum = sum
                            + *a_tile.add(row_in_tile * kc + reduction_lane)
                                * *b_panel.add(reduction_lane * micro_tile_cols + col_in_tile);
                    }
                    *acc.add(row_in_tile * micro_tile_cols + col_in_tile) = sum;
                }
            }
        }
    }

    /// compute2: out_row[j] += acc_row[j] * factor for j < len.
    /// compute2：对 j < len 执行 out_row[j] += acc_row[j] * factor。
    fn compute2(&self, out_row: *mut T, acc_row: *const T, factor: *const T, len: usize) {
        unsafe {
            let factor_val = *factor;
            for col_in_tile in 0..len {
                let out = *out_row.add(col_in_tile);
                let acc = *acc_row.add(col_in_tile);
                *out_row.add(col_in_tile) = out + acc * factor_val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::expert::expert_routing::routing_from_dense;

    // Shared dims: h%nr==0, hmid%kc==0, shared_inter%kc==0, nb%nr==0.
    // 共享维度：h%nr==0, hmid%kc==0, shared_inter%kc==0, nb%nr==0。
    const B: usize = 5;
    const H: usize = 16; // output hidden
    const HMID: usize = 8; // routed reduction
    const IS: usize = 16; // shared reduction
    const E: usize = 2;
    const KTOP: usize = 2;
    const MB: usize = 3;
    const NB: usize = 4;
    const KC: usize = 8;
    const MR: usize = 3;
    const NR: usize = 4;

    #[inline]
    fn slot_of(topk: &[usize], b: usize, e: usize) -> usize {
        let row = &topk[b * KTOP..b * KTOP + KTOP];
        row.iter().position(|&x| x == e).unwrap_or(0)
    }

    fn run_and_verify(cpu_num: usize) {
        // Routed nonlinear input [E,B,HMID] and NT down weight [E,H,HMID].
        let mut nonlin = vec![0.0f32; E * B * HMID];
        let mut wdown_nt = vec![0.0f32; E * H * HMID];
        for e in 0..E {
            for b in 0..B {
                for k in 0..HMID {
                    nonlin[e * (B * HMID) + b * HMID + k] =
                        0.003 * e as f32 + 0.01 * b as f32 + 0.002 * k as f32 - 0.02;
                }
            }
            for j in 0..H {
                for k in 0..HMID {
                    wdown_nt[e * (H * HMID) + j * HMID + k] =
                        0.001 * e as f32 + 0.002 * j as f32 + 0.001 * k as f32 + 0.005;
                }
            }
        }

        // Shared nonlinear input [B,IS] and NT down weight [H,IS].
        let mut shared_nonlin = vec![0.0f32; B * IS];
        let mut shared_wdown_nt = vec![0.0f32; H * IS];
        for b in 0..B {
            for is in 0..IS {
                shared_nonlin[b * IS + is] = 0.01 * b as f32 + 0.002 * is as f32 - 0.03;
            }
        }
        for j in 0..H {
            for is in 0..IS {
                shared_wdown_nt[j * IS + is] = 0.002 * j as f32 + 0.001 * is as f32 - 0.01;
            }
        }

        // Routing: expert0 hits {0,1,2}, expert1 hits {2,3}; topk rows are [0,1].
        let mut indice = vec![false; E * B];
        let mut score = vec![0.0f32; E * B];
        let mut topk = vec![0usize; B * KTOP];
        for b in 0..B {
            topk[b * KTOP + 0] = 0;
            topk[b * KTOP + 1] = 1;
        }
        for &b in &[0usize, 1, 2] {
            indice[0 * B + b] = true;
        }
        for &b in &[2usize, 3] {
            indice[1 * B + b] = true;
        }
        for e in 0..E {
            for b in 0..B {
                score[e * B + b] = 0.5 + 0.1 * e as f32 + 0.05 * b as f32;
            }
        }

        let mut routed_out = vec![0.0f32; B * KTOP * H];
        let mut shared_out = vec![0.0f32; B * H];

        unsafe {
            let routing =
                routing_from_dense(E, B, KTOP, indice.as_ptr(), score.as_ptr(), topk.as_ptr());
            let params = MatMulParams {
                a_row_step_macro: MB,
                b_row_step_macro: NB,
                column_step_macro: KC,
                a_row_step_micro: MR,
                b_row_step_micro: NR,
            };
            let op = SharedExpertMatMulDown::<f32>::new(
                nonlin.as_ptr(),
                wdown_nt.as_ptr(),
                shared_nonlin.as_ptr(),
                shared_wdown_nt.as_ptr(),
                routing,
                routed_out.as_mut_ptr(),
                shared_out.as_mut_ptr(),
                E,
                B,
                HMID,
                IS,
                H,
                KTOP,
                params,
                false,
            );
            for tid in 0..cpu_num {
                op.run(B, 0, B, B, cpu_num, tid);
            }
        }

        // Reference: shared dense GEMM.
        for b in 0..B {
            for j in 0..H {
                let mut acc = 0.0f32;
                for is in 0..IS {
                    acc += shared_nonlin[b * IS + is] * shared_wdown_nt[j * IS + is];
                }
                assert!(
                    (shared_out[b * H + j] - acc).abs() < 1e-4,
                    "shared mismatch b={} j={} got={} exp={}",
                    b,
                    j,
                    shared_out[b * H + j],
                    acc
                );
            }
        }

        // Reference: routed slot scatter with score scaling.
        let mut routed_ref = vec![0.0f32; B * KTOP * H];
        for e in 0..E {
            for b in 0..B {
                if !indice[e * B + b] {
                    continue;
                }
                let slot = slot_of(&topk, b, e);
                let w = score[e * B + b];
                for j in 0..H {
                    let mut acc = 0.0f32;
                    for k in 0..HMID {
                        acc += nonlin[e * (B * HMID) + b * HMID + k]
                            * wdown_nt[e * (H * HMID) + j * HMID + k];
                    }
                    routed_ref[(b * KTOP + slot) * H + j] += w * acc;
                }
            }
        }
        for idx in 0..(B * KTOP * H) {
            assert!(
                (routed_out[idx] - routed_ref[idx]).abs() < 1e-4,
                "routed mismatch at {} got={} exp={}",
                idx,
                routed_out[idx],
                routed_ref[idx]
            );
        }
    }

    #[test]
    fn test_shared_down_single_thread() {
        run_and_verify(1);
    }

    #[test]
    fn test_shared_down_multithread() {
        run_and_verify(4);
    }
}
