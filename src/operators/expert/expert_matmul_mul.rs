// === compiler/mul/experts_matmul_mul.rs ===
#![allow(non_snake_case)]

use crate::kernel::common::matmul_params::MatMulParams;
use crate::operators::assign::assign;
use crate::operators::expert::expert_routing::{task_assign, ExpertRouting, ExpertTaskMeta};
use crate::operators::send_sync_ptr::{ConstPtr, MutPtr};
use crate::operators::traits::ExpertsDownTrait;
use std::f16;
use std::marker::PhantomData;
use std::ops::{Add, Mul};
use std::sync::atomic::Ordering;

// Variable naming used in this operator:
// - token_block_rows / token_block_start: routed-token macro block inside one expert.
// - output_cols / output_col_start: down-projection output hidden columns.
// - reduction_cols / reduction_col_start: intermediate Hmid dimension reduced by GEMM.
// - micro_tile_rows / micro_tile_cols: micro-kernel tile size.
// - routed_token_begin / token_offset_in_block: positions in the compact expert queue.
// - topk_slot: token-major output slot for one expert route.
// 本算子的变量命名约定：
// - token_block_rows / token_block_start：单个 expert 内 routed token 的宏块。
// - output_cols / output_col_start：down projection 输出 hidden 列。
// - reduction_cols / reduction_col_start：GEMM 中被规约的 intermediate Hmid 维度。
// - micro_tile_rows / micro_tile_cols：微内核 tile 大小。
// - routed_token_begin / token_offset_in_block：compact expert queue 中的位置。
// - topk_slot：某个 expert route 在 token-major 输出中的 slot。

/// Experts Down Projection:
///   NONLIN[e, b, Hmid]   ×  W_down[e, Hmid, H]   → OUT[b, slot(b,e), H]
///
/// Routing uses a compact per-expert queue.
/// 路由使用按 expert 压紧的 token 队列。
///
/// Output slots are resolved from the token-major top-k expert list.
/// 输出 slot 从 token-major 的 top-k expert 列表中解析。
///
/// This operator does not add residual.
/// 该算子不做 residual 累加。
#[derive(Clone)]
pub struct ExpertMatMulDown<T> {
    pub nonlin_ptr: ConstPtr<T>, // Nonlinear input: [E, B, Hmid]. 非线性输入。
    pub wdown_nt_ptr: ConstPtr<T>, // Down weight NT: [E, H, Hmid]. down 权重 NT 布局。

    pub routing: ExpertRouting<T>,

    pub output_ptr: MutPtr<T>, // Token-major output: [B, Ktop, H]. token-major 输出。

    pub num_experts: usize, // Expert count. expert 数量。
    pub num_token: usize,   // Token capacity. token 容量。
    pub hmid: usize,        // Intermediate hidden size. 中间层 hidden 大小。
    pub h: usize,           // Output hidden size. 输出 hidden 大小。
    pub num_topk: usize,    // Top-k experts per token. 每个 token 的 top-k expert 数。
    pub decode_only_flag: bool,

    pub params: MatMulParams,
    _marker: PhantomData<T>,

    // ---- prepacked weights ----
    // Weight panels are packed in new(), so run() only reads prebuilt memory.
    // 权重 panel 在 new() 中提前 pack，run() 中只读取已准备好的内存。
    packed_wdown: Box<[T]>, // [E][reduction_panels][output_panels][reduction_block * micro_cols]
    packed_panel_stride: usize, // reduction_block_cols * micro_tile_cols

    // Input tile: micro_rows × reduction_block, row-major.
    // 输入 tile：micro_rows × reduction_block，行主序。
    a_tile_pool: Box<[T]>,
    a_tile_stride: usize, // micro_tile_rows * reduction_block_cols

    // Accumulator tile: micro_rows × micro_cols, row-major.
    // 累加 tile：micro_rows × micro_cols，行主序。
    acc_pool: Box<[T]>,
    acc_stride: usize, // micro_tile_rows * micro_tile_cols

    // Routed token index buffer, one slice per thread.
    // 路由后的 token 下标缓存，每个线程一份。
    idx_buf_pool: Box<[usize]>,
    idx_stride: usize, // token_block_rows

    // Task-space buffers, one slice per thread. run() reuses them without allocation.
    // task 空间缓存，每个线程一份；run() 中只复用，不动态分配。
    task_meta_pool: Box<[ExpertTaskMeta]>,
    task_meta_stride: usize, // num_experts
    routed_tokens_pool: Box<[usize]>,
    routed_slots_pool: Box<[usize]>,
    routed_scores_pool: Box<[T]>,
    routed_stride: usize, // num_experts * capacity_per_expert
}

impl<T> ExpertMatMulDown<T>
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

    pub unsafe fn new(
        nonlin_ptr: *const T,   // Nonlinear input: [E,B,Hmid]. 非线性输入。
        wdown_nt_ptr: *const T, // Down weight NT: [E,H,Hmid]. down 权重 NT 布局。
        routing: ExpertRouting<T>,
        output_ptr: *mut T, // Token-major output: [B,Ktop,H]. token-major 输出。

        num_experts: usize,
        num_token: usize,
        hmid: usize,
        h: usize,
        num_topk: usize,

        params: MatMulParams,
        decode_only_flag: bool,
    ) -> Self {
        let token_block_rows = params.a_row_step_macro.max(1);
        let reduction_block_cols = params.column_step_macro.max(1);
        let micro_tile_rows = params.a_row_step_micro.max(1);
        let micro_tile_cols = params.b_row_step_micro.max(1);

        // Current micro-kernel convention is micro_tile_rows=3, micro_tile_cols=32.
        // 当前微内核约定 micro_tile_rows=3, micro_tile_cols=32；这里不 assert，由外部保证。
        // debug_assert_eq!(mr, 3);
        // debug_assert_eq!(nr, 32);

        // Weight layout is expected as NT per expert: [H][Hmid], row stride = Hmid.
        // 权重要求按每个 expert 的 NT 布局传入：[H][Hmid]，行距为 Hmid。

        let packed_panel_stride = reduction_block_cols * micro_tile_cols;
        let packed_wdown = Self::pack_expert_b_panels(
            wdown_nt_ptr,
            num_experts,
            h,
            hmid,
            reduction_block_cols,
            micro_tile_cols,
        );

        // Allocate thread-private pools once in new().
        // 在 new() 中一次性分配每线程私有缓存。
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

            routing,

            output_ptr: MutPtr { ptr: output_ptr },

            num_experts,
            num_token,
            hmid,
            h,
            num_topk,
            decode_only_flag,

            params,
            _marker: PhantomData,

            packed_wdown,
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
        weight_nt: *const T, // [E, output_cols, reduction_cols]
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
    fn packed_panel_ptr(
        &self,
        expert_id: usize,
        output_col_start: usize,
        reduction_col_start: usize,
    ) -> *const T {
        let reduction_block_cols = self.params.column_step_macro.max(1);
        let micro_tile_cols = self.params.b_row_step_micro.max(1);
        let output_panel_count = self.h.div_ceil(micro_tile_cols);
        let reduction_panel_count = self.hmid.div_ceil(reduction_block_cols);
        let expert_stride = reduction_panel_count * output_panel_count * self.packed_panel_stride;
        let panel_index = (reduction_col_start / reduction_block_cols) * output_panel_count
            + (output_col_start / micro_tile_cols);
        unsafe {
            self.packed_wdown
                .as_ptr()
                .add(expert_id * expert_stride + panel_index * self.packed_panel_stride)
        }
    }

    /// Pack routed tokens into a micro input tile and zero-pad unused rows.
    /// 将路由后的 token 收集到微内核输入 tile，未使用的行补零。
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

                let token_count = routed_count - token_begin;
                if token_count == 0 {
                    routed_count = token_begin;
                    continue;
                }

                let token_tile_count = token_count.div_ceil(token_block_rows);
                let task_count = token_tile_count * output_column_tile_count;
                *expert_tasks_ptr.add(expert_task_count) = ExpertTaskMeta {
                    expert_id,
                    token_begin,
                    token_count,
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
        prefill_size: usize,
        decode_size: usize,
        thread_num: usize,
        thread_id: usize,
    ) {
        unsafe {
            let active_token_count = if prefill_size == 0 {
                decode_size
            } else {
                prefill_size
            };
            let output_cols = self.h;
            let reduction_cols = self.hmid;

            let token_block_rows = self.params.a_row_step_macro.max(1);
            let output_block_cols = self.params.b_row_step_macro.max(1);
            let reduction_block_cols = self.params.column_step_macro.max(1);
            let micro_tile_rows = self.params.a_row_step_micro.max(1);
            let micro_tile_cols = self.params.b_row_step_micro.max(1);

            let output_column_tile_count = output_cols.div_ceil(output_block_cols);
            let (expert_tasks, routed_tokens, routed_slots, routed_scores, total_tasks) = self
                .build_task_space(
                    thread_id,
                    active_token_count,
                    token_block_rows,
                    output_column_tile_count,
                );

            // Thread-private scratch slices.
            // 每线程私有 scratch 切片。
            let (a_tile, acc, idx_buf) = self.thread_slices(thread_id);

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
                        (task_meta.token_count - token_block_start).min(token_block_rows);
                    debug_assert!(tokens_in_block > 0);

                    let routed_token_begin = task_meta.token_begin + token_block_start;
                    for token_offset in 0..tokens_in_block {
                        *idx_buf.add(token_offset) =
                            routed_tokens[routed_token_begin + token_offset];
                    }

                    let expert_id = task_meta.expert_id;
                    // The macro output block may be wider than one micro tile.
                    // 输出宏块可能大于一个微内核 tile，因此内部继续按 micro_tile_cols 切分。
                    let mut output_col_offset = 0usize;
                    while output_col_offset < output_cols_in_block {
                        let output_cols_this =
                            (output_cols_in_block - output_col_offset).min(micro_tile_cols);

                        // Process routed tokens by micro rows to keep accumulator bounds valid.
                        // 按微内核行数处理 routed tokens，避免累加器越界。
                        let mut token_offset_in_block = 0usize;
                        while token_offset_in_block < tokens_in_block {
                            let valid_rows =
                                (tokens_in_block - token_offset_in_block).min(micro_tile_rows);

                            for accumulator_index in 0..(micro_tile_rows * micro_tile_cols) {
                                *acc.add(accumulator_index) = T::default();
                            }

                            // Accumulate along reduction dimension: acc += A_tile * B_panel.
                            // 沿 reduction 维度累加：acc += A_tile * B_panel。
                            let mut reduction_col_start = 0usize;
                            debug_assert!(reduction_cols % reduction_block_cols == 0);
                            while reduction_col_start < reduction_cols {
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

                                let weight_panel = self.packed_panel_ptr(
                                    expert_id,
                                    output_col_start + output_col_offset,
                                    reduction_col_start,
                                );

                                self.compute1(a_tile as *const T, weight_panel, acc);

                                reduction_col_start += reduction_block_cols;
                            }

                            // Scatter each valid row back to token-major output with route weight.
                            // 将每个有效行乘以路由权重后写回 token-major 输出。
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

/* ---------------- ExpertDownTrait default implementation ---------------- */
/* ---------------- ExpertDownTrait 默认实现 ---------------- */

impl<T> ExpertsDownTrait<T> for ExpertMatMulDown<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    // compute1: acc += A_tile * B_panel.
    // compute1：acc += A_tile * B_panel。
    default fn compute1(&self, _a_tile: *const T, _b_panel: *const T, _acc: *mut T) {}

    // compute2: out_row[j] += acc_row[j] * factor for len <= micro_tile_cols.
    // compute2：对 len <= micro_tile_cols 的输出执行 out_row[j] += acc_row[j] * factor。
    default fn compute2(
        &self,
        _out_row: *mut T,
        _acc_row: *const T,
        _factor: *const T,
        _len: usize,
    ) {
    }
}

/* ---------------- f16 specialization: AVX-512 FP16 ---------------- */
/* ---------------- f16 专用实现：AVX-512 FP16 ---------------- */

impl ExpertsDownTrait<f16> for ExpertMatMulDown<f16> {
    /// compute1: acc += A_tile * B_panel with the 3x32 micro-kernel.
    /// compute1：使用 3x32 微内核执行 acc += A_tile * B_panel。
    fn compute1(&self, a_tile: *const f16, b_panel: *const f16, acc: *mut f16) {
        let reduction_block_cols = self.params.column_step_macro.max(1);
        let micro_tile_rows = self.params.a_row_step_micro.max(1);
        let micro_tile_cols = self.params.b_row_step_micro.max(1);

        let _call_param = MatMulParams {
            // Map packed tile layout to the generic matmul kernel parameters.
            // 将 packed tile 的布局映射到通用 matmul kernel 参数。
            a_row_step_macro: reduction_block_cols,
            b_row_step_macro: micro_tile_cols,
            column_step_macro: reduction_block_cols,
            a_row_step_micro: micro_tile_rows,
            b_row_step_micro: micro_tile_cols,
        };

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::matmul_block::matmul_block(a_tile, b_panel, acc, &call_param);
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        unsafe {
            for row_in_tile in 0..micro_tile_rows {
                for col_in_tile in 0..micro_tile_cols {
                    let mut sum = *acc.add(row_in_tile * micro_tile_cols + col_in_tile) as f32;
                    for reduction_lane in 0..reduction_block_cols {
                        sum += (*a_tile.add(row_in_tile * reduction_block_cols + reduction_lane)
                            as f32)
                            * (*b_panel.add(reduction_lane * micro_tile_cols + col_in_tile) as f32);
                    }
                    *acc.add(row_in_tile * micro_tile_cols + col_in_tile) = sum as f16;
                }
            }
        }
    }

    /// compute2: out_row[j] += acc_row[j] * factor for a short output tail.
    /// compute2：对较短输出 tail 执行 out_row[j] += acc_row[j] * factor。
    fn compute2(&self, out_row: *mut f16, acc_row: *const f16, factor: *const f16, len: usize) {
        let factor_val = unsafe { *factor };
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::moe_down::moe_down_scale_add(
                out_row, acc_row, factor_val, len,
            );
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        unsafe {
            for i in 0..len {
                let out = *out_row.add(i) as f32;
                let acc = *acc_row.add(i) as f32;
                *out_row.add(i) = (out + acc * (factor_val as f32)) as f16;
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::arch::is_x86_feature_detected;
    use std::mem;

    use crate::num_traits::FromNumber;

    // ========================================================================
    // Helpers
    // ========================================================================

    fn test_routing_from_dense(
        num_experts: usize,
        num_token: usize,
        num_topk: usize,
        indice: &[bool],
        weight: &[f16],
        topk: &[usize],
    ) -> ExpertRouting<f16> {
        unsafe {
            crate::operators::expert::expert_routing::routing_from_dense(
                num_experts,
                num_token,
                num_topk,
                indice.as_ptr(),
                weight.as_ptr(),
                topk.as_ptr(),
            )
        }
    }

    #[inline]
    fn f16_from_f32(x: f32) -> f16 {
        <f16 as FromNumber>::from_f32(x)
    }

    /// 只给测试用：把 f16 bits 转成 f32
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
    fn approx_eq_f32(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    /// 找 slot(b,e)：topk 行是升序 expert id
    #[inline]
    fn slot_of(topk: &[usize], b: usize, ktop: usize, e: usize) -> usize {
        let row = &topk[b * ktop..b * ktop + ktop];
        row.iter().position(|&x| x == e).unwrap_or(0)
    }

    fn verify_output(out: &[f16], out_ref: &[f32], tol: f32, msg: &str) {
        for i in 0..out.len() {
            let got = f32_from_f16(out[i]);
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

    // ========================================================================
    // Test Cases
    // ========================================================================

    #[test]
    fn test_down_mb_gt_mr_basic_no_tail() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }

        let num_experts = 1usize; // E
        let num_token = 6usize; // B (be_total=6 > mr=3)
        let hmid = 32usize; // Hmid
        let h = 64usize; // H
        let num_topk = 1usize; // Ktop

        let params = MatMulParams {
            a_row_step_macro: 6,   // MB
            b_row_step_macro: 32,  // NB
            column_step_macro: 16, // KC (hmid%kc==0)
            a_row_step_micro: 3,   // MR
            b_row_step_micro: 32,  // NR
        };

        let mut nonlin = vec![f16_from_f32(0.0); num_experts * num_token * hmid];
        let mut wdown = vec![f16_from_f32(0.0); num_experts * hmid * h];
        let mut out = vec![f16_from_f32(0.0); num_token * num_topk * h];

        let experts_indicator = vec![true; num_experts];
        let mut indice = vec![false; num_experts * num_token];
        let mut weight = vec![f16_from_f32(0.0); num_experts * num_token];
        let mut topk = vec![0usize; num_token * num_topk];

        // expert0 命中全部 token
        for b in 0..num_token {
            indice[0 * num_token + b] = true;
            weight[0 * num_token + b] = f16_from_f32(0.5 + 0.01 * b as f32);
            topk[b * num_topk + 0] = 0;
        }

        for b in 0..num_token {
            for kk in 0..hmid {
                nonlin[(0 * num_token + b) * hmid + kk] =
                    f16_from_f32(0.01 * b as f32 + 0.001 * kk as f32);
            }
        }
        for kk in 0..hmid {
            for j in 0..h {
                wdown[(0 * hmid + kk) * h + j] =
                    f16_from_f32(0.002 * kk as f32 + 0.0005 * j as f32);
            }
        }

        let runner = unsafe {
            ExpertMatMulDown::<f16>::new(
                nonlin.as_ptr(),
                wdown.as_ptr(),
                test_routing_from_dense(num_experts, num_token, num_topk, &indice, &weight, &topk),
                out.as_mut_ptr(),
                num_experts,
                num_token,
                hmid,
                h,
                num_topk,
                params,
                false,
            )
        };

        runner.run(num_token, 0, 1, 0);

        // reference (f32)
        let mut out_ref = vec![0.0f32; num_token * num_topk * h];
        for b in 0..num_token {
            if !indice[0 * num_token + b] {
                continue;
            }
            let slot = slot_of(&topk, b, num_topk, 0);
            let w = f32_from_f16(weight[0 * num_token + b]);

            for j in 0..h {
                let mut acc = 0.0f32;
                for kk in 0..hmid {
                    let a = f32_from_f16(nonlin[(0 * num_token + b) * hmid + kk]);
                    let bv = f32_from_f16(wdown[(0 * hmid + kk) * h + j]);
                    acc += a * bv;
                }
                out_ref[(b * num_topk + slot) * h + j] += w * acc;
            }
        }

        verify_output(&out, &out_ref, 5e-2, "basic");
    }

    #[test]
    fn test_down_tail_len_lt_32() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }

        // H=48 => 32 + 16 tail
        let num_experts = 1usize;
        let num_token = 3usize;
        let hmid = 32usize;
        let h = 48usize;
        let num_topk = 1usize;

        let params = MatMulParams {
            a_row_step_macro: 3,
            b_row_step_macro: 48, // NB=48（一个 tile 覆盖 48）
            column_step_macro: 16,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let mut nonlin = vec![f16_from_f32(0.0); num_experts * num_token * hmid];
        let mut wdown = vec![f16_from_f32(0.0); num_experts * hmid * h];
        let mut out = vec![f16_from_f32(0.0); num_token * num_topk * h];

        let experts_indicator = vec![true; num_experts];
        let indice = vec![true; num_experts * num_token];
        let mut weight = vec![f16_from_f32(0.0); num_experts * num_token];
        let mut topk = vec![0usize; num_token * num_topk];

        for b in 0..num_token {
            weight[b] = f16_from_f32(0.7 + 0.02 * b as f32);
            topk[b] = 0;
        }

        for b in 0..num_token {
            for kk in 0..hmid {
                nonlin[b * hmid + kk] = f16_from_f32(0.01 * b as f32 + 0.0003 * kk as f32);
            }
        }
        for kk in 0..hmid {
            for j in 0..h {
                wdown[kk * h + j] = f16_from_f32(0.001 * kk as f32 + 0.0009 * j as f32);
            }
        }

        let runner = unsafe {
            ExpertMatMulDown::<f16>::new(
                nonlin.as_ptr(),
                wdown.as_ptr(),
                test_routing_from_dense(num_experts, num_token, num_topk, &indice, &weight, &topk),
                out.as_mut_ptr(),
                num_experts,
                num_token,
                hmid,
                h,
                num_topk,
                params,
                false,
            )
        };

        runner.run(num_token, 0, 1, 0);

        // reference
        let mut out_ref = vec![0.0f32; num_token * num_topk * h];
        for b in 0..num_token {
            let w = f32_from_f16(weight[b]);
            for j in 0..h {
                let mut acc = 0.0f32;
                for kk in 0..hmid {
                    let a = f32_from_f16(nonlin[b * hmid + kk]);
                    let bv = f32_from_f16(wdown[kk * h + j]);
                    acc += a * bv;
                }
                out_ref[b * h + j] += w * acc;
            }
        }

        verify_output(&out, &out_ref, 5e-2, "tail");
    }

    #[test]
    fn test_down_two_experts_slot_scatter() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }

        let num_experts = 2usize;
        let num_token = 4usize;
        let hmid = 32usize;
        let h = 64usize;
        let num_topk = 2usize;

        let params = MatMulParams {
            a_row_step_macro: 4,
            b_row_step_macro: 32,
            column_step_macro: 16,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let mut nonlin = vec![f16_from_f32(0.0); num_experts * num_token * hmid];
        let mut wdown = vec![f16_from_f32(0.0); num_experts * hmid * h];
        let mut out = vec![f16_from_f32(0.0); num_token * num_topk * h];

        let experts_indicator = vec![true; num_experts];
        let mut indice = vec![false; num_experts * num_token];
        let mut weight = vec![f16_from_f32(0.0); num_experts * num_token];

        // topk: [0,1]
        let mut topk = vec![0usize; num_token * num_topk];
        for b in 0..num_token {
            topk[b * num_topk + 0] = 0;
            topk[b * num_topk + 1] = 1;
        }

        // routing
        indice[0 * num_token + 0] = true;
        indice[0 * num_token + 1] = true;
        indice[0 * num_token + 2] = true;

        indice[1 * num_token + 1] = true;
        indice[1 * num_token + 3] = true;

        for e in 0..num_experts {
            for b in 0..num_token {
                weight[e * num_token + b] = f16_from_f32(0.3 + 0.01 * e as f32 + 0.02 * b as f32);
            }
        }

        for e in 0..num_experts {
            for b in 0..num_token {
                for kk in 0..hmid {
                    nonlin[(e * num_token + b) * hmid + kk] =
                        f16_from_f32(0.005 * e as f32 + 0.01 * b as f32 + 0.0007 * kk as f32);
                }
            }
        }
        for e in 0..num_experts {
            for kk in 0..hmid {
                for j in 0..h {
                    wdown[(e * hmid + kk) * h + j] =
                        f16_from_f32(0.001 * e as f32 + 0.0009 * kk as f32 + 0.0002 * j as f32);
                }
            }
        }

        let runner = unsafe {
            ExpertMatMulDown::<f16>::new(
                nonlin.as_ptr(),
                wdown.as_ptr(),
                test_routing_from_dense(num_experts, num_token, num_topk, &indice, &weight, &topk),
                out.as_mut_ptr(),
                num_experts,
                num_token,
                hmid,
                h,
                num_topk,
                params,
                false,
            )
        };

        runner.run(num_token, 0, 1, 0);

        // token0 只命中 expert0 => slot1 应接近 0
        {
            let b = 0usize;
            let slot1 = 1usize;
            for j in 0..h {
                let v = f32_from_f16(out[(b * num_topk + slot1) * h + j]);
                assert!(v.abs() <= 1e-2, "token0 slot1 should be ~0, got {}", v);
            }
        }

        // token3 只命中 expert1 => slot0 应接近 0
        {
            let b = 3usize;
            let slot0 = 0usize;
            for j in 0..h {
                let v = f32_from_f16(out[(b * num_topk + slot0) * h + j]);
                assert!(v.abs() <= 1e-2, "token3 slot0 should be ~0, got {}", v);
            }
        }
    }
    #[test]
    fn test_down_multithread_tail_and_accumulate_semantics() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }

        let num_experts = 2usize; // E
        let num_token = 9usize; // B（故意 > MR=3 且 > MB）
        let hmid = 32usize; // K（整除 KC）
        let h = 48usize; // N：32 + 16 tail
        let num_topk = 2usize; // 每 token 两个 expert 槽

        let params = MatMulParams {
            a_row_step_macro: 6,   // MB
            b_row_step_macro: 48,  // NB 覆盖整行
            column_step_macro: 16, // KC（hmid%kc==0）
            a_row_step_micro: 3,   // MR
            b_row_step_micro: 32,  // NR
        };

        // nonlin[E,B,Hmid]
        let mut nonlin = vec![f16_from_f32(0.0); num_experts * num_token * hmid];
        // wdown[E,Hmid,H]  (这里仍按你算子输入的 K×N 形式给，算子内部会转置)
        let mut wdown = vec![f16_from_f32(0.0); num_experts * hmid * h];

        // 输出 [B,Ktop,H]：注意这里验证“+=”语义，所以我们先填一个非零底噪，再跑两次对比
        let mut out = vec![f16_from_f32(0.01); num_token * num_topk * h];

        let experts_indicator = vec![true; num_experts];

        // indice[E,B]：让 token 奇偶分别命中不同 expert，同时都在 topk 里
        let mut indice = vec![false; num_experts * num_token];
        for b in 0..num_token {
            indice[0 * num_token + b] = true; // expert0 命中所有
            indice[1 * num_token + b] = (b % 2 == 0); // expert1 命中偶数 token
        }

        // weight[E,B]
        let mut weight = vec![f16_from_f32(0.0); num_experts * num_token];
        for e in 0..num_experts {
            for b in 0..num_token {
                weight[e * num_token + b] = f16_from_f32(0.2 + 0.01 * e as f32 + 0.001 * b as f32);
            }
        }

        // topk[B,Ktop]：每行 [0,1] 升序
        let mut topk = vec![0usize; num_token * num_topk];
        for b in 0..num_token {
            topk[b * num_topk + 0] = 0;
            topk[b * num_topk + 1] = 1;
        }

        // 填 nonlin / wdown
        for e in 0..num_experts {
            for b in 0..num_token {
                for kk in 0..hmid {
                    nonlin[(e * num_token + b) * hmid + kk] =
                        f16_from_f32(0.01 * e as f32 + 0.002 * b as f32 + 0.0003 * kk as f32);
                }
            }
        }
        for e in 0..num_experts {
            for kk in 0..hmid {
                for j in 0..h {
                    // 让不同列/行有区分
                    wdown[(e * hmid + kk) * h + j] =
                        f16_from_f32(0.001 * e as f32 + 0.0007 * kk as f32 + 0.0002 * j as f32);
                }
            }
        }

        let runner = unsafe {
            ExpertMatMulDown::<f16>::new(
                nonlin.as_ptr(),
                wdown.as_ptr(),
                test_routing_from_dense(num_experts, num_token, num_topk, &indice, &weight, &topk),
                out.as_mut_ptr(),
                num_experts,
                num_token,
                hmid,
                h,
                num_topk,
                params,
                false,
            )
        };

        // 用 2 线程跑一遍
        let cpu_num = 2usize;
        for tid in 0..cpu_num {
            runner.run(num_token, 0, cpu_num, tid);
        }

        // 保存一次结果
        let out_once = out.clone();

        // 再跑一遍（验证 += 语义：第二次的增量应基本等于第一次的增量）
        for tid in 0..cpu_num {
            runner.run(num_token, 0, cpu_num, tid);
        }

        // reference：我们不做全量 ref（太慢），抽样检查若干点
        // 关键：检查 tail 区间 j>=32 也被正确写入（len=16），且两次运行的增量一致
        let mut checks = 0usize;
        for b in 0..num_token {
            for slot in 0..num_topk {
                let e = topk[b * num_topk + slot];
                if !indice[e * num_token + b] {
                    continue;
                }
                // 抽两个 j：一个在 0..32，一个在 tail 32..48
                for &j in &[7usize, 40usize] {
                    let idx = (b * num_topk + slot) * h + j;
                    let v0 = f32_from_f16(out_once[idx]);
                    let v1 = f32_from_f16(out[idx]);

                    // v1 - v0 应约等于 v0 - baseline(0.01)，但 baseline 我们只要求“增量一致”
                    let delta1 = v0 - 0.01;
                    let delta2 = v1 - v0;

                    assert!(
                        (delta1 - delta2).abs() <= 8e-2,
                        "accumulate mismatch at b={},slot={},j={}: delta1={}, delta2={}",
                        b,
                        slot,
                        j,
                        delta1,
                        delta2
                    );
                    checks += 1;
                    if checks >= 32 {
                        break;
                    }
                }
                if checks >= 32 {
                    break;
                }
            }
            if checks >= 32 {
                break;
            }
        }

        assert!(checks > 0, "no checks performed");
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_down_stride_must_use_capacity_not_run_batch() {
        use std::arch::is_x86_feature_detected;
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }

        use crate::num_traits::FromNumber;

        #[inline]
        fn f16_from_f32(x: f32) -> f16 {
            <f16 as FromNumber>::from_f32(x)
        }

        const E: usize = 2;
        const B_CAP: usize = 9; // capacity（构造时 num_token）
        const B_RUN: usize = 7; // run 时 batch_size
        const HMID: usize = 32;
        const H: usize = 32;
        const KTOP: usize = 2;

        // 3x32 微核配置（H=32 不会 tail）
        let params = MatMulParams {
            a_row_step_macro: 6,   // MB（随便 >= MR 即可）
            b_row_step_macro: 32,  // NB
            column_step_macro: 16, // KC（HMID%KC==0）
            a_row_step_micro: 3,   // MR
            b_row_step_micro: 32,  // NR
        };

        // -----------------------
        // 数据准备
        // -----------------------

        // NONLIN[e,b,k]
        // 全设为 1，便于算 expected
        let mut nonlin = vec![f16_from_f32(1.0); E * B_CAP * HMID];

        // W_down: 原始传入预期是 [E, HMID, H] (K×N) 行主 stride=H
        // 我们构造：
        //  - expert0: 全 1
        //  - expert1: 全 2
        // 这样 expert0 的输出是 32，expert1 的输出是 64（权重=1）
        let mut wdown = vec![f16_from_f32(0.0); E * HMID * H];
        for kk in 0..HMID {
            for j in 0..H {
                wdown[0 * (HMID * H) + kk * H + j] = f16_from_f32(1.0);
                wdown[1 * (HMID * H) + kk * H + j] = f16_from_f32(2.0);
            }
        }

        // experts_indicator：两个 expert 都开
        let experts_indicator = vec![true; E];

        // topk: 每个 token 的 topk 列表为 [0, 1]（升序）
        let mut topk = vec![0usize; B_CAP * KTOP];
        for b in 0..B_CAP {
            topk[b * KTOP + 0] = 0;
            topk[b * KTOP + 1] = 1;
        }

        // indice: [E, B_CAP]
        // 关键构造：
        //  - expert0: 只让 b=7,8 为 true（注意：这两个是“超出 B_RUN 的 token”）
        //  - expert1: 只让 b=5,6 为 true（这两个在 B_RUN 内）
        //
        // 正确实现（stride 用 B_CAP）：
        //  - expert1 只会写 token5/6 的 slot=1
        //  - token0/1 的 slot=1 必须保持 0
        //
        // 错误实现（stride 用 B_RUN=7）：
        //  - expert1 读 indice 时 base = 1*7 = 7
        //    它会把 expert0 的 b=7,8（正好在 indice[7],indice[8]）误当成 expert1 的 b=0,1
        //  => token0/1 会被错误写入 slot=1（出现非 0）
        let mut indice = vec![false; E * B_CAP];
        indice[0 * B_CAP + 7] = true;
        indice[0 * B_CAP + 8] = true;
        indice[1 * B_CAP + 5] = true;
        indice[1 * B_CAP + 6] = true;

        // weight: [E, B_CAP]，全 1
        let mut weight = vec![f16_from_f32(1.0); E * B_CAP];

        // output: [B_CAP, KTOP, H]，先清 0
        let mut out = vec![f16_from_f32(0.0); B_CAP * KTOP * H];

        // -----------------------
        // 运行
        // -----------------------
        let runner = unsafe {
            ExpertMatMulDown::<f16>::new(
                nonlin.as_ptr(),
                wdown.as_ptr(), // 如果你的 new() 现在接受的是 wdown_nt_ptr(已经转置)，那这里要传入转置后的；但你这份代码仍按 K×N 传进来更合理
                test_routing_from_dense(E, B_CAP, KTOP, &indice, &weight, &topk),
                out.as_mut_ptr(),
                E,
                B_CAP,
                HMID,
                H,
                KTOP,
                params,
                false,
            )
        };

        // 单线程即可复现问题
        runner.run(B_RUN, 0, 1, 0);

        // -----------------------
        // 断言：token0/1 的 expert1(slot=1) 必须保持 0
        // -----------------------
        for b in 0..2 {
            let slot = 1usize; // expert1 的位置
            for j in 0..H {
                let idx = (b * KTOP + slot) * H + j;
                let got = out[idx] as f32;
                assert!(
                    got.abs() <= 1e-3,
                    "stride bug: token {} slot1 should be 0, got {} (j={})",
                    b,
                    got,
                    j
                );
            }
        }

        // 再检查：token5/6 的 expert1(slot=1) 应该是非零（避免“啥都没算”假通过）
        // 期望值：sum_{kk=0..31} 1 * 2 = 64
        for b in 5..=6 {
            let slot = 1usize;
            for j in 0..H {
                let idx = (b * KTOP + slot) * H + j;
                let got = out[idx] as f32;
                assert!(
                    (got - 64.0).abs() <= 2.0, // fp16 累积误差给点余量
                    "expected token {} slot1 ~= 64, got {} (j={})",
                    b,
                    got,
                    j
                );
            }
        }
    }
    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_down_stride_must_use_capacity_not_run_batch_nt_weight() {
        use std::arch::is_x86_feature_detected;
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }

        use crate::num_traits::FromNumber;
        #[inline]
        fn f16_from_f32(x: f32) -> f16 {
            <f16 as FromNumber>::from_f32(x)
        }

        const E: usize = 2;
        const B_CAP: usize = 9; // capacity（构造时 num_token）
        const B_RUN: usize = 7; // run 时 batch_size
        const HMID: usize = 32;
        const H: usize = 32;
        const KTOP: usize = 2;

        let params = MatMulParams {
            a_row_step_macro: 6,   // MB
            b_row_step_macro: 32,  // NB
            column_step_macro: 16, // KC (HMID%KC==0)
            a_row_step_micro: 3,   // MR
            b_row_step_micro: 32,  // NR
        };

        // NONLIN[e,b,k]：全 1
        let nonlin = vec![f16_from_f32(1.0); E * B_CAP * HMID];

        // W_down_nt: [E, H, HMID]（你现在外部已经转置好就是这个）
        // 构造：
        //   expert0: 全 1
        //   expert1: 全 2
        // 则对任意输出列 j：dot = sum_k 1*W = 32 或 64（weight=1）
        let mut wdown_nt = vec![f16_from_f32(0.0); E * H * HMID];
        for j in 0..H {
            for kk in 0..HMID {
                wdown_nt[0 * (H * HMID) + j * HMID + kk] = f16_from_f32(1.0);
                wdown_nt[1 * (H * HMID) + j * HMID + kk] = f16_from_f32(2.0);
            }
        }

        let experts_indicator = vec![true; E];

        // topk: 每 token 的 topk 列表为 [0,1]（升序）
        let mut topk = vec![0usize; B_CAP * KTOP];
        for b in 0..B_CAP {
            topk[b * KTOP + 0] = 0;
            topk[b * KTOP + 1] = 1;
        }

        // indice: [E, B_CAP]
        // 关键构造（看解释）：
        let mut indice = vec![false; E * B_CAP];
        indice[0 * B_CAP + 7] = true;
        indice[0 * B_CAP + 8] = true;
        indice[1 * B_CAP + 5] = true;
        indice[1 * B_CAP + 6] = true;

        // weight: [E, B_CAP] 全 1
        let weight = vec![f16_from_f32(1.0); E * B_CAP];

        // output: [B_CAP, KTOP, H] 先清 0
        let mut out = vec![f16_from_f32(0.0); B_CAP * KTOP * H];

        let runner = unsafe {
            ExpertMatMulDown::<f16>::new(
                nonlin.as_ptr(),
                wdown_nt.as_ptr(), // ✅ 直接传 NT
                test_routing_from_dense(E, B_CAP, KTOP, &indice, &weight, &topk),
                out.as_mut_ptr(),
                E,
                B_CAP,
                HMID,
                H,
                KTOP,
                params,
                false,
            )
        };

        // 单线程足够复现
        runner.run(B_RUN, 0, 1, 0);

        // 断言1：token0/1 的 expert1(slot=1) 必须仍为 0
        for b in 0..2 {
            let slot = 1usize;
            for j in 0..H {
                let idx = (b * KTOP + slot) * H + j;
                let got = out[idx] as f32;
                assert!(
                    got.abs() <= 1e-3,
                    "stride bug: token {} slot1 should be 0, got {} (j={})",
                    b,
                    got,
                    j
                );
            }
        }

        // 断言2：token5/6 的 expert1(slot=1) 应该是非零，并且接近 64
        // ref: sum_k 1*2 = 64
        for b in 5..=6 {
            let slot = 1usize;
            for j in 0..H {
                let idx = (b * KTOP + slot) * H + j;
                let got = out[idx] as f32;
                assert!(
                    (got - 64.0).abs() <= 2.0,
                    "expected token {} slot1 ~= 64, got {} (j={})",
                    b,
                    got,
                    j
                );
            }
        }
    }

    #[test]
    fn test_down_uneven_expert_loads_many_threads() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }

        let num_experts = 3usize;
        let num_token = 13usize;
        let hmid = 32usize;
        let h = 64usize;
        let num_topk = 3usize;

        let params = MatMulParams {
            a_row_step_macro: 3,
            b_row_step_macro: 32,
            column_step_macro: 16,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let cpu_num = 8usize;

        let mut nonlin = vec![f16_from_f32(0.0); num_experts * num_token * hmid];
        let mut wdown = vec![f16_from_f32(0.0); num_experts * hmid * h];
        let mut out = vec![f16_from_f32(0.0); num_token * num_topk * h];

        let experts_indicator = vec![true; num_experts];
        let mut indice = vec![false; num_experts * num_token];
        let mut weight = vec![f16_from_f32(0.0); num_experts * num_token];
        let mut topk = vec![0usize; num_token * num_topk];

        for b in 0..num_token {
            let row = &mut topk[b * num_topk..(b + 1) * num_topk];
            row.copy_from_slice(&[0, 1, 2]);
        }

        for &b in &[0usize, 1, 3, 4, 7, 9, 12] {
            indice[0 * num_token + b] = true;
            weight[0 * num_token + b] = f16_from_f32(0.4 + b as f32 * 0.01);
        }
        for &b in &[2usize, 10] {
            indice[1 * num_token + b] = true;
            weight[1 * num_token + b] = f16_from_f32(0.7 + b as f32 * 0.01);
        }
        indice[2 * num_token + 5] = true;
        weight[2 * num_token + 5] = f16_from_f32(1.1);

        for e in 0..num_experts {
            for b in 0..num_token {
                for kk in 0..hmid {
                    nonlin[(e * num_token + b) * hmid + kk] =
                        f16_from_f32(0.01 * e as f32 + 0.02 * b as f32 + 0.001 * kk as f32);
                }
            }
        }
        for e in 0..num_experts {
            for kk in 0..hmid {
                for j in 0..h {
                    wdown[(e * hmid + kk) * h + j] =
                        f16_from_f32(0.003 * e as f32 + 0.001 * kk as f32 + 0.0004 * j as f32);
                }
            }
        }

        let runner = unsafe {
            ExpertMatMulDown::<f16>::new(
                nonlin.as_ptr(),
                wdown.as_ptr(),
                test_routing_from_dense(num_experts, num_token, num_topk, &indice, &weight, &topk),
                out.as_mut_ptr(),
                num_experts,
                num_token,
                hmid,
                h,
                num_topk,
                params,
                false,
            )
        };

        for tid in 0..cpu_num {
            runner.run(num_token, 0, cpu_num, tid);
        }

        let mut out_ref = vec![0.0f32; num_token * num_topk * h];
        for e in 0..num_experts {
            for b in 0..num_token {
                if !indice[e * num_token + b] {
                    continue;
                }
                let slot = slot_of(&topk, b, num_topk, e);
                let w = f32_from_f16(weight[e * num_token + b]);
                for j in 0..h {
                    let mut acc = 0.0f32;
                    for kk in 0..hmid {
                        let a = f32_from_f16(nonlin[(e * num_token + b) * hmid + kk]);
                        let bv = f32_from_f16(wdown[(e * hmid + kk) * h + j]);
                        acc += a * bv;
                    }
                    out_ref[(b * num_topk + slot) * h + j] += w * acc;
                }
            }
        }

        verify_output(&out, &out_ref, 8e-2, "uneven_expert_loads");
    }

    #[test]
    fn test_down_more_threads_than_tasks() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }

        let num_experts = 2usize;
        let num_token = 3usize;
        let hmid = 32usize;
        let h = 64usize;
        let num_topk = 2usize;

        let params = MatMulParams {
            a_row_step_macro: 3,
            b_row_step_macro: 32,
            column_step_macro: 16,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let cpu_num = 16usize;

        let mut nonlin = vec![f16_from_f32(0.0); num_experts * num_token * hmid];
        let mut wdown = vec![f16_from_f32(0.0); num_experts * hmid * h];
        let mut out = vec![f16_from_f32(0.0); num_token * num_topk * h];

        let experts_indicator = vec![true, false];
        let mut indice = vec![false; num_experts * num_token];
        let mut weight = vec![f16_from_f32(0.0); num_experts * num_token];
        let mut topk = vec![0usize; num_token * num_topk];

        for b in 0..num_token {
            indice[b] = true;
            weight[b] = f16_from_f32(0.5 + 0.1 * b as f32);
            let row = &mut topk[b * num_topk..(b + 1) * num_topk];
            row.copy_from_slice(&[0, 1]);
        }

        for b in 0..num_token {
            for kk in 0..hmid {
                nonlin[b * hmid + kk] = f16_from_f32(0.02 * b as f32 + 0.001 * kk as f32);
            }
        }
        for kk in 0..hmid {
            for j in 0..h {
                wdown[kk * h + j] = f16_from_f32(0.0015 * kk as f32 + 0.0003 * j as f32);
            }
        }

        let runner = unsafe {
            ExpertMatMulDown::<f16>::new(
                nonlin.as_ptr(),
                wdown.as_ptr(),
                test_routing_from_dense(num_experts, num_token, num_topk, &indice, &weight, &topk),
                out.as_mut_ptr(),
                num_experts,
                num_token,
                hmid,
                h,
                num_topk,
                params,
                false,
            )
        };

        for tid in 0..cpu_num {
            runner.run(num_token, 0, cpu_num, tid);
        }

        let mut out_ref = vec![0.0f32; num_token * num_topk * h];
        for b in 0..num_token {
            let w = f32_from_f16(weight[b]);
            for j in 0..h {
                let mut acc = 0.0f32;
                for kk in 0..hmid {
                    let a = f32_from_f16(nonlin[b * hmid + kk]);
                    let bv = f32_from_f16(wdown[kk * h + j]);
                    acc += a * bv;
                }
                out_ref[b * num_topk * h + j] += w * acc;
            }
        }

        verify_output(&out, &out_ref, 8e-2, "more_threads_than_tasks");
    }

    #[test]
    fn test_down_active_expert_with_zero_tokens_keeps_output_zero() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }

        let num_experts = 3usize;
        let num_token = 6usize;
        let hmid = 32usize;
        let h = 64usize;
        let num_topk = 3usize;

        let params = MatMulParams {
            a_row_step_macro: 3,
            b_row_step_macro: 32,
            column_step_macro: 16,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let mut nonlin = vec![f16_from_f32(0.0); num_experts * num_token * hmid];
        let mut wdown = vec![f16_from_f32(0.0); num_experts * hmid * h];
        let mut out = vec![f16_from_f32(0.0); num_token * num_topk * h];

        let experts_indicator = vec![true, true, false];
        let mut indice = vec![false; num_experts * num_token];
        let mut weight = vec![f16_from_f32(0.0); num_experts * num_token];
        let mut topk = vec![0usize; num_token * num_topk];

        for b in 0..num_token {
            let row = &mut topk[b * num_topk..(b + 1) * num_topk];
            row.copy_from_slice(&[0, 1, 2]);
        }
        for &b in &[0usize, 2, 4] {
            indice[0 * num_token + b] = true;
            weight[0 * num_token + b] = f16_from_f32(0.8);
        }

        for e in 0..num_experts {
            for b in 0..num_token {
                for kk in 0..hmid {
                    nonlin[(e * num_token + b) * hmid + kk] =
                        f16_from_f32(0.01 * e as f32 + 0.02 * b as f32 + 0.001 * kk as f32);
                }
            }
        }
        for e in 0..num_experts {
            for kk in 0..hmid {
                for j in 0..h {
                    wdown[(e * hmid + kk) * h + j] =
                        f16_from_f32(0.002 * e as f32 + 0.0005 * kk as f32 + 0.0002 * j as f32);
                }
            }
        }

        let runner = unsafe {
            ExpertMatMulDown::<f16>::new(
                nonlin.as_ptr(),
                wdown.as_ptr(),
                test_routing_from_dense(num_experts, num_token, num_topk, &indice, &weight, &topk),
                out.as_mut_ptr(),
                num_experts,
                num_token,
                hmid,
                h,
                num_topk,
                params,
                false,
            )
        };

        for tid in 0..4usize {
            runner.run(num_token, 0, 4, tid);
        }

        for b in 0..num_token {
            let slot = slot_of(&topk, b, num_topk, 1);
            for j in 0..h {
                let v = f32_from_f16(out[(b * num_topk + slot) * h + j]);
                assert!(
                    approx_eq_f32(v, 0.0, 1e-3),
                    "active expert with zero tokens wrote output at b={}, j={}, got={}",
                    b,
                    j,
                    v
                );
            }
        }
    }
}
