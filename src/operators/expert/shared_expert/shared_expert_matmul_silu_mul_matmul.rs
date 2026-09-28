// === operators/expert/shared_expert/shared_expert_matmul_silu_mul_matmul.rs ===
#![allow(non_snake_case)]

use std::marker::PhantomData;
use std::ops::{Add, Mul};
use std::sync::atomic::Ordering;

use crate::kernel::common::matmul_params::MatMulParams;
use crate::num_traits::Sigmoid;

use crate::operators::assign::assign;
use crate::operators::expert::expert_routing::{task_assign, ExpertRouting, ExpertTaskMeta};
use crate::operators::send_sync_ptr::{ConstPtr, MutPtr};
use crate::operators::traits::SharedExpertsSiluTrait;

// Variable naming used in this operator:
// - token_block_rows / token_block_start: routed-token macro block inside one expert (sparse
//   branch) or a dense token macro block (shared branch).
// - output_cols / output_col_start: intermediate I (routed) / Is (shared) columns produced by
//   gate/up projections.
// - reduction_cols / reduction_col_start: hidden H dimension reduced by GEMM.
// - micro_tile_rows / micro_tile_cols: micro-kernel tile size.
// - gate_acc / up_acc: per-thread accumulators before SiLU(gate) * up.
// 本算子的变量命名约定：
// - token_block_rows / token_block_start：sparse 分支单个 expert 内的 routed token 宏块，
//   或 shared 分支的稠密 token 宏块。
// - output_cols / output_col_start：gate/up 投影产生的 intermediate I（routed）/ Is（shared）列。
// - reduction_cols / reduction_col_start：GEMM 中被规约的 hidden H 维度。
// - micro_tile_rows / micro_tile_cols：微内核 tile 大小。
// - gate_acc / up_acc：执行 SiLU(gate) * up 前的每线程累加器。

/// Shared expert + routed experts gate/up projection fused with SiLU(gate) * up.
/// shared expert 与 routed experts 的 gate/up 投影，融合 SiLU(gate) * up。
///
/// run() executes two phases in a fixed order:
///   Phase A (shared, dense): x[b] over ALL tokens × shared gate/up → shared_out[b, Is].
///   Phase B (routed, sparse): routed tokens per expert × routed gate/up → out[e, b, I].
/// run() 按固定顺序执行两阶段：
///   阶段 A（shared，稠密）：所有 token 的 x[b] × shared gate/up → shared_out[b, Is]。
///   阶段 B（routed，稀疏）：每个 expert 的 routed token × routed gate/up → out[e, b, I]。
///
/// Compute is plain scalar Rust; no f16 / AVX-512 specialization yet.
/// compute 为普通标量 Rust；暂不做 f16 / AVX-512 特化。
#[derive(Clone)]
pub struct SharedExpertMatMulSilu<T> {
    pub input_ptr: ConstPtr<T>, // Input hidden states: [B,H]. 输入 hidden states。

    // Routed gate/up weights, NT layout: [E][I x H], row stride = H.
    // routed gate/up 权重，NT 布局：[E][I x H]，行距为 H。
    pub gate_nt_ptr: ConstPtr<T>,
    pub up_nt_ptr: ConstPtr<T>,

    // Shared gate/up weights, NT layout: [Is x H], row stride = H.
    // shared gate/up 权重，NT 布局：[Is x H]，行距为 H。
    pub shared_gate_nt_ptr: ConstPtr<T>,
    pub shared_up_nt_ptr: ConstPtr<T>,

    pub routing: ExpertRouting<T>,

    pub output_ptr: MutPtr<T>, // Routed nonlinear output: [E,B,I]. routed 非线性输出。
    pub shared_output_ptr: MutPtr<T>, // Shared nonlinear output: [B,Is]. shared 非线性输出。

    pub params: MatMulParams,

    pub batch: usize,        // Token capacity. token 容量。
    pub inter: usize,        // Routed intermediate size. routed intermediate 大小。
    pub shared_inter: usize, // Shared intermediate size. shared intermediate 大小。
    pub hidden: usize,       // Hidden size. hidden 大小。
    pub num_experts: usize,  // Expert count. expert 数量。
    pub decode_only_flag: bool,

    // === strides ===
    pub packed_panel_stride: usize, // reduction_block_cols * micro_tile_cols
    pub acc_stride: usize,          // micro_tile_rows * micro_tile_cols
    pub a_tile_stride: usize,       // micro_tile_rows * reduction_block_cols

    // === prepacked weights (packed once in new()) ===
    pub packed_gate: Box<[T]>,
    pub packed_up: Box<[T]>,
    pub packed_shared_gate: Box<[T]>,
    pub packed_shared_up: Box<[T]>,

    // === per-thread scratch pools, reused by run() without allocation ===
    pub gate_acc_pool: Box<[T]>,
    pub up_acc_pool: Box<[T]>,
    pub a_tile_pool: Box<[T]>,
    pub idx_buf_pool: Box<[usize]>,

    task_meta_pool: Box<[ExpertTaskMeta]>,
    task_meta_stride: usize, // num_experts
    routed_tokens_pool: Box<[usize]>,
    routed_stride: usize, // num_experts * capacity_per_expert

    _marker: PhantomData<T>,
}

impl<T> SharedExpertMatMulSilu<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default + Sigmoid,
{
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        input_ptr: *const T,          // Input hidden states: [B,H].
        gate_nt_ptr: *const T,        // Routed W_gate_nt[E,I,H].
        up_nt_ptr: *const T,          // Routed W_up_nt[E,I,H].
        shared_gate_nt_ptr: *const T, // Shared W_gate_nt[Is,H].
        shared_up_nt_ptr: *const T,   // Shared W_up_nt[Is,H].
        routing: ExpertRouting<T>,
        output_ptr: *mut T,        // Routed nonlinear output: [E,B,I].
        shared_output_ptr: *mut T, // Shared nonlinear output: [B,Is].
        batch: usize,
        inter: usize,
        shared_inter: usize,
        hidden: usize,
        num_experts: usize,
        a_row_step_macro: usize,
        b_row_step_macro: usize,
        column_step_macro: usize,
        a_row_step_micro: usize,
        b_row_step_micro: usize,
        decode_only_flag: bool,
    ) -> Self {
        let token_block_rows = a_row_step_macro.max(1);
        let reduction_block_cols = column_step_macro.max(1);
        let micro_tile_rows = a_row_step_micro.max(1);
        let micro_tile_cols = b_row_step_micro.max(1);

        let packed_panel_stride = reduction_block_cols * micro_tile_cols;
        let acc_stride = micro_tile_rows * micro_tile_cols;
        let a_tile_stride = micro_tile_rows * reduction_block_cols;

        let packed_gate = Self::pack_expert_b_panels(
            gate_nt_ptr,
            num_experts,
            inter,
            hidden,
            reduction_block_cols,
            micro_tile_cols,
        );
        let packed_up = Self::pack_expert_b_panels(
            up_nt_ptr,
            num_experts,
            inter,
            hidden,
            reduction_block_cols,
            micro_tile_cols,
        );
        // Shared expert = a single dense "expert" (expert_count = 1).
        // shared expert 视作单个稠密 "expert"（expert_count = 1）。
        let packed_shared_gate = Self::pack_expert_b_panels(
            shared_gate_nt_ptr,
            1,
            shared_inter,
            hidden,
            reduction_block_cols,
            micro_tile_cols,
        );
        let packed_shared_up = Self::pack_expert_b_panels(
            shared_up_nt_ptr,
            1,
            shared_inter,
            hidden,
            reduction_block_cols,
            micro_tile_cols,
        );

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(16);

        let gate_acc_pool = vec![T::default(); threads * acc_stride].into_boxed_slice();
        let up_acc_pool = vec![T::default(); threads * acc_stride].into_boxed_slice();
        let a_tile_pool = vec![T::default(); threads * a_tile_stride].into_boxed_slice();
        let idx_buf_pool = vec![0usize; threads * token_block_rows].into_boxed_slice();

        let task_meta_stride = num_experts;
        let routed_stride = num_experts * routing.capacity_per_expert;
        let task_meta_pool =
            vec![ExpertTaskMeta::default(); threads * task_meta_stride].into_boxed_slice();
        let routed_tokens_pool = vec![0usize; threads * routed_stride].into_boxed_slice();

        Self {
            input_ptr: ConstPtr { ptr: input_ptr },
            gate_nt_ptr: ConstPtr { ptr: gate_nt_ptr },
            up_nt_ptr: ConstPtr { ptr: up_nt_ptr },
            shared_gate_nt_ptr: ConstPtr {
                ptr: shared_gate_nt_ptr,
            },
            shared_up_nt_ptr: ConstPtr {
                ptr: shared_up_nt_ptr,
            },

            routing,
            output_ptr: MutPtr { ptr: output_ptr },
            shared_output_ptr: MutPtr {
                ptr: shared_output_ptr,
            },

            params: MatMulParams {
                a_row_step_macro,
                b_row_step_macro,
                column_step_macro,
                a_row_step_micro,
                b_row_step_micro,
            },

            batch,
            inter,
            shared_inter,
            hidden,
            num_experts,
            decode_only_flag,

            packed_panel_stride,
            acc_stride,
            a_tile_stride,

            packed_gate,
            packed_up,
            packed_shared_gate,
            packed_shared_up,

            gate_acc_pool,
            up_acc_pool,
            a_tile_pool,
            idx_buf_pool,

            task_meta_pool,
            task_meta_stride,
            routed_tokens_pool,
            routed_stride,

            _marker: PhantomData,
        }
    }

    #[inline(always)]
    fn thread_slices(&self, tid: usize) -> (*mut T, *mut T, *mut T, *mut usize) {
        unsafe {
            let ga = self.gate_acc_pool.as_ptr().add(tid * self.acc_stride) as *mut T;
            let ua = self.up_acc_pool.as_ptr().add(tid * self.acc_stride) as *mut T;
            let at = self.a_tile_pool.as_ptr().add(tid * self.a_tile_stride) as *mut T;
            let idx = self
                .idx_buf_pool
                .as_ptr()
                .add(tid * self.params.a_row_step_macro.max(1)) as *mut usize;
            (ga, ua, at, idx)
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

    /// Resolve a packed panel by explicit output/reduction column counts so the
    /// same helper serves both routed (inter, hidden) and shared (shared_inter,
    /// hidden) weights.
    /// 通过显式的 output/reduction 列数解析 packed panel，使同一 helper 同时服务
    /// routed（inter, hidden）与 shared（shared_inter, hidden）权重。
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

    /// Pack routed/dense tokens into a micro input tile and zero-pad unused rows.
    /// 将 routed/dense token 收集到微内核输入 tile，未使用的行补零。
    #[inline(always)]
    pub unsafe fn pack_a_tile_mrkc(
        input_base: *const T, // [B,H]
        input_row_stride: usize,
        routed_token_indices: *const usize,
        idx_off: usize,
        valid_rows: usize,
        reduction_col_start: usize,
        reduction_block_cols: usize,
        output_tile: *mut T,
        micro_tile_rows: usize,
    ) {
        for tile_index in 0..(micro_tile_rows * reduction_block_cols) {
            *output_tile.add(tile_index) = T::default();
        }
        for row_in_tile in 0..valid_rows {
            let token_id = *routed_token_indices.add(idx_off + row_in_tile);
            let source_row = input_base.add(token_id * input_row_stride + reduction_col_start);
            let packed_row = output_tile.add(row_in_tile * reduction_block_cols);
            for reduction_lane in 0..reduction_block_cols {
                *packed_row.add(reduction_lane) = *source_row.add(reduction_lane);
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
    ) -> (&[ExpertTaskMeta], &[usize], usize) {
        let expert_tasks_ptr =
            self.task_meta_pool
                .as_ptr()
                .wrapping_add(thread_id * self.task_meta_stride) as *mut ExpertTaskMeta;
        let routed_tokens_ptr =
            self.routed_tokens_pool
                .as_ptr()
                .wrapping_add(thread_id * self.routed_stride) as *mut usize;
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
                    let offset = self.routing.expert_offset(expert_id, expert_queue_pos);
                    *routed_tokens_ptr.add(routed_count) =
                        *self.routing.index_tensor.ptr.add(offset);
                    routed_count += 1;
                }

                let token_tile_count = routed_token_count.div_ceil(token_block_rows);
                let task_count = token_tile_count * output_column_tile_count;
                *expert_tasks_ptr.add(expert_task_count) = ExpertTaskMeta {
                    expert_id,
                    token_begin,
                    sequence_length: routed_token_count,
                    task_begin: total_tasks,
                    task_end: total_tasks + task_count,
                };
                expert_task_count += 1;
                total_tasks += task_count;
            }

            (
                std::slice::from_raw_parts(expert_tasks_ptr, expert_task_count),
                std::slice::from_raw_parts(routed_tokens_ptr, routed_count),
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

            let reduction_cols = self.hidden;
            let token_block_rows = self.params.a_row_step_macro.max(1);
            let output_block_cols = self.params.b_row_step_macro.max(1);
            let reduction_block_cols = self.params.column_step_macro.max(1);
            let micro_tile_rows = self.params.a_row_step_micro.max(1);
            let micro_tile_cols = self.params.b_row_step_micro.max(1);

            debug_assert!(reduction_cols % reduction_block_cols == 0);
            debug_assert!(self.inter % micro_tile_cols == 0);
            debug_assert!(self.shared_inter % micro_tile_cols == 0);

            let input_base = self.input_ptr.ptr;
            let input_row_stride = self.hidden;

            let (gate_acc, up_acc, a_tile, idx_buf) = self.thread_slices(thread_id);

            /* ---------------- Phase A: shared expert (dense) ---------------- */
            /* ---------------- 阶段 A：shared expert（稠密） ---------------- */
            let shared_output_cols = self.shared_inter;
            let shared_output_base = self.shared_output_ptr.ptr;
            let shared_output_column_tile_count = shared_output_cols.div_ceil(output_block_cols);
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
                        (shared_output_cols - output_col_start).min(output_block_cols);
                    if output_cols_in_block == 0 || tokens_in_block == 0 {
                        continue;
                    }

                    let mut output_col_offset = 0usize;
                    while output_col_offset < output_cols_in_block {
                        let mut token_offset_in_block = 0usize;
                        while token_offset_in_block < tokens_in_block {
                            let valid_rows =
                                (tokens_in_block - token_offset_in_block).min(micro_tile_rows);

                            // Dense: token ids are contiguous starting from token_block_start.
                            // 稠密：token id 从 token_block_start 起连续。
                            for row_in_tile in 0..valid_rows {
                                *idx_buf.add(row_in_tile) =
                                    token_block_start + token_offset_in_block + row_in_tile;
                            }

                            for accumulator_index in 0..(micro_tile_rows * micro_tile_cols) {
                                *gate_acc.add(accumulator_index) = T::default();
                                *up_acc.add(accumulator_index) = T::default();
                            }

                            let mut reduction_col_start = 0usize;
                            while reduction_col_start < reduction_cols {
                                let gate_panel = self.packed_panel_ptr_dim(
                                    &self.packed_shared_gate,
                                    0,
                                    shared_output_cols,
                                    reduction_cols,
                                    output_col_start + output_col_offset,
                                    reduction_col_start,
                                );
                                let up_panel = self.packed_panel_ptr_dim(
                                    &self.packed_shared_up,
                                    0,
                                    shared_output_cols,
                                    reduction_cols,
                                    output_col_start + output_col_offset,
                                    reduction_col_start,
                                );

                                if valid_rows == 1 {
                                    let token_id = *idx_buf;
                                    let input_row = input_base
                                        .add(token_id * input_row_stride + reduction_col_start);
                                    self.compute1_single(
                                        input_row,
                                        gate_panel,
                                        up_panel,
                                        gate_acc,
                                        up_acc,
                                        reduction_block_cols,
                                    );
                                } else if valid_rows < micro_tile_rows {
                                    Self::pack_a_tile_mrkc(
                                        input_base,
                                        input_row_stride,
                                        idx_buf,
                                        0,
                                        valid_rows,
                                        reduction_col_start,
                                        reduction_block_cols,
                                        a_tile,
                                        micro_tile_rows,
                                    );
                                    self.compute1_rows(
                                        a_tile as *const T,
                                        gate_panel,
                                        up_panel,
                                        gate_acc,
                                        up_acc,
                                        reduction_block_cols,
                                        valid_rows,
                                    );
                                } else {
                                    Self::pack_a_tile_mrkc(
                                        input_base,
                                        input_row_stride,
                                        idx_buf,
                                        0,
                                        valid_rows,
                                        reduction_col_start,
                                        reduction_block_cols,
                                        a_tile,
                                        micro_tile_rows,
                                    );
                                    self.compute1(
                                        a_tile as *const T,
                                        gate_panel,
                                        up_panel,
                                        gate_acc,
                                        up_acc,
                                        reduction_block_cols,
                                    );
                                }

                                reduction_col_start += reduction_block_cols;
                            }

                            for row_in_tile in 0..valid_rows {
                                let token_id = *idx_buf.add(row_in_tile);
                                let out_row = shared_output_base
                                    .add(token_id * self.shared_inter)
                                    .add(output_col_start + output_col_offset);
                                let gate_row = gate_acc.add(row_in_tile * micro_tile_cols);
                                let up_row = up_acc.add(row_in_tile * micro_tile_cols);
                                self.compute2(gate_row as *const T, up_row as *const T, out_row);
                            }

                            token_offset_in_block += valid_rows;
                        }

                        output_col_offset += micro_tile_cols;
                    }
                }
            }

            /* ---------------- Phase B: routed experts (sparse) ---------------- */
            /* ---------------- 阶段 B：routed experts（稀疏） ---------------- */
            let output_cols = self.inter;
            let routed_output_base = self.output_ptr.ptr;
            let output_expert_stride = self.batch * self.inter;

            let output_column_tile_count = output_cols.div_ceil(output_block_cols);
            let (expert_tasks, routed_tokens, total_tasks) = self.build_task_space(
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

                    let output_col_start = output_tile_id * output_block_cols;
                    let output_cols_in_block =
                        (output_cols - output_col_start).min(output_block_cols);
                    if output_cols_in_block == 0 {
                        continue;
                    }

                    let token_block_start = token_tile_id * token_block_rows;
                    let tokens_in_block =
                        (task_meta.sequence_length - token_block_start).min(token_block_rows);
                    debug_assert!(tokens_in_block > 0);

                    let token_slice = &routed_tokens[(task_meta.token_begin + token_block_start)
                        ..(task_meta.token_begin + token_block_start + tokens_in_block)];
                    for (buffer_offset, &token_id) in token_slice.iter().enumerate() {
                        *idx_buf.add(buffer_offset) = token_id;
                    }

                    let expert_id = task_meta.expert_id;

                    let mut output_col_offset = 0usize;
                    while output_col_offset < output_cols_in_block {
                        let mut token_offset_in_block = 0usize;
                        while token_offset_in_block < tokens_in_block {
                            let valid_rows =
                                (tokens_in_block - token_offset_in_block).min(micro_tile_rows);

                            for accumulator_index in 0..(micro_tile_rows * micro_tile_cols) {
                                *gate_acc.add(accumulator_index) = T::default();
                                *up_acc.add(accumulator_index) = T::default();
                            }

                            let mut reduction_col_start = 0usize;
                            while reduction_col_start < reduction_cols {
                                let gate_panel = self.packed_panel_ptr_dim(
                                    &self.packed_gate,
                                    expert_id,
                                    output_cols,
                                    reduction_cols,
                                    output_col_start + output_col_offset,
                                    reduction_col_start,
                                );
                                let up_panel = self.packed_panel_ptr_dim(
                                    &self.packed_up,
                                    expert_id,
                                    output_cols,
                                    reduction_cols,
                                    output_col_start + output_col_offset,
                                    reduction_col_start,
                                );

                                if valid_rows == 1 {
                                    let token_id = *idx_buf.add(token_offset_in_block);
                                    let input_row = input_base
                                        .add(token_id * input_row_stride + reduction_col_start);
                                    self.compute1_single(
                                        input_row,
                                        gate_panel,
                                        up_panel,
                                        gate_acc,
                                        up_acc,
                                        reduction_block_cols,
                                    );
                                } else if valid_rows < micro_tile_rows {
                                    Self::pack_a_tile_mrkc(
                                        input_base,
                                        input_row_stride,
                                        idx_buf,
                                        token_offset_in_block,
                                        valid_rows,
                                        reduction_col_start,
                                        reduction_block_cols,
                                        a_tile,
                                        micro_tile_rows,
                                    );
                                    self.compute1_rows(
                                        a_tile as *const T,
                                        gate_panel,
                                        up_panel,
                                        gate_acc,
                                        up_acc,
                                        reduction_block_cols,
                                        valid_rows,
                                    );
                                } else {
                                    Self::pack_a_tile_mrkc(
                                        input_base,
                                        input_row_stride,
                                        idx_buf,
                                        token_offset_in_block,
                                        valid_rows,
                                        reduction_col_start,
                                        reduction_block_cols,
                                        a_tile,
                                        micro_tile_rows,
                                    );
                                    self.compute1(
                                        a_tile as *const T,
                                        gate_panel,
                                        up_panel,
                                        gate_acc,
                                        up_acc,
                                        reduction_block_cols,
                                    );
                                }

                                reduction_col_start += reduction_block_cols;
                            }

                            for row_in_tile in 0..valid_rows {
                                let token_id = *idx_buf.add(token_offset_in_block + row_in_tile);
                                let output_row = routed_output_base
                                    .add(expert_id * output_expert_stride)
                                    .add(token_id * self.inter)
                                    .add(output_col_start + output_col_offset);
                                let gate_row = gate_acc.add(row_in_tile * micro_tile_cols);
                                let up_row = up_acc.add(row_in_tile * micro_tile_cols);
                                self.compute2(
                                    gate_row as *const T,
                                    up_row as *const T,
                                    output_row as *mut T,
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

/* -------------------- SharedExpertsSiluTrait scalar implementation -------------------- */
/* -------------------- SharedExpertsSiluTrait 标量实现 -------------------- */

impl<T> SharedExpertsSiluTrait<T> for SharedExpertMatMulSilu<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default + Sigmoid,
{
    fn compute1(
        &self,
        a_tile: *const T,
        gate_panel: *const T,
        up_panel: *const T,
        gate_acc: *mut T,
        up_acc: *mut T,
        kc: usize,
    ) {
        unsafe {
            let micro_tile_rows = self.params.a_row_step_micro.max(1);
            let micro_tile_cols = self.params.b_row_step_micro.max(1);
            for row_in_tile in 0..micro_tile_rows {
                for col_in_tile in 0..micro_tile_cols {
                    let mut gate = *gate_acc.add(row_in_tile * micro_tile_cols + col_in_tile);
                    let mut up = *up_acc.add(row_in_tile * micro_tile_cols + col_in_tile);
                    for reduction_lane in 0..kc {
                        let input = *a_tile.add(row_in_tile * kc + reduction_lane);
                        gate = gate
                            + input
                                * *gate_panel.add(reduction_lane * micro_tile_cols + col_in_tile);
                        up = up
                            + input * *up_panel.add(reduction_lane * micro_tile_cols + col_in_tile);
                    }
                    *gate_acc.add(row_in_tile * micro_tile_cols + col_in_tile) = gate;
                    *up_acc.add(row_in_tile * micro_tile_cols + col_in_tile) = up;
                }
            }
        }
    }

    fn compute1_single(
        &self,
        input_row: *const T,
        gate_panel: *const T,
        up_panel: *const T,
        gate_acc: *mut T,
        up_acc: *mut T,
        kc: usize,
    ) {
        unsafe {
            let micro_tile_cols = self.params.b_row_step_micro.max(1);
            for col_in_tile in 0..micro_tile_cols {
                let mut gate = *gate_acc.add(col_in_tile);
                let mut up = *up_acc.add(col_in_tile);
                for reduction_lane in 0..kc {
                    let input = *input_row.add(reduction_lane);
                    gate = gate
                        + input * *gate_panel.add(reduction_lane * micro_tile_cols + col_in_tile);
                    up = up + input * *up_panel.add(reduction_lane * micro_tile_cols + col_in_tile);
                }
                *gate_acc.add(col_in_tile) = gate;
                *up_acc.add(col_in_tile) = up;
            }
        }
    }

    fn compute1_rows(
        &self,
        a_tile: *const T,
        gate_panel: *const T,
        up_panel: *const T,
        gate_acc: *mut T,
        up_acc: *mut T,
        kc: usize,
        rows: usize,
    ) {
        unsafe {
            let micro_tile_cols = self.params.b_row_step_micro.max(1);
            for row_in_tile in 0..rows {
                for col_in_tile in 0..micro_tile_cols {
                    let mut gate = *gate_acc.add(row_in_tile * micro_tile_cols + col_in_tile);
                    let mut up = *up_acc.add(row_in_tile * micro_tile_cols + col_in_tile);
                    for reduction_lane in 0..kc {
                        let input = *a_tile.add(row_in_tile * kc + reduction_lane);
                        gate = gate
                            + input
                                * *gate_panel.add(reduction_lane * micro_tile_cols + col_in_tile);
                        up = up
                            + input * *up_panel.add(reduction_lane * micro_tile_cols + col_in_tile);
                    }
                    *gate_acc.add(row_in_tile * micro_tile_cols + col_in_tile) = gate;
                    *up_acc.add(row_in_tile * micro_tile_cols + col_in_tile) = up;
                }
            }
        }
    }

    /// c_row[j] = SiLU(gate_row[j]) * up_row[j], SiLU(x) = x * sigmoid(x).
    /// c_row[j] = SiLU(gate_row[j]) * up_row[j]，SiLU(x) = x * sigmoid(x)。
    fn compute2(&self, gate_row: *const T, up_row: *const T, c_row: *mut T) {
        unsafe {
            let micro_tile_cols = self.params.b_row_step_micro.max(1);
            for col_in_tile in 0..micro_tile_cols {
                let gate = *gate_row.add(col_in_tile);
                let up = *up_row.add(col_in_tile);
                let silu = gate * gate.sigmoid();
                *c_row.add(col_in_tile) = silu * up;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::expert::expert_routing::routing_from_dense;

    #[inline]
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    // Transpose per-expert K×N ([E,H,I]) to NT ([E,I,H]).
    // 将每个 expert 的 K×N（[E,H,I]）转置为 NT（[E,I,H]）。
    fn transpose_kxn_to_nt(src: &[f32], e: usize, h: usize, i: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; e * i * h];
        for ex in 0..e {
            for kk in 0..h {
                for ii in 0..i {
                    out[ex * (i * h) + ii * h + kk] = src[ex * (h * i) + kk * i + ii];
                }
            }
        }
        out
    }

    // Shared dims: hidden%kc==0, inter%nr==0, shared_inter%nr==0, nb%nr==0.
    // 共享维度：hidden%kc==0, inter%nr==0, shared_inter%nr==0, nb%nr==0。
    const B: usize = 5;
    const H: usize = 16;
    const I: usize = 8; // routed inter
    const IS: usize = 12; // shared inter
    const E: usize = 2;
    const MB: usize = 3;
    const NB: usize = 4;
    const KC: usize = 8;
    const MR: usize = 3;
    const NR: usize = 4;

    fn build_and_run(cpu_num: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<bool>) {
        let mut x = vec![0.0f32; B * H];
        for b in 0..B {
            for k in 0..H {
                x[b * H + k] = 0.01 * b as f32 + 0.002 * k as f32 - 0.03;
            }
        }

        let mut wg = vec![0.0f32; E * H * I];
        let mut wu = vec![0.0f32; E * H * I];
        for e in 0..E {
            for k in 0..H {
                for i in 0..I {
                    wg[e * (H * I) + k * I + i] =
                        0.003 * e as f32 + 0.002 * k as f32 + 0.001 * i as f32 - 0.02;
                    wu[e * (H * I) + k * I + i] =
                        0.001 * e as f32 - 0.0015 * k as f32 + 0.002 * i as f32 + 0.01;
                }
            }
        }
        let wg_nt = transpose_kxn_to_nt(&wg, E, H, I);
        let wu_nt = transpose_kxn_to_nt(&wu, E, H, I);

        let mut sg = vec![0.0f32; H * IS];
        let mut su = vec![0.0f32; H * IS];
        for k in 0..H {
            for i in 0..IS {
                sg[k * IS + i] = 0.002 * k as f32 + 0.001 * i as f32 - 0.01;
                su[k * IS + i] = -0.001 * k as f32 + 0.002 * i as f32 + 0.005;
            }
        }
        let sg_nt = transpose_kxn_to_nt(&sg, 1, H, IS);
        let su_nt = transpose_kxn_to_nt(&su, 1, H, IS);

        let num_topk = 1usize;
        let mut indice = vec![false; E * B];
        let score = vec![1.0f32; E * B];
        let mut topk = vec![0usize; B * num_topk];
        for &b in &[0usize, 1, 2] {
            indice[0 * B + b] = true;
            topk[b] = 0;
        }
        for &b in &[2usize, 3] {
            indice[1 * B + b] = true;
        }

        let mut routed_out = vec![0.0f32; E * B * I];
        let mut shared_out = vec![0.0f32; B * IS];

        unsafe {
            let routing = routing_from_dense(
                E,
                B,
                num_topk,
                indice.as_ptr(),
                score.as_ptr(),
                topk.as_ptr(),
            );
            let op = SharedExpertMatMulSilu::<f32>::new(
                x.as_ptr(),
                wg_nt.as_ptr(),
                wu_nt.as_ptr(),
                sg_nt.as_ptr(),
                su_nt.as_ptr(),
                routing,
                routed_out.as_mut_ptr(),
                shared_out.as_mut_ptr(),
                B,
                I,
                IS,
                H,
                E,
                MB,
                NB,
                KC,
                MR,
                NR,
                false,
            );
            for tid in 0..cpu_num {
                op.run(B, 0, B, B, cpu_num, tid);
            }
        }

        // Flatten reference weights into the returned tuple for the checker.
        let mut ref_weights = Vec::new();
        ref_weights.extend_from_slice(&sg);
        ref_weights.extend_from_slice(&su);
        ref_weights.extend_from_slice(&wg);
        ref_weights.extend_from_slice(&wu);
        (shared_out, routed_out, ref_weights, x, indice)
    }

    fn verify(
        shared_out: &[f32],
        routed_out: &[f32],
        ref_weights: &[f32],
        x: &[f32],
        indice: &[bool],
    ) {
        let sg = &ref_weights[0..H * IS];
        let su = &ref_weights[H * IS..2 * H * IS];
        let wg = &ref_weights[2 * H * IS..2 * H * IS + E * H * I];
        let wu = &ref_weights[2 * H * IS + E * H * I..];

        for b in 0..B {
            for i in 0..IS {
                let mut g = 0.0f32;
                let mut u = 0.0f32;
                for k in 0..H {
                    g += x[b * H + k] * sg[k * IS + i];
                    u += x[b * H + k] * su[k * IS + i];
                }
                let exp = silu(g) * u;
                assert!(
                    (shared_out[b * IS + i] - exp).abs() < 1e-4,
                    "shared mismatch b={} i={} got={} exp={}",
                    b,
                    i,
                    shared_out[b * IS + i],
                    exp
                );
            }
        }

        for e in 0..E {
            for b in 0..B {
                let routed = indice[e * B + b];
                for i in 0..I {
                    let mut g = 0.0f32;
                    let mut u = 0.0f32;
                    for k in 0..H {
                        g += x[b * H + k] * wg[e * (H * I) + k * I + i];
                        u += x[b * H + k] * wu[e * (H * I) + k * I + i];
                    }
                    let exp = if routed { silu(g) * u } else { 0.0 };
                    let got = routed_out[e * (B * I) + b * I + i];
                    assert!(
                        (got - exp).abs() < 1e-4,
                        "routed mismatch e={} b={} i={} got={} exp={}",
                        e,
                        b,
                        i,
                        got,
                        exp
                    );
                }
            }
        }
    }

    #[test]
    fn test_shared_silu_single_thread() {
        let (shared_out, routed_out, ref_weights, x, indice) = build_and_run(1);
        verify(&shared_out, &routed_out, &ref_weights, &x, &indice);
    }

    #[test]
    fn test_shared_silu_multithread() {
        let (shared_out, routed_out, ref_weights, x, indice) = build_and_run(4);
        verify(&shared_out, &routed_out, &ref_weights, &x, &indice);
    }
}
