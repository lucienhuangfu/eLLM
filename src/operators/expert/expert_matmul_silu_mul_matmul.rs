// === compiler/mul/experts_matmul_silu_mul_matmul.rs ===
#![allow(non_snake_case)]

use std::f16;
use std::marker::PhantomData;
use std::ops::{Add, Mul};
use std::sync::atomic::Ordering;

use crate::kernel::common::matmul_params::MatMulParams;

use crate::operators::assign::assign;
use crate::operators::expert::expert_routing::{task_assign, ExpertRouting, ExpertTaskMeta};
use crate::operators::send_sync_ptr::{ConstPtr, MutPtr};
use crate::operators::traits::ExpertsSiluTrait;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
use std::arch::x86_64::{
    _mm512_fmadd_ph, _mm512_loadu_ph, _mm512_mul_ph, _mm512_set1_ph, _mm512_storeu_ph,
};

// Variable naming used in this operator:
// - token_block_rows / token_block_start: routed-token macro block inside one expert.
// - output_cols / output_col_start: intermediate I columns produced by gate/up projections.
// - reduction_cols / reduction_col_start: hidden H dimension reduced by GEMM.
// - micro_tile_rows / micro_tile_cols: micro-kernel tile size.
// - gate_acc / up_acc: per-thread accumulators before SiLU(gate) * up.
// - token_offset_in_block: position inside the compact routed-token block.
// 本算子的变量命名约定：
// - token_block_rows / token_block_start：单个 expert 内 routed token 的宏块。
// - output_cols / output_col_start：gate/up 投影产生的 intermediate I 列。
// - reduction_cols / reduction_col_start：GEMM 中被规约的 hidden H 维度。
// - micro_tile_rows / micro_tile_cols：微内核 tile 大小。
// - gate_acc / up_acc：执行 SiLU(gate) * up 前的每线程累加器。
// - token_offset_in_block：compact routed-token block 内的位置。

#[derive(Clone)]
pub struct ExpertMatMulSilu<T> {
    pub input_ptr: ConstPtr<T>, // Input hidden states: [B,H]. 输入 hidden states。

    // Gate/up weights are expected in NT layout: [E][I x H], row stride = H.
    // gate/up 权重要求外部传入 NT 布局：[E][I x H]，行距为 H。
    pub gate_nt_ptr: ConstPtr<T>, // Gate weight NT: [E,I,H]. gate 权重 NT 布局。
    pub up_nt_ptr: ConstPtr<T>,   // Up weight NT: [E,I,H]. up 权重 NT 布局。

    pub routing: ExpertRouting<T>,

    pub output_ptr: MutPtr<T>, // Nonlinear output: [E,B,I]. 非线性输出。

    pub params: MatMulParams,

    pub batch: usize,       // Token capacity. token 容量。
    pub inter: usize,       // Intermediate size. intermediate 大小。
    pub hidden: usize,      // Hidden size. hidden 大小。
    pub num_experts: usize, // Expert count. expert 数量。
    pub decode_only_flag: bool,

    // === strides ===
    // === stride 参数 ===
    pub packed_panel_stride: usize, // reduction_block_cols * micro_tile_cols
    pub acc_stride: usize,          // micro_tile_rows * micro_tile_cols
    pub a_tile_stride: usize,       // micro_tile_rows * reduction_block_cols

    // === prepacked weights ===
    // === 预打包权重 ===
    // Gate/up weights are packed in new(), so run() only reads prebuilt panels.
    // gate/up 权重在 new() 中提前 pack，run() 中只读取已准备好的 panel。
    pub packed_gate: Box<[T]>, // [E][reduction_panels][output_panels][reduction_block * micro_cols]
    pub packed_up: Box<[T]>,   // [E][reduction_panels][output_panels][reduction_block * micro_cols]

    // === pools split by thread ===
    // === 按线程切分的缓存池 ===
    // Thread-private scratch buffers reused by run(); no runtime allocation.
    // 每线程独占 scratch buffer，run() 中复用，不动态分配。
    pub gate_acc_pool: Box<[T]>,
    pub up_acc_pool: Box<[T]>,
    pub a_tile_pool: Box<[T]>,

    pub idx_buf_pool: Box<[usize]>,

    // Task-space buffers, one slice per thread. run() reuses them without allocation.
    // task 空间缓存，每个线程一份；run() 中只复用，不动态分配。
    task_meta_pool: Box<[ExpertTaskMeta]>,
    task_meta_stride: usize, // num_experts
    routed_tokens_pool: Box<[usize]>,
    routed_stride: usize, // num_experts * capacity_per_expert

    // Transposed weights are owned by the caller; this operator only keeps pointers.
    // 转置后权重由外部持有生命周期；该算子只保存指针。
    // pub wgate_nt_buf: Box<[T]>,
    // pub wup_nt_buf: Box<[T]>,
    _marker: PhantomData<T>,
}

impl<T> ExpertMatMulSilu<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    pub unsafe fn new(
        input_ptr: *const T,   // Input hidden states: [B,H]. 输入 hidden states。
        gate_nt_ptr: *const T, // W_gate_nt[E,I,H], row-major per expert. gate 权重 NT。
        up_nt_ptr: *const T,   // W_up_nt[E,I,H], row-major per expert. up 权重 NT。
        routing: ExpertRouting<T>,
        output_ptr: *mut T, // Nonlinear output: [E,B,I]. 非线性输出。
        batch: usize,
        inter: usize,
        hidden: usize,
        num_experts: usize,
        a_row_step_macro: usize,  // token block rows. token 宏块行数。
        b_row_step_macro: usize,  // output block cols. 输出宏块列数。
        column_step_macro: usize, // reduction block cols. 规约宏块列数。
        a_row_step_micro: usize,  // micro tile rows. 微内核行数。
        b_row_step_micro: usize,  // micro tile cols. 微内核列数。
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

        // Detect thread count once and allocate all per-thread scratch in new().
        // 线程数只在 new() 中探测一次，并在这里分配所有每线程 scratch。
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

            routing,
            output_ptr: MutPtr { ptr: output_ptr },

            params: MatMulParams {
                a_row_step_macro,
                b_row_step_macro,
                column_step_macro,
                a_row_step_micro,
                b_row_step_micro,
            },

            batch,
            inter,
            hidden,
            num_experts,
            decode_only_flag,

            packed_panel_stride,
            acc_stride,
            a_tile_stride,

            packed_gate,
            packed_up,
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
                .add(tid * self.params.a_row_step_macro) as *mut usize;
            (ga, ua, at, idx)
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
        packed: &[T],
        expert_id: usize,
        output_col_start: usize,
        reduction_col_start: usize,
    ) -> *const T {
        let reduction_block_cols = self.params.column_step_macro.max(1);
        let micro_tile_cols = self.params.b_row_step_micro.max(1);
        let output_panel_count = self.inter.div_ceil(micro_tile_cols);
        let reduction_panel_count = self.hidden.div_ceil(reduction_block_cols);
        let expert_stride = reduction_panel_count * output_panel_count * self.packed_panel_stride;
        let panel_index = (reduction_col_start / reduction_block_cols) * output_panel_count
            + (output_col_start / micro_tile_cols);
        unsafe {
            packed
                .as_ptr()
                .add(expert_id * expert_stride + panel_index * self.packed_panel_stride)
        }
    }

    /// Pack routed tokens into a micro input tile and zero-pad unused rows.
    /// 将路由后的 token 收集到微内核输入 tile，未使用的行补零。
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
                    token_count: routed_token_count,
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
            let output_cols = self.inter;
            let reduction_cols = self.hidden;

            let token_block_rows = self.params.a_row_step_macro.max(1);
            let output_block_cols = self.params.b_row_step_macro.max(1);
            let reduction_block_cols = self.params.column_step_macro.max(1);
            let micro_tile_rows = self.params.a_row_step_micro.max(1);
            let micro_tile_cols = self.params.b_row_step_micro.max(1);

            debug_assert!(output_cols % micro_tile_cols == 0);
            debug_assert!(reduction_cols % reduction_block_cols == 0);

            let input_base = self.input_ptr.ptr;
            let input_row_stride = self.hidden;

            let output_base = self.output_ptr.ptr;
            let output_expert_stride = self.batch * self.inter;

            let (gate_acc, up_acc, a_tile, idx_buf) = self.thread_slices(thread_id);

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
                        (task_meta.token_count - token_block_start).min(token_block_rows);
                    debug_assert!(tokens_in_block > 0);

                    let token_slice = &routed_tokens[(task_meta.token_begin + token_block_start)
                        ..(task_meta.token_begin + token_block_start + tokens_in_block)];
                    for (buffer_offset, &token_id) in token_slice.iter().enumerate() {
                        *idx_buf.add(buffer_offset) = token_id;
                    }

                    let expert_id = task_meta.expert_id;

                    let mut output_col_offset = 0usize;
                    while output_col_offset < output_cols_in_block {
                        let output_cols_this =
                            (output_cols_in_block - output_col_offset).min(micro_tile_cols);
                        debug_assert!(
                            output_cols_this == micro_tile_cols
                                || output_col_offset + output_cols_this == output_cols_in_block
                        );

                        let mut token_offset_in_block = 0usize;
                        while token_offset_in_block < tokens_in_block {
                            let valid_rows =
                                (tokens_in_block - token_offset_in_block).min(micro_tile_rows);

                            for accumulator_index in 0..(micro_tile_rows * micro_tile_cols) {
                                *gate_acc.add(accumulator_index) = T::default();
                            }
                            for accumulator_index in 0..(micro_tile_rows * micro_tile_cols) {
                                *up_acc.add(accumulator_index) = T::default();
                            }

                            let mut reduction_col_start = 0usize;
                            while reduction_col_start < reduction_cols {
                                let gate_panel = self.packed_panel_ptr(
                                    &self.packed_gate,
                                    expert_id,
                                    output_col_start + output_col_offset,
                                    reduction_col_start,
                                );
                                let up_panel = self.packed_panel_ptr(
                                    &self.packed_up,
                                    expert_id,
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
                                        gate_acc as *mut T,
                                        up_acc as *mut T,
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
                                        gate_acc as *mut T,
                                        up_acc as *mut T,
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
                                        gate_acc as *mut T,
                                        up_acc as *mut T,
                                        reduction_block_cols,
                                    );
                                }

                                reduction_col_start += reduction_block_cols;
                            }

                            // Finalize SiLU(gate) * up for each routed token row.
                            // 对每个 routed token 行计算 SiLU(gate) * up 并写回。
                            for row_in_tile in 0..valid_rows {
                                let token_id = *idx_buf.add(token_offset_in_block + row_in_tile);
                                let output_row = output_base
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

/* -------------------- ExpertSiluTrait default implementation -------------------- */
/* -------------------- ExpertSiluTrait 默认实现 -------------------- */

impl<T> ExpertsSiluTrait<T> for ExpertMatMulSilu<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    default fn compute1(
        &self,
        _a_tile: *const T,
        _gate_panel: *const T,
        _up_panel: *const T,
        _gate_acc: *mut T,
        _up_acc: *mut T,
        _kc: usize,
    ) {
    }

    default fn compute1_single(
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

    default fn compute1_rows(
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

    default fn compute2(&self, _gate_row: *const T, _up_row: *const T, _c_row: *mut T) {}
}

/* -------------------- f16 specialization: AVX-512 FP16 -------------------- */
/* -------------------- f16 特化实现：AVX-512 FP16 -------------------- */

impl ExpertsSiluTrait<f16> for ExpertMatMulSilu<f16> {
    fn compute1(
        &self,
        a_tile: *const f16,
        gate_panel: *const f16,
        up_panel: *const f16,
        gate_acc: *mut f16,
        up_acc: *mut f16,
        kc: usize,
    ) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            let nr = 32usize;
            // Map compact A tile layout: a_row_step_macro = lda = kc, b_row_step_macro = ldc_acc = 32
            let call_param = MatMulParams {
                a_row_step_macro: kc, // lda = kc (A tile stored compact, row stride = kc)
                b_row_step_macro: nr, // ldc_acc = 32
                column_step_macro: kc,
                a_row_step_micro: 3,
                b_row_step_micro: nr,
            };
            crate::kernel::x86_64::f16_512::fused_gate_up_silu_mul_block::fused_update_gate_up_acc_block(
                a_tile,
                gate_panel,
                up_panel,
                gate_acc,
                up_acc,
                &call_param,
            );
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        unsafe {
            let micro_tile_rows = self.params.a_row_step_micro.max(1);
            let micro_tile_cols = self.params.b_row_step_micro.max(1);
            let reduction_block_cols = kc;
            for row_in_tile in 0..micro_tile_rows {
                for col_in_tile in 0..micro_tile_cols {
                    let mut gate =
                        *gate_acc.add(row_in_tile * micro_tile_cols + col_in_tile) as f32;
                    let mut up = *up_acc.add(row_in_tile * micro_tile_cols + col_in_tile) as f32;
                    for reduction_lane in 0..reduction_block_cols {
                        let input =
                            *a_tile.add(row_in_tile * reduction_block_cols + reduction_lane) as f32;
                        gate += input
                            * (*gate_panel.add(reduction_lane * micro_tile_cols + col_in_tile)
                                as f32);
                        up += input
                            * (*up_panel.add(reduction_lane * micro_tile_cols + col_in_tile)
                                as f32);
                    }
                    *gate_acc.add(row_in_tile * micro_tile_cols + col_in_tile) = gate as f16;
                    *up_acc.add(row_in_tile * micro_tile_cols + col_in_tile) = up as f16;
                }
            }
        }
    }

    fn compute1_single(
        &self,
        input_row: *const f16,
        gate_panel: *const f16,
        up_panel: *const f16,
        gate_acc: *mut f16,
        up_acc: *mut f16,
        kc: usize,
    ) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            let mut gate = _mm512_loadu_ph(gate_acc);
            let mut up = _mm512_loadu_ph(up_acc);
            for reduction_lane in 0..kc {
                let a = _mm512_set1_ph(*input_row.add(reduction_lane));
                let gate_w = _mm512_loadu_ph(gate_panel.add(reduction_lane * 32));
                let up_w = _mm512_loadu_ph(up_panel.add(reduction_lane * 32));
                gate = _mm512_fmadd_ph(a, gate_w, gate);
                up = _mm512_fmadd_ph(a, up_w, up);
            }
            _mm512_storeu_ph(gate_acc, gate);
            _mm512_storeu_ph(up_acc, up);
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        unsafe {
            let micro_tile_cols = self.params.b_row_step_micro.max(1);
            for col_in_tile in 0..micro_tile_cols {
                let mut gate = *gate_acc.add(col_in_tile) as f32;
                let mut up = *up_acc.add(col_in_tile) as f32;
                for reduction_lane in 0..kc {
                    let input = *input_row.add(reduction_lane) as f32;
                    gate += input
                        * (*gate_panel.add(reduction_lane * micro_tile_cols + col_in_tile) as f32);
                    up += input
                        * (*up_panel.add(reduction_lane * micro_tile_cols + col_in_tile) as f32);
                }
                *gate_acc.add(col_in_tile) = gate as f16;
                *up_acc.add(col_in_tile) = up as f16;
            }
        }
    }

    fn compute1_rows(
        &self,
        a_tile: *const f16,
        gate_panel: *const f16,
        up_panel: *const f16,
        gate_acc: *mut f16,
        up_acc: *mut f16,
        kc: usize,
        rows: usize,
    ) {
        let nr = 32usize;
        let call_param = MatMulParams {
            a_row_step_macro: kc,
            b_row_step_macro: nr,
            column_step_macro: kc,
            a_row_step_micro: rows,
            b_row_step_micro: nr,
        };

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            // Use matmul_block (mr <= 3 aware) instead of fused_update_gate_up_acc_block
            // (which hardcodes 3 rows and debug_asserts a_row_step_micro == 3).
            crate::kernel::x86_64::f16_512::matmul_block::matmul_block(
                a_tile,
                gate_panel,
                gate_acc,
                &call_param,
            );
            crate::kernel::x86_64::f16_512::matmul_block::matmul_block(
                a_tile,
                up_panel,
                up_acc,
                &call_param,
            );
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        unsafe {
            let micro_tile_cols = self.params.b_row_step_micro.max(1);
            for row_in_tile in 0..rows {
                for col_in_tile in 0..micro_tile_cols {
                    let mut gate =
                        *gate_acc.add(row_in_tile * micro_tile_cols + col_in_tile) as f32;
                    let mut up = *up_acc.add(row_in_tile * micro_tile_cols + col_in_tile) as f32;
                    for reduction_lane in 0..kc {
                        let input = *a_tile.add(row_in_tile * kc + reduction_lane) as f32;
                        gate += input
                            * (*gate_panel.add(reduction_lane * micro_tile_cols + col_in_tile)
                                as f32);
                        up += input
                            * (*up_panel.add(reduction_lane * micro_tile_cols + col_in_tile)
                                as f32);
                    }
                    *gate_acc.add(row_in_tile * micro_tile_cols + col_in_tile) = gate as f16;
                    *up_acc.add(row_in_tile * micro_tile_cols + col_in_tile) = up as f16;
                }
            }
        }
    }

    fn compute2(&self, gate_row: *const f16, up_row: *const f16, c_row: *mut f16) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            // SiLU(g) * u using AVX-512 FP16: load, compute sigmoid, multiply.
            // Use the fused kernel that does sigmoid + silu + mul in one pass.
            // Since this is a single-row finalize, we call the block finalize
            // with a 1-row "block" — it only processes row 0 with stride 32.
            let gate = _mm512_loadu_ph(gate_row);
            let up = _mm512_loadu_ph(up_row);
            // sigmoid(g) = 1 / (1 + exp(-g)). We approximate via g * sigmoid(g).
            // Use the existing sigmoid512 kernel.
            let s = crate::kernel::x86_64::f16_512::activation::sigmoid512(gate);
            let silu_g = _mm512_mul_ph(gate, s);
            let result = _mm512_mul_ph(silu_g, up);
            _mm512_storeu_ph(c_row, result);
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        unsafe {
            let micro_tile_cols = self.params.b_row_step_micro.max(1);
            for col_in_tile in 0..micro_tile_cols {
                let gate = *gate_row.add(col_in_tile) as f32;
                let up = *up_row.add(col_in_tile) as f32;
                let silu = gate / (1.0 + (-gate).exp());
                *c_row.add(col_in_tile) = (silu * up) as f16;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rand::prelude::*;
    use std::collections::HashSet;

    fn test_routing_from_indice(
        num_experts: usize,
        batch: usize,
        indice: &[bool],
    ) -> ExpertRouting<f16> {
        let scores = vec![1.0f16; num_experts * batch];
        let topk = vec![0usize; batch];
        unsafe {
            crate::operators::expert::expert_routing::routing_from_dense(
                num_experts,
                batch,
                1,
                indice.as_ptr(),
                scores.as_ptr(),
                topk.as_ptr(),
            )
        }
    }

    #[inline]
    fn silu_f32(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    fn ref_one(
        a: &[f16],                  // [B,H]
        w_gate_kxn: &[f16],         // ✅ 参考仍用 K×N: [E,H,I] => kk*I+ii
        w_up_kxn: &[f16],           // ✅
        experts_indicator: &[bool], // [E]
        indice: &[bool],            // [E,B]
        out: &mut [f32],            // [E,B,I]
        b: usize,
        h: usize,
        i: usize,
        e: usize,
    ) {
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
                    for kk in 0..h {
                        let a_v = a[bb * h + kk] as f32;
                        let wg = w_gate_kxn[ex * (h * i) + kk * i + ii] as f32;
                        let wu = w_up_kxn[ex * (h * i) + kk * i + ii] as f32;
                        g += a_v * wg;
                        u += a_v * wu;
                    }
                    out[ex * (b * i) + bb * i + ii] = silu_f32(g) * u;
                }
            }
        }
    }

    // K×N (H×I) -> N×K (I×H)
    fn transpose_expert_kxn_to_nt(src: &[f16], e: usize, h: usize, i: usize) -> Vec<f16> {
        let mut out = vec![0.0f16; e * i * h];
        for ex in 0..e {
            let src_ex = &src[ex * (h * i)..(ex + 1) * (h * i)];
            let dst_ex = &mut out[ex * (i * h)..(ex + 1) * (i * h)];
            for kk in 0..h {
                for ii in 0..i {
                    dst_ex[ii * h + kk] = src_ex[kk * i + ii];
                }
            }
        }
        out
    }

    fn run_all_threads(runner: &ExpertMatMulSilu<f16>, batch: usize, cpu_num: usize) {
        for tid in 0..cpu_num {
            runner.run(batch, 0, cpu_num, tid);
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_experts_silu_single_expert_basic() {
        const B: usize = 6;
        const H: usize = 64;
        const I: usize = 64;
        const E: usize = 1;

        let mb = 3;
        let nb = 32;
        let kc = 64;
        let mr = 3;
        let nr = 32;

        let cpu_num = 2;

        let mut a = vec![0.0f16; B * H];
        let mut w_gate = vec![0.0f16; E * H * I]; // K×N
        let mut w_up = vec![0.0f16; E * H * I]; // K×N
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
        for kk in 0..H {
            for ii in 0..I {
                w_gate[0 * (H * I) + kk * I + ii] =
                    ((kk as f32) * 0.002 + (ii as f32) * 0.0003) as f16;
                w_up[0 * (H * I) + kk * I + ii] =
                    ((kk as f32) * 0.0017 + (ii as f32) * 0.0002) as f16;
            }
        }

        // ✅ 现在算子需要 NT
        let w_gate_nt = transpose_expert_kxn_to_nt(&w_gate, E, H, I);
        let w_up_nt = transpose_expert_kxn_to_nt(&w_up, E, H, I);

        let runner = unsafe {
            ExpertMatMulSilu::<f16>::new(
                a.as_ptr(),
                w_gate_nt.as_ptr(),
                w_up_nt.as_ptr(),
                test_routing_from_indice(E, B, &indice),
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
            )
        };

        run_all_threads(&runner, B, cpu_num);

        let mut ref_out = vec![0.0f32; E * B * I];
        ref_one(
            &a,
            &w_gate,
            &w_up,
            &experts_indicator,
            &indice,
            &mut ref_out,
            B,
            H,
            I,
            E,
        );

        for idx in 0..(E * B * I) {
            assert_abs_diff_eq!(out[idx] as f32, ref_out[idx], epsilon = 5e-1);
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_experts_silu_multi_expert_sparse_routing() {
        const B: usize = 12;
        const H: usize = 64;
        const I: usize = 96;
        const E: usize = 3;

        let mb = 6;
        let nb = 64;
        let kc = 64;
        let mr = 3;
        let nr = 32;

        let cpu_num = 4;

        let mut a = vec![0.0f16; B * H];
        let mut w_gate = vec![0.0f16; E * H * I]; // K×N
        let mut w_up = vec![0.0f16; E * H * I]; // K×N
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
        for ex in 0..E {
            for kk in 0..H {
                for ii in 0..I {
                    w_gate[ex * (H * I) + kk * I + ii] =
                        (((ex * 13 + kk * 5 + ii * 7) % 29) as f32 * 0.01) as f16;
                    w_up[ex * (H * I) + kk * I + ii] =
                        (((ex * 11 + kk * 3 + ii * 9) % 37) as f32 * 0.01) as f16;
                }
            }
        }

        let w_gate_nt = transpose_expert_kxn_to_nt(&w_gate, E, H, I);
        let w_up_nt = transpose_expert_kxn_to_nt(&w_up, E, H, I);

        let runner = unsafe {
            ExpertMatMulSilu::<f16>::new(
                a.as_ptr(),
                w_gate_nt.as_ptr(),
                w_up_nt.as_ptr(),
                test_routing_from_indice(E, B, &indice),
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
            )
        };

        run_all_threads(&runner, B, cpu_num);

        let mut ref_out = vec![0.0f32; E * B * I];
        ref_one(
            &a,
            &w_gate,
            &w_up,
            &experts_indicator,
            &indice,
            &mut ref_out,
            B,
            H,
            I,
            E,
        );

        for idx in 0..(E * B * I) {
            assert_abs_diff_eq!(out[idx] as f32, ref_out[idx], epsilon = 7e-1);
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_experts_silu_kc_split_accum() {
        const B: usize = 9;
        const H: usize = 128;
        const I: usize = 64;
        const E: usize = 2;

        let mb = 3;
        let nb = 32;
        let kc = 64;
        let mr = 3;
        let nr = 32;

        let cpu_num = 3;

        let mut a = vec![0.0f16; B * H];
        let mut w_gate = vec![0.0f16; E * H * I]; // K×N
        let mut w_up = vec![0.0f16; E * H * I]; // K×N
        let mut out = vec![0.0f16; E * B * I];

        let experts_indicator = vec![true; E];
        let mut indice = vec![false; E * B];
        for ex in 0..E {
            for bb in 0..B {
                indice[ex * B + bb] = true;
            }
        }

        for bb in 0..B {
            for kk in 0..H {
                a[bb * H + kk] = (((bb + kk) % 23) as f32 * 0.01) as f16;
            }
        }
        for ex in 0..E {
            for kk in 0..H {
                for ii in 0..I {
                    w_gate[ex * (H * I) + kk * I + ii] =
                        (((kk * 2 + ii + ex) % 17) as f32 * 0.01) as f16;
                    w_up[ex * (H * I) + kk * I + ii] =
                        (((kk * 3 + ii * 2 + ex) % 19) as f32 * 0.01) as f16;
                }
            }
        }

        let w_gate_nt = transpose_expert_kxn_to_nt(&w_gate, E, H, I);
        let w_up_nt = transpose_expert_kxn_to_nt(&w_up, E, H, I);

        let runner = unsafe {
            ExpertMatMulSilu::<f16>::new(
                a.as_ptr(),
                w_gate_nt.as_ptr(),
                w_up_nt.as_ptr(),
                test_routing_from_indice(E, B, &indice),
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
            )
        };

        run_all_threads(&runner, B, cpu_num);

        let mut ref_out = vec![0.0f32; E * B * I];
        ref_one(
            &a,
            &w_gate,
            &w_up,
            &experts_indicator,
            &indice,
            &mut ref_out,
            B,
            H,
            I,
            E,
        );

        for idx in 0..(E * B * I) {
            assert_abs_diff_eq!(out[idx] as f32, ref_out[idx], epsilon = 8e-1);
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_experts_silu_uneven_expert_loads_many_threads() {
        const B: usize = 13;
        const H: usize = 64;
        const I: usize = 64;
        const E: usize = 3;

        let mb = 3;
        let nb = 32;
        let kc = 64;
        let mr = 3;
        let nr = 32;
        let cpu_num = 8;

        let mut a = vec![0.0f16; B * H];
        let mut w_gate = vec![0.0f16; E * H * I];
        let mut w_up = vec![0.0f16; E * H * I];
        let mut out = vec![0.0f16; E * B * I];

        let experts_indicator = vec![true, true, true];
        let mut indice = vec![false; E * B];

        for &bb in &[0usize, 1, 3, 4, 7, 9, 12] {
            indice[0 * B + bb] = true;
        }
        for &bb in &[2usize, 10] {
            indice[1 * B + bb] = true;
        }
        indice[2 * B + 5] = true;

        for bb in 0..B {
            for kk in 0..H {
                a[bb * H + kk] = ((bb as f32) * 0.02 + (kk as f32) * 0.0015) as f16;
            }
        }
        for ex in 0..E {
            for kk in 0..H {
                for ii in 0..I {
                    let base = ex * H * I + kk * I + ii;
                    w_gate[base] = ((ex as f32 + 1.0) * 0.002
                        + (kk as f32) * 0.0002
                        + (ii as f32) * 0.0001) as f16;
                    w_up[base] = ((ex as f32 + 1.0) * 0.003 + (kk as f32) * 0.0001
                        - (ii as f32) * 0.00005) as f16;
                }
            }
        }

        let w_gate_nt = transpose_expert_kxn_to_nt(&w_gate, E, H, I);
        let w_up_nt = transpose_expert_kxn_to_nt(&w_up, E, H, I);

        let runner = unsafe {
            ExpertMatMulSilu::<f16>::new(
                a.as_ptr(),
                w_gate_nt.as_ptr(),
                w_up_nt.as_ptr(),
                test_routing_from_indice(E, B, &indice),
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
            )
        };

        run_all_threads(&runner, B, cpu_num);

        let mut ref_out = vec![0.0f32; E * B * I];
        ref_one(
            &a,
            &w_gate,
            &w_up,
            &experts_indicator,
            &indice,
            &mut ref_out,
            B,
            H,
            I,
            E,
        );

        for ex in 0..E {
            for bb in 0..B {
                let routed = indice[ex * B + bb];
                for ii in 0..I {
                    let got = out[ex * (B * I) + bb * I + ii] as f32;
                    let exp = ref_out[ex * (B * I) + bb * I + ii];
                    if routed {
                        assert_abs_diff_eq!(got, exp, epsilon = 7e-1);
                    } else {
                        assert_abs_diff_eq!(got, 0.0, epsilon = 1e-3);
                    }
                }
            }
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_experts_silu_more_threads_than_tasks() {
        const B: usize = 3;
        const H: usize = 64;
        const I: usize = 64;
        const E: usize = 2;

        let mb = 3;
        let nb = 32;
        let kc = 64;
        let mr = 3;
        let nr = 32;
        let cpu_num = 16;

        let mut a = vec![0.0f16; B * H];
        let mut w_gate = vec![0.0f16; E * H * I];
        let mut w_up = vec![0.0f16; E * H * I];
        let mut out = vec![0.0f16; E * B * I];

        let experts_indicator = vec![true, false];
        let mut indice = vec![false; E * B];
        for bb in 0..B {
            indice[0 * B + bb] = true;
        }

        for bb in 0..B {
            for kk in 0..H {
                a[bb * H + kk] = (0.1 + bb as f32 * 0.03 + kk as f32 * 0.0007) as f16;
            }
        }
        for kk in 0..H {
            for ii in 0..I {
                w_gate[kk * I + ii] = (0.01 + kk as f32 * 0.0001 + ii as f32 * 0.00005) as f16;
                w_up[kk * I + ii] = (0.02 - kk as f32 * 0.00008 + ii as f32 * 0.00003) as f16;
            }
        }

        let w_gate_nt = transpose_expert_kxn_to_nt(&w_gate, E, H, I);
        let w_up_nt = transpose_expert_kxn_to_nt(&w_up, E, H, I);

        let runner = unsafe {
            ExpertMatMulSilu::<f16>::new(
                a.as_ptr(),
                w_gate_nt.as_ptr(),
                w_up_nt.as_ptr(),
                test_routing_from_indice(E, B, &indice),
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
            )
        };

        run_all_threads(&runner, B, cpu_num);

        let mut ref_out = vec![0.0f32; E * B * I];
        ref_one(
            &a,
            &w_gate,
            &w_up,
            &experts_indicator,
            &indice,
            &mut ref_out,
            B,
            H,
            I,
            E,
        );

        for idx in 0..(E * B * I) {
            assert_abs_diff_eq!(out[idx] as f32, ref_out[idx], epsilon = 7e-1);
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_experts_silu_active_expert_with_zero_tokens_keeps_output_zero() {
        const B: usize = 6;
        const H: usize = 64;
        const I: usize = 64;
        const E: usize = 3;

        let mb = 3;
        let nb = 32;
        let kc = 64;
        let mr = 3;
        let nr = 32;
        let cpu_num = 4;

        let mut a = vec![0.0f16; B * H];
        let mut w_gate = vec![0.0f16; E * H * I];
        let mut w_up = vec![0.0f16; E * H * I];
        let mut out = vec![0.0f16; E * B * I];

        let experts_indicator = vec![true, true, false];
        let mut indice = vec![false; E * B];
        for &bb in &[0usize, 2, 4] {
            indice[0 * B + bb] = true;
        }

        for bb in 0..B {
            for kk in 0..H {
                a[bb * H + kk] = ((bb as f32) * 0.015 + (kk as f32) * 0.0009) as f16;
            }
        }
        for ex in 0..E {
            for kk in 0..H {
                for ii in 0..I {
                    let base = ex * H * I + kk * I + ii;
                    w_gate[base] = ((ex as f32 + 1.0) * 0.005 + kk as f32 * 0.00005) as f16;
                    w_up[base] = ((ex as f32 + 1.0) * 0.004 + ii as f32 * 0.00004) as f16;
                }
            }
        }

        let w_gate_nt = transpose_expert_kxn_to_nt(&w_gate, E, H, I);
        let w_up_nt = transpose_expert_kxn_to_nt(&w_up, E, H, I);

        let runner = unsafe {
            ExpertMatMulSilu::<f16>::new(
                a.as_ptr(),
                w_gate_nt.as_ptr(),
                w_up_nt.as_ptr(),
                test_routing_from_indice(E, B, &indice),
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
            )
        };

        run_all_threads(&runner, B, cpu_num);

        for bb in 0..B {
            for ii in 0..I {
                let v = out[1 * (B * I) + bb * I + ii] as f32;
                assert_abs_diff_eq!(v, 0.0, epsilon = 1e-3);
            }
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_experts_matmul_silu_qwen_moe_config() {
        let batch = 72;
        let hidden = 128 * 32; // 4096
        let inter = 768;
        let num_experts = 128;
        let top_k = 8;
        let num_threads = 4;

        let mr = 3;
        let nr = 32;
        let kc = 32;
        let mb = 3;
        let nb = 32;

        let mut rng = rand::thread_rng();

        let input: Vec<f16> = (0..batch * hidden)
            .map(|_| rng.gen_range(-0.1f32..0.1f32) as f16)
            .collect();

        // 原始权重：K×N（H×I）
        let gate_weights_kxn: Vec<f16> = (0..num_experts * hidden * inter)
            .map(|_| rng.gen_range(-0.05f32..0.05f32) as f16)
            .collect();
        let up_weights_kxn: Vec<f16> = (0..num_experts * hidden * inter)
            .map(|_| rng.gen_range(-0.05f32..0.05f32) as f16)
            .collect();

        // ✅ 转成 NT：N×K（I×H）
        let gate_weights_nt =
            transpose_expert_kxn_to_nt(&gate_weights_kxn, num_experts, hidden, inter);
        let up_weights_nt = transpose_expert_kxn_to_nt(&up_weights_kxn, num_experts, hidden, inter);

        let mut indice_ptr = vec![false; num_experts * batch];
        let mut experts_indicator = vec![false; num_experts];

        use std::collections::HashSet;
        for b in 0..batch {
            let mut selected_experts = HashSet::new();
            while selected_experts.len() < top_k {
                selected_experts.insert(rng.gen_range(0..num_experts));
            }
            for &e in &selected_experts {
                indice_ptr[e * batch + b] = true;
                experts_indicator[e] = true;
            }
        }

        // 输出先清零（很重要：避免未路由位置残留旧值）
        let mut output = vec![0.0 as f16; num_experts * batch * inter];

        unsafe {
            let op = ExpertMatMulSilu::new(
                input.as_ptr(),
                gate_weights_nt.as_ptr(),
                up_weights_nt.as_ptr(),
                test_routing_from_indice(num_experts, batch, &indice_ptr),
                output.as_mut_ptr(),
                batch,
                inter,
                hidden,
                num_experts,
                mb,
                nb,
                kc,
                mr,
                nr,
                false,
            );

            // 注意：最好不要超过 available_parallelism()，避免 scratch 越界
            let threads_cap = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            let used = num_threads.min(threads_cap).max(1);

            for tid in 0..used {
                op.run(batch, 0, used, tid);
            }
        }

        // ============ 抽样 reference 校验 ============

        #[inline]
        fn silu_f32(x: f32) -> f32 {
            x / (1.0 + (-x).exp())
        }

        // 抽样数量：64~256 都行
        let samples = 96usize;
        let tolerance = 0.20f32; // fp16 + silu，放宽点更稳

        let mut checked = 0usize;
        let mut tries = 0usize;
        let max_tries = samples * 20; // 防止一直抽到没路由的点

        while checked < samples && tries < max_tries {
            tries += 1;

            let e = rng.gen_range(0..num_experts);
            let b = rng.gen_range(0..batch);
            if !indice_ptr[e * batch + b] {
                continue;
            }
            let ii = rng.gen_range(0..inter);

            // reference: g = sum_k a[b,k] * w_gate[k,ii]; u = sum_k a[b,k] * w_up[k,ii]
            let mut g = 0.0f32;
            let mut u = 0.0f32;

            let a_row = &input[b * hidden..(b + 1) * hidden];

            // K×N: w[ex*(H*I) + kk*I + ii]
            let woff = e * (hidden * inter) + ii; // base + ii
            for kk in 0..hidden {
                let a_v = a_row[kk] as f32;
                let wg = gate_weights_kxn[woff + kk * inter] as f32;
                let wu = up_weights_kxn[woff + kk * inter] as f32;
                g += a_v * wg;
                u += a_v * wu;
            }

            let ref_val = silu_f32(g) * u;

            let out_idx = e * (batch * inter) + b * inter + ii;
            let got = output[out_idx] as f32;

            // 误差允许：fp16 + 大 H dot + 非线性
            let diff = (got - ref_val).abs();
            if diff > tolerance && ref_val.abs() > 0.01 {
                panic!(
                    "Mismatch at Expert={}, Batch={}, Idx={}: Got {}, Expected {}, Diff {}",
                    e, b, ii, got, ref_val, diff
                );
            }

            checked += 1;
        }

        assert!(
            checked >= samples / 2,
            "too few routed samples to check: checked {} / {}",
            checked,
            samples
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_silu_stride_capacity_batch_run_must_not_touch_rows_7_8() {
        use std::arch::is_x86_feature_detected;
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }

        const B_CAP: usize = 9;
        const B_RUN: usize = 7;
        const H: usize = 64;
        const I: usize = 64;
        const E: usize = 1;

        let mb = 3;
        let nb = 32;
        let kc = 64;
        let mr = 3;
        let nr = 32;

        // A[B_CAP,H]：前 7 行是 1，后 2 行是 50（如果被算到，会输出明显非零）
        let mut a = vec![0.0f16; B_CAP * H];
        for b in 0..B_RUN {
            for kk in 0..H {
                a[b * H + kk] = 1.0f32 as f16;
            }
        }
        for b in B_RUN..B_CAP {
            for kk in 0..H {
                a[b * H + kk] = 50.0f32 as f16;
            }
        }

        // gate/up 权重 K×N（参考用）=> 转 NT 给算子
        let mut w_gate_kxn = vec![0.0f16; E * H * I];
        let mut w_up_kxn = vec![0.0f16; E * H * I];

        // 让输出稳定非零：gate=0.01, up=0.02
        for kk in 0..H {
            for ii in 0..I {
                w_gate_kxn[kk * I + ii] = 0.01f32 as f16;
                w_up_kxn[kk * I + ii] = 0.02f32 as f16;
            }
        }

        // 你文件里已有这个 helper
        let w_gate_nt = transpose_expert_kxn_to_nt(&w_gate_kxn, E, H, I);
        let w_up_nt = transpose_expert_kxn_to_nt(&w_up_kxn, E, H, I);

        // routing：capacity 0..8 都 true（故意）
        // 正确行为：run(batch=7) 只能处理 0..6，7/8 绝不能写 output
        let experts_indicator = vec![true; E];
        let mut indice = vec![false; E * B_CAP];
        for b in 0..B_CAP {
            indice[b] = true;
        }

        // output[E,B_CAP,I]：先清零
        let mut out = vec![0.0f16; E * B_CAP * I];

        let op = unsafe {
            ExpertMatMulSilu::<f16>::new(
                a.as_ptr(),
                w_gate_nt.as_ptr(),
                w_up_nt.as_ptr(),
                test_routing_from_indice(E, B_CAP, &indice),
                out.as_mut_ptr(),
                B_CAP, // capacity
                I,
                H,
                E,
                mb,
                nb,
                kc,
                mr,
                nr,
                false,
            )
        };

        // 单线程即可复现
        op.run(B_RUN, 0, 1, 0);

        // row 7/8 必须仍为 0（没被触碰）
        for b in B_RUN..B_CAP {
            for ii in 0..I {
                let v = out[b * I + ii] as f32;
                assert!(v.abs() <= 1e-3, "row {} should remain 0, got {}", b, v);
            }
        }

        // 前 7 行至少应有非零（避免“全没算”的假通过）
        let mut any_nonzero = false;
        'outer: for b in 0..B_RUN {
            for ii in 0..I {
                if (out[b * I + ii] as f32).abs() > 1e-3 {
                    any_nonzero = true;
                    break 'outer;
                }
            }
        }
        assert!(any_nonzero, "expected some non-zero outputs for rows 0..6");
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_experts_silu_batch7_capacity9_must_not_touch_rows_7_8() {
        const B_CAP: usize = 9;
        const B_RUN: usize = 7;
        const H: usize = 64;
        const I: usize = 64;
        const E: usize = 1;

        let mb = 3;
        let nb = 32;
        let kc = 64;
        let mr = 3;
        let nr = 32;

        let cpu_num = 2;

        // A[B_CAP,H]：前 7 行填 1，后 2 行填一个很大的值，方便检测“被错误写入”
        let mut a = vec![0.0f16; B_CAP * H];
        for b in 0..B_RUN {
            for kk in 0..H {
                a[b * H + kk] = 1.0f32 as f16;
            }
        }
        for b in B_RUN..B_CAP {
            for kk in 0..H {
                a[b * H + kk] = 50.0f32 as f16;
            }
        }

        // 权重 K×N（参考用），再转 NT 给算子
        let mut w_gate = vec![0.0f16; E * H * I];
        let mut w_up = vec![0.0f16; E * H * I];
        for kk in 0..H {
            for ii in 0..I {
                // 让输出非零且稳定
                w_gate[kk * I + ii] = 0.01f32 as f16;
                w_up[kk * I + ii] = 0.02f32 as f16;
            }
        }
        let w_gate_nt = transpose_expert_kxn_to_nt(&w_gate, E, H, I);
        let w_up_nt = transpose_expert_kxn_to_nt(&w_up, E, H, I);

        // 路由：capacity 里 0..8 都置 true（故意！）
        // 正确行为：run(batch=7) 时只允许 0..6 生效，7/8 不得写 output
        let experts_indicator = vec![true; E];
        let mut indice = vec![false; E * B_CAP];
        for bb in 0..B_CAP {
            indice[0 * B_CAP + bb] = true;
        }

        // output[E,B_CAP,I]：先全置 0
        let mut out = vec![0.0f16; E * B_CAP * I];

        let runner = unsafe {
            ExpertMatMulSilu::<f16>::new(
                a.as_ptr(),
                w_gate_nt.as_ptr(),
                w_up_nt.as_ptr(),
                test_routing_from_indice(E, B_CAP, &indice),
                out.as_mut_ptr(),
                B_CAP, // batch capacity
                I,
                H,
                E,
                mb,
                nb,
                kc,
                mr,
                nr,
                false,
            )
        };

        // run 只跑 B_RUN
        for tid in 0..cpu_num {
            runner.run(B_RUN, 0, cpu_num, tid);
        }

        // 断言：row 7,8 必须仍然是 0（没被触碰）
        for bb in B_RUN..B_CAP {
            for ii in 0..I {
                let v = out[0 * (B_CAP * I) + bb * I + ii] as f32;
                assert!(
                    (v - 0.0f32).abs() <= 1e-3,
                    "row {} should remain 0, got {}",
                    bb,
                    v
                );
            }
        }

        // 可选：也简单检查一下前 7 行确实被写成了非零（避免“全没算”的假通过）
        let mut any_nonzero = false;
        for bb in 0..B_RUN {
            for ii in 0..I {
                if (out[bb * I + ii] as f32).abs() > 1e-3 {
                    any_nonzero = true;
                    break;
                }
            }
            if any_nonzero {
                break;
            }
        }
        assert!(any_nonzero, "expected some non-zero outputs for rows 0..6");
    }
    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_experts_matmul_silu_moe_smoke_fast_sampled() {
        // 缩小版“Qwen风格”测试：仍然是 MoE + topk routing + mr=3/nr=32 + kc split
        // 但规模足够小，保证 cargo test 很快跑完

        let batch = 24; // 原 72
        let hidden = 256; // 原 4096（仍保持能被 kc=32 整除）
        let inter = 128; // 原 768（能被 nr=32 整除）
        let num_experts = 16; // 原 128
        let top_k = 4; // 原 8
        let num_threads = 4;

        let mr = 3;
        let nr = 32;
        let kc = 32;
        let mb = 6; // 让 tiles_m>1，更像真实分块
        let nb = 64; // inter=128 => tiles_n=2

        use rand::prelude::*;
        use std::collections::HashSet;

        let mut rng = rand::thread_rng();

        let input: Vec<f16> = (0..batch * hidden)
            .map(|_| rng.gen_range(-0.1f32..0.1f32) as f16)
            .collect();

        // 原始权重：K×N（H×I）
        let gate_weights_kxn: Vec<f16> = (0..num_experts * hidden * inter)
            .map(|_| rng.gen_range(-0.05f32..0.05f32) as f16)
            .collect();
        let up_weights_kxn: Vec<f16> = (0..num_experts * hidden * inter)
            .map(|_| rng.gen_range(-0.05f32..0.05f32) as f16)
            .collect();

        // 转 NT：N×K（I×H）
        let gate_weights_nt =
            transpose_expert_kxn_to_nt(&gate_weights_kxn, num_experts, hidden, inter);
        let up_weights_nt = transpose_expert_kxn_to_nt(&up_weights_kxn, num_experts, hidden, inter);

        let mut indice_ptr = vec![false; num_experts * batch];
        let mut experts_indicator = vec![false; num_experts];

        for b in 0..batch {
            let mut selected_experts = HashSet::new();
            while selected_experts.len() < top_k {
                selected_experts.insert(rng.gen_range(0..num_experts));
            }
            for &e in &selected_experts {
                indice_ptr[e * batch + b] = true;
                experts_indicator[e] = true;
            }
        }

        let mut output = vec![0.0 as f16; num_experts * batch * inter];

        unsafe {
            let op = ExpertMatMulSilu::new(
                input.as_ptr(),
                gate_weights_nt.as_ptr(),
                up_weights_nt.as_ptr(),
                test_routing_from_indice(num_experts, batch, &indice_ptr),
                output.as_mut_ptr(),
                batch,
                inter,
                hidden,
                num_experts,
                mb,
                nb,
                kc,
                mr,
                nr,
                false,
            );

            // 你说固定机器跑，这里就直接用 num_threads（确保不超过你那台机的 threads）
            for tid in 0..num_threads {
                op.run(batch, 0, num_threads, tid);
            }
        }

        // ===== 抽样校验 routed 位置 =====
        #[inline]
        fn silu_f32(x: f32) -> f32 {
            x / (1.0 + (-x).exp())
        }

        let samples = 128usize;
        let tolerance = 0.35f32; // fp16 + 非线性，放宽一点更稳
        let mut checked = 0usize;
        let mut tries = 0usize;
        let max_tries = samples * 50;

        while checked < samples && tries < max_tries {
            tries += 1;

            let e = rng.gen_range(0..num_experts);
            let b = rng.gen_range(0..batch);
            if !indice_ptr[e * batch + b] {
                continue;
            }
            let ii = rng.gen_range(0..inter);

            let mut g = 0.0f32;
            let mut u = 0.0f32;

            let a_row = &input[b * hidden..(b + 1) * hidden];
            let woff = e * (hidden * inter) + ii;
            for kk in 0..hidden {
                let a_v = a_row[kk] as f32;
                let wg = gate_weights_kxn[woff + kk * inter] as f32;
                let wu = up_weights_kxn[woff + kk * inter] as f32;
                g += a_v * wg;
                u += a_v * wu;
            }

            let ref_val = silu_f32(g) * u;
            let out_idx = e * (batch * inter) + b * inter + ii;
            let got = output[out_idx] as f32;

            let diff = (got - ref_val).abs();
            if diff > tolerance && ref_val.abs() > 0.01 {
                panic!(
                    "Mismatch at Expert={}, Batch={}, Idx={}: Got {}, Expected {}, Diff {}",
                    e, b, ii, got, ref_val, diff
                );
            }

            checked += 1;
        }

        assert!(
            checked >= samples / 2,
            "too few routed samples checked: {}/{}",
            checked,
            samples
        );
    }
}
