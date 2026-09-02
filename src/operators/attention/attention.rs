use std::ops::{Add, Div, Mul, Sub};
use std::sync::{Arc, Once};

use crate::num_traits::NegInfinity;
use crate::operators::send_sync_ptr::{ConstPtr, MutPtr};
use crate::operators::traits::AttentionTrait;
use crate::runtime::scheduling::SequenceSlice;

use super::scratch::{AttentionScratch, AttentionScratchSlice};
use super::utils::{split_sequence_by_triangle, RowVisitPlan};

static BRGEMM_FALLBACK_WARNING: Once = Once::new();

/// Core Attention computation structure
/// Handles Q, K, V pointers and manages the attention computation
#[derive(Clone)]
pub struct Attention<T> {
    pub(super) q_ptr: ConstPtr<T>,
    pub(super) k_ptr: ConstPtr<T>,
    pub(super) v_ptr: ConstPtr<T>,
    pub(super) output_ptr: MutPtr<T>,
    pub(super) sequence_length: usize,
    pub(super) batch_size: usize,
    pub(super) attention_head_num: usize,
    pub(super) kv_head_num: usize,
    pub(super) k_batch_stride: usize,
    pub(super) k_head_stride: usize,
    pub(super) k_seq_stride: usize,
    pub(super) v_batch_stride: usize,
    pub(super) v_head_stride: usize,
    pub(super) v_seq_stride: usize,
    pub(super) q_seq_stride: usize,
    pub(super) head_size: usize,
    pub(super) inverse_sqrt_head: T,
    pub(super) row_step: usize,
    pub(super) col_step: usize,
    pub(super) decode_only_flag: bool,
    pub(super) hybrid_split: bool,
    pub(super) profile_split: bool,
    pub(super) force_head_split: bool,
    pub(super) head_row_split: bool,
    pub(super) causal_col_prune: bool,
    pub(super) decode_gqa8: bool,
    pub(super) brgemm_backend: bool,
    pub(super) brgemm_shared_cache:
        Arc<crate::kernel::x86_64::f16_512::brgemm_attention::SharedCache>,
    pub(super) thread_num: usize,
    pub(super) rotate_head_assignment: bool,
    scratch: AttentionScratch<T>,
}

impl<T> Attention<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd
        + NegInfinity,
{
    pub fn new(
        q_ptr: *const T,
        k_ptr: *const T,
        v_ptr: *const T,
        output_ptr: *mut T,
        sequence_length: usize,
        batch_size: usize,
        attention_head_num: usize,
        kv_head_num: usize,
        k_batch_stride: usize,
        k_head_stride: usize,
        k_seq_stride: usize,
        v_batch_stride: usize,
        v_head_stride: usize,
        v_seq_stride: usize,
        q_seq_stride: usize,
        row_step: usize,
        col_step: usize,
        head_size: usize,
        inverse_sqrt_head: T,
        decode_only_flag: bool,
        thread_num: usize,
    ) -> Self {
        let thread_num = thread_num.max(1);
        let mut row_step = row_step.max(1);
        let mut col_step = col_step.max(1);
        let hybrid_split = std::env::var("ELLM_ATTENTION_HYBRID_SPLIT")
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);
        let profile_split = std::env::var("ELLM_PROFILE_ATTENTION_SPLIT")
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);
        let force_head_split = std::env::var("ELLM_ATTENTION_FORCE_HEAD_SPLIT")
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);
        let requested_brgemm = std::env::var("ELLM_ATTENTION_BACKEND")
            .ok()
            .map(|value| value.eq_ignore_ascii_case("brgemm"))
            .unwrap_or(true);
        let head_row_split = std::env::var("ELLM_ATTENTION_HEAD_ROW_SPLIT")
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(requested_brgemm);
        let causal_col_prune = std::env::var("ELLM_ATTENTION_CAUSAL_COL_PRUNE")
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(requested_brgemm);
        let decode_gqa8 = std::env::var("ELLM_ATTENTION_DECODE_GQA8")
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);
        let rotate_head_assignment = std::env::var("ELLM_ATTENTION_ROTATE_HEADS")
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            // Rotation helped the 32-core/64-thread host at 20k+, but was
            // neutral-to-negative on the 24-core/48-thread deployment host.
            // Keep it available as an explicit machine-specific tuning knob.
            .unwrap_or(false);
        let brgemm_backend = requested_brgemm
            && std::mem::size_of::<T>() == std::mem::size_of::<f16>()
            && cfg!(all(target_arch = "x86_64", target_feature = "avx512fp16"))
            && crate::kernel::x86_64::f16_512::brgemm_attention::available();
        if requested_brgemm && !brgemm_backend {
            BRGEMM_FALLBACK_WARNING.call_once(|| {
                eprintln!(
                    "ELLM_ATTENTION_BACKEND=brgemm requires f16, AMX-FP16, and libtorch_cpu; using native attention"
                );
            });
        }
        if brgemm_backend {
            (row_step, col_step) = match sequence_length {
                0..=256 => (32, 64),
                257..=1024 => (128, 256),
                1025..=4096 => (256, 768),
                // For FP16 BRGEMM the per-thread hot workspace is approximately
                // M * (2 * sizeof(f32) + N * (sizeof(f32) + sizeof(f16))
                //     + head_size * sizeof(f32)).
                // M=160/N=384 uses about 441 KiB at head_size=128. It was at
                // least as fast as N=448 on the 24-core/48-thread deployment
                // host and leaves more L2 room for its SMT sibling and Q/K/V.
                _ => (160, 384),
            };
            row_step = std::env::var("ELLM_ATTENTION_BRGEMM_ROW_STEP")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(row_step);
            col_step = std::env::var("ELLM_ATTENTION_BRGEMM_COL_STEP")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(col_step);
        }

        Self {
            q_ptr: ConstPtr { ptr: q_ptr },
            k_ptr: ConstPtr { ptr: k_ptr },
            v_ptr: ConstPtr { ptr: v_ptr },
            output_ptr: MutPtr { ptr: output_ptr },
            batch_size,
            attention_head_num,
            kv_head_num,
            k_batch_stride,
            k_head_stride,
            k_seq_stride,
            v_batch_stride,
            v_head_stride,
            v_seq_stride,
            q_seq_stride,
            sequence_length,
            row_step,
            col_step,
            head_size,
            inverse_sqrt_head,
            decode_only_flag,
            hybrid_split,
            profile_split,
            force_head_split,
            head_row_split,
            causal_col_prune,
            decode_gqa8,
            brgemm_backend,
            brgemm_shared_cache: Arc::new(
                crate::kernel::x86_64::f16_512::brgemm_attention::SharedCache::default(),
            ),
            thread_num,
            rotate_head_assignment,
            scratch: AttentionScratch::new(thread_num, row_step, col_step),
        }
    }

    #[inline(always)]
    pub(super) fn thread_buffers(
        &self,
        thread_id: usize,
        row_count: usize,
        col_count: usize,
    ) -> AttentionScratchSlice<'_, T> {
        debug_assert!(thread_id < self.thread_num);
        self.scratch.thread_buffers(thread_id, row_count, col_count)
    }

    #[inline]
    pub(super) fn split_contiguous_range(
        total: usize,
        part_num: usize,
        part_id: usize,
    ) -> Option<(usize, usize)> {
        if total == 0 || part_num == 0 || part_id >= part_num {
            return None;
        }

        let begin = total * part_id / part_num;
        let end = total * (part_id + 1) / part_num;
        (begin < end).then_some((begin, end))
    }

    #[inline]
    pub(super) fn kv_heads_per_wave(
        &self,
        active_thread_num: usize,
        attention_heads_per_kv: usize,
    ) -> usize {
        if active_thread_num == 0 || attention_heads_per_kv == 0 || self.kv_head_num == 0 {
            return 0;
        }

        active_thread_num
            .div_ceil(attention_heads_per_kv)
            .max(1)
            .min(self.kv_head_num)
    }
}

impl<T> Attention<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd
        + NegInfinity
        + crate::num_traits::Exp,
{
    /// Visit aligned row ranges for attention computation
    #[inline(always)]
    unsafe fn visit_aligned_row_range(
        &self,
        q_head_ptr: *const T,
        output_head_ptr: *mut T,
        k_head_ptr: *const T,
        v_head_ptr: *const T,
        thread_id: usize,
        sequence_index: usize,
        col_end: usize,
        row_begin: usize,
        row_end: usize,
    ) {
        let row_step = self.row_step;

        for row_chunk in (row_begin..row_end).step_by(row_step) {
            let row_chunk_end = row_chunk + row_step;
            let visible_row_end = row_chunk_end.min(col_end.saturating_sub(sequence_index));
            if row_chunk >= visible_row_end {
                continue;
            }

            let row_count = visible_row_end - row_chunk;
            let col_step = if self.brgemm_backend && row_count == 1 {
                32
            } else {
                self.col_step.max(1)
            };
            let mut scratch = self.thread_buffers(thread_id, row_count, col_step);
            scratch.clear();

            for row in row_chunk..visible_row_end {
                let q_offset = row * self.q_seq_stride;
                for index in 0..self.head_size {
                    *output_head_ptr.add(q_offset + index) = T::default();
                }
            }

            let key_end = if self.causal_col_prune {
                (sequence_index + visible_row_end).min(col_end)
            } else {
                col_end
            };
            for col_begin in (0..key_end).step_by(col_step) {
                let col_chunk_end = (col_begin + col_step).min(key_end);
                AttentionTrait::compute(
                    self,
                    q_head_ptr,
                    k_head_ptr,
                    v_head_ptr,
                    output_head_ptr,
                    row_chunk,
                    visible_row_end,
                    col_begin,
                    col_chunk_end,
                    col_end,
                    sequence_index,
                    self.k_seq_stride,
                    self.v_seq_stride,
                    self.q_seq_stride,
                    scratch.running_max,
                    scratch.running_denom,
                    scratch.scores,
                );
            }
        }
    }

    /// Visit tail row ranges (unaligned) for attention computation
    #[inline(always)]
    unsafe fn visit_tail_row_range(
        &self,
        q_head_ptr: *const T,
        output_head_ptr: *mut T,
        k_head_ptr: *const T,
        v_head_ptr: *const T,
        thread_id: usize,
        sequence_index: usize,
        col_end: usize,
        row_begin: usize,
        row_end: usize,
    ) {
        let visible_row_end = row_end.min(col_end.saturating_sub(sequence_index));
        if row_begin >= visible_row_end {
            return;
        }

        for row in row_begin..visible_row_end {
            let q_offset = row * self.q_seq_stride;
            for index in 0..self.head_size {
                *output_head_ptr.add(q_offset + index) = T::default();
            }
        }

        let row_count = visible_row_end - row_begin;
        let col_step = if self.brgemm_backend && row_count == 1 {
            32
        } else {
            self.col_step.max(1)
        };
        let mut scratch = self.thread_buffers(thread_id, row_count, col_step);
        scratch.clear();

        let key_end = if self.causal_col_prune {
            (sequence_index + visible_row_end).min(col_end)
        } else {
            col_end
        };
        for col_begin in (0..key_end).step_by(col_step) {
            let col_chunk_end = (col_begin + col_step).min(key_end);
            AttentionTrait::compute(
                self,
                q_head_ptr,
                k_head_ptr,
                v_head_ptr,
                output_head_ptr,
                row_begin,
                visible_row_end,
                col_begin,
                col_chunk_end,
                col_end,
                sequence_index,
                self.k_seq_stride,
                self.v_seq_stride,
                self.q_seq_stride,
                scratch.running_max,
                scratch.running_denom,
                scratch.scores,
            );
        }
    }

    /// Visit blocks for a single attention head
    pub(super) unsafe fn visit_blocks_for_head(
        &self,
        q_head_ptr: *const T,
        output_head_ptr: *mut T,
        k_head_ptr: *const T,
        v_head_ptr: *const T,
        thread_id: usize,
        sequence_index: usize,
        col_end: usize,
        row_plan: RowVisitPlan,
    ) {
        if let Some((row_begin, row_end)) = row_plan.main {
            self.visit_aligned_row_range(
                q_head_ptr,
                output_head_ptr,
                k_head_ptr,
                v_head_ptr,
                thread_id,
                sequence_index,
                col_end,
                row_begin,
                row_end,
            );
        }

        if let Some((row_begin, row_end)) = row_plan.tail {
            self.visit_tail_row_range(
                q_head_ptr,
                output_head_ptr,
                k_head_ptr,
                v_head_ptr,
                thread_id,
                sequence_index,
                col_end,
                row_begin,
                row_end,
            );
        }
    }

    /// Run attention computation with sequence split strategy
    unsafe fn run_sequence_split(
        &self,
        q_slice_ptr: *const T,
        output_slice_ptr: *mut T,
        k_batch_ptr: *const T,
        v_batch_ptr: *const T,
        sequence_index: usize,
        col_end: usize,
        slice_len: usize,
        aligned_len: usize,
        thread_num: usize,
        thread_id: usize,
        k_head_stride: usize,
        v_head_stride: usize,
        attention_heads_per_kv: usize,
    ) {
        let row_plan = RowVisitPlan {
            main: split_sequence_by_triangle(aligned_len, self.row_step, thread_num, thread_id),
            tail: if aligned_len < slice_len && thread_num != 0 && thread_id + 1 == thread_num {
                Some((aligned_len, slice_len))
            } else {
                None
            },
        };

        for kv_head in 0..self.kv_head_num {
            let k_head_ptr = k_batch_ptr.add(kv_head * k_head_stride);
            let v_head_ptr = v_batch_ptr.add(kv_head * v_head_stride);
            if attention_heads_per_kv == 8 {
                let q_group_offset = kv_head * attention_heads_per_kv * self.head_size;
                let q_group_ptr = q_slice_ptr.add(q_group_offset);
                let output_group_ptr = output_slice_ptr.add(q_group_offset);
                let mut gqa8_handled = false;
                let mut gqa8_supported = true;

                if let Some((row_begin, row_end)) = row_plan.main {
                    gqa8_supported &= AttentionTrait::compute_gqa8(
                        self,
                        q_group_ptr,
                        k_head_ptr,
                        v_head_ptr,
                        output_group_ptr,
                        row_begin,
                        row_end,
                        col_end,
                        sequence_index,
                        self.k_seq_stride,
                        self.v_seq_stride,
                        self.q_seq_stride,
                    );
                    gqa8_handled = true;
                }

                if let Some((row_begin, row_end)) = row_plan.tail {
                    gqa8_supported &= AttentionTrait::compute_gqa8(
                        self,
                        q_group_ptr,
                        k_head_ptr,
                        v_head_ptr,
                        output_group_ptr,
                        row_begin,
                        row_end,
                        col_end,
                        sequence_index,
                        self.k_seq_stride,
                        self.v_seq_stride,
                        self.q_seq_stride,
                    );
                    gqa8_handled = true;
                }

                if gqa8_handled && gqa8_supported {
                    continue;
                }
            }

            for local_head in 0..attention_heads_per_kv {
                let attention_head = kv_head * attention_heads_per_kv + local_head;
                let q_head_offset = attention_head * self.head_size;
                let q_head_ptr = q_slice_ptr.add(q_head_offset);
                let output_head_ptr = output_slice_ptr.add(q_head_offset);

                self.visit_blocks_for_head(
                    q_head_ptr,
                    output_head_ptr,
                    k_head_ptr,
                    v_head_ptr,
                    thread_id,
                    sequence_index,
                    col_end,
                    row_plan,
                );
            }
        }
    }

    /// Run long prefill with a 2D static split:
    /// each task owns one KV group and a triangle-balanced row range.
    unsafe fn run_grouped_sequence_split(
        &self,
        q_slice_ptr: *const T,
        output_slice_ptr: *mut T,
        k_batch_ptr: *const T,
        v_batch_ptr: *const T,
        sequence_index: usize,
        col_end: usize,
        slice_len: usize,
        aligned_len: usize,
        thread_num: usize,
        thread_id: usize,
        k_head_stride: usize,
        v_head_stride: usize,
        attention_heads_per_kv: usize,
    ) {
        if thread_num == 0
            || thread_id >= thread_num
            || self.kv_head_num == 0
            || attention_heads_per_kv == 0
        {
            return;
        }

        let row_parts_per_kv = thread_num.div_ceil(self.kv_head_num).max(1);
        if row_parts_per_kv == 1 {
            self.run_sequence_split(
                q_slice_ptr,
                output_slice_ptr,
                k_batch_ptr,
                v_batch_ptr,
                sequence_index,
                col_end,
                slice_len,
                aligned_len,
                thread_num,
                thread_id,
                k_head_stride,
                v_head_stride,
                attention_heads_per_kv,
            );
            return;
        }

        let total_tasks = self.kv_head_num * row_parts_per_kv;
        let Some((task_begin, task_end)) =
            Self::split_contiguous_range(total_tasks, thread_num, thread_id)
        else {
            return;
        };

        for task_id in task_begin..task_end {
            let kv_head = task_id / row_parts_per_kv;
            let row_part = task_id % row_parts_per_kv;
            if kv_head >= self.kv_head_num {
                continue;
            }

            let row_plan = RowVisitPlan {
                main: split_sequence_by_triangle(
                    aligned_len,
                    self.row_step,
                    row_parts_per_kv,
                    row_part,
                ),
                tail: if aligned_len < slice_len && row_part + 1 == row_parts_per_kv {
                    Some((aligned_len, slice_len))
                } else {
                    None
                },
            };

            let k_head_ptr = k_batch_ptr.add(kv_head * k_head_stride);
            let v_head_ptr = v_batch_ptr.add(kv_head * v_head_stride);

            if attention_heads_per_kv == 8 {
                let q_group_offset = kv_head * attention_heads_per_kv * self.head_size;
                let q_group_ptr = q_slice_ptr.add(q_group_offset);
                let output_group_ptr = output_slice_ptr.add(q_group_offset);
                let mut gqa8_handled = false;
                let mut gqa8_supported = true;

                if let Some((row_begin, row_end)) = row_plan.main {
                    gqa8_supported &= AttentionTrait::compute_gqa8(
                        self,
                        q_group_ptr,
                        k_head_ptr,
                        v_head_ptr,
                        output_group_ptr,
                        row_begin,
                        row_end,
                        col_end,
                        sequence_index,
                        self.k_seq_stride,
                        self.v_seq_stride,
                        self.q_seq_stride,
                    );
                    gqa8_handled = true;
                }

                if let Some((row_begin, row_end)) = row_plan.tail {
                    gqa8_supported &= AttentionTrait::compute_gqa8(
                        self,
                        q_group_ptr,
                        k_head_ptr,
                        v_head_ptr,
                        output_group_ptr,
                        row_begin,
                        row_end,
                        col_end,
                        sequence_index,
                        self.k_seq_stride,
                        self.v_seq_stride,
                        self.q_seq_stride,
                    );
                    gqa8_handled = true;
                }

                if gqa8_handled && gqa8_supported {
                    continue;
                }
            }

            for local_head in 0..attention_heads_per_kv {
                let attention_head = kv_head * attention_heads_per_kv + local_head;
                let q_head_offset = attention_head * self.head_size;
                let q_head_ptr = q_slice_ptr.add(q_head_offset);
                let output_head_ptr = output_slice_ptr.add(q_head_offset);

                self.visit_blocks_for_head(
                    q_head_ptr,
                    output_head_ptr,
                    k_head_ptr,
                    v_head_ptr,
                    thread_id,
                    sequence_index,
                    col_end,
                    row_plan,
                );
            }
        }
    }

    /// Run attention computation with head split strategy
    unsafe fn run_head_split(
        &self,
        q_slice_ptr: *const T,
        output_slice_ptr: *mut T,
        k_batch_ptr: *const T,
        v_batch_ptr: *const T,
        sequence_index: usize,
        col_end: usize,
        slice_len: usize,
        aligned_len: usize,
        thread_num: usize,
        thread_id: usize,
        attention_heads_per_kv: usize,
        k_head_stride: usize,
        v_head_stride: usize,
    ) {
        if thread_num == 0 || thread_id >= thread_num || attention_heads_per_kv == 0 {
            return;
        }

        let active_thread_num = thread_num.min(self.attention_head_num);
        if thread_id >= active_thread_num || active_thread_num == 0 || self.kv_head_num == 0 {
            return;
        }

        let row_plan = RowVisitPlan {
            main: (aligned_len != 0).then_some((0, aligned_len)),
            tail: (aligned_len < slice_len).then_some((aligned_len, slice_len)),
        };

        if self.decode_gqa8 && slice_len == 1 && attention_heads_per_kv == 8 {
            let active_thread_num = thread_num.min(self.kv_head_num);
            if thread_id >= active_thread_num {
                return;
            }

            let Some((kv_begin, kv_end)) =
                Self::split_contiguous_range(self.kv_head_num, active_thread_num, thread_id)
            else {
                return;
            };

            for kv_head in kv_begin..kv_end {
                let k_head_ptr = k_batch_ptr.add(kv_head * k_head_stride);
                let v_head_ptr = v_batch_ptr.add(kv_head * v_head_stride);
                let q_group_offset = kv_head * attention_heads_per_kv * self.head_size;
                let q_group_ptr = q_slice_ptr.add(q_group_offset);
                let output_group_ptr = output_slice_ptr.add(q_group_offset);
                let mut handled = false;

                if let Some((row_begin, row_end)) = row_plan.main {
                    handled |= AttentionTrait::compute_gqa8(
                        self,
                        q_group_ptr,
                        k_head_ptr,
                        v_head_ptr,
                        output_group_ptr,
                        row_begin,
                        row_end,
                        col_end,
                        sequence_index,
                        self.k_seq_stride,
                        self.v_seq_stride,
                        self.q_seq_stride,
                    );
                }
                if let Some((row_begin, row_end)) = row_plan.tail {
                    handled |= AttentionTrait::compute_gqa8(
                        self,
                        q_group_ptr,
                        k_head_ptr,
                        v_head_ptr,
                        output_group_ptr,
                        row_begin,
                        row_end,
                        col_end,
                        sequence_index,
                        self.k_seq_stride,
                        self.v_seq_stride,
                        self.q_seq_stride,
                    );
                }

                if !handled {
                    for local_head in 0..attention_heads_per_kv {
                        let attention_head = kv_head * attention_heads_per_kv + local_head;
                        let q_head_offset = attention_head * self.head_size;
                        self.visit_blocks_for_head(
                            q_slice_ptr.add(q_head_offset),
                            output_slice_ptr.add(q_head_offset),
                            k_head_ptr,
                            v_head_ptr,
                            thread_id,
                            sequence_index,
                            col_end,
                            row_plan,
                        );
                    }
                }
            }
            return;
        }

        let kv_heads_per_wave = self.kv_heads_per_wave(active_thread_num, attention_heads_per_kv);
        if kv_heads_per_wave == 0 {
            return;
        }

        for kv_head_begin in (0..self.kv_head_num).step_by(kv_heads_per_wave) {
            let kv_head_end = (kv_head_begin + kv_heads_per_wave).min(self.kv_head_num);
            let slot_num = (kv_head_end - kv_head_begin) * attention_heads_per_kv;
            let active_threads_this_wave = active_thread_num.min(slot_num);

            if thread_id >= active_threads_this_wave {
                continue;
            }

            let Some((slot_begin, slot_end)) =
                Self::split_contiguous_range(slot_num, active_threads_this_wave, thread_id)
            else {
                continue;
            };

            for slot in slot_begin..slot_end {
                let kv_head = kv_head_begin + slot / attention_heads_per_kv;
                let local_head = slot % attention_heads_per_kv;
                let k_head_ptr = k_batch_ptr.add(kv_head * k_head_stride);
                let v_head_ptr = v_batch_ptr.add(kv_head * v_head_stride);
                let attention_head = kv_head * attention_heads_per_kv + local_head;
                let q_head_offset = attention_head * self.head_size;
                let q_head_ptr = q_slice_ptr.add(q_head_offset);
                let output_head_ptr = output_slice_ptr.add(q_head_offset);

                self.visit_blocks_for_head(
                    q_head_ptr,
                    output_head_ptr,
                    k_head_ptr,
                    v_head_ptr,
                    thread_id,
                    sequence_index,
                    col_end,
                    row_plan,
                );
            }
        }
    }

    /// Run prefill as independent `(query head, row block)` tasks.
    ///
    /// This follows the CPU FlashAttention scheduling used by SGLang/PyTorch:
    /// exposing row blocks in addition to heads keeps machines with more worker
    /// threads than query heads busy without serializing all GQA heads.
    unsafe fn run_head_row_split(
        &self,
        q_slice_ptr: *const T,
        output_slice_ptr: *mut T,
        k_batch_ptr: *const T,
        v_batch_ptr: *const T,
        sequence_index: usize,
        col_end: usize,
        slice_len: usize,
        thread_num: usize,
        thread_id: usize,
        attention_heads_per_kv: usize,
        k_head_stride: usize,
        v_head_stride: usize,
    ) {
        if slice_len == 0
            || thread_num == 0
            || thread_id >= thread_num
            || attention_heads_per_kv == 0
        {
            return;
        }

        let row_blocks = slice_len.div_ceil(self.row_step.max(1));
        let mut thread_loads = vec![0u128; thread_num];

        // Longest-processing-time assignment: later causal blocks are more
        // expensive, so distribute them first instead of handing each worker a
        // contiguous (and potentially much heavier) range.
        for row_block in (0..row_blocks).rev() {
            let row_begin = row_block * self.row_step;
            let row_end = (row_begin + self.row_step).min(slice_len);
            let key_end = if self.causal_col_prune {
                (sequence_index + row_end).min(col_end)
            } else {
                col_end
            };
            let task_weight = (row_end - row_begin) as u128 * key_end as u128;

            for head_slot in 0..self.attention_head_num {
                let attention_head = if self.rotate_head_assignment {
                    (head_slot + row_block) % self.attention_head_num
                } else {
                    head_slot
                };
                let mut owner = 0;
                for candidate in 1..thread_num {
                    if thread_loads[candidate] < thread_loads[owner] {
                        owner = candidate;
                    }
                }
                thread_loads[owner] += task_weight;
                if owner != thread_id {
                    continue;
                }

                let kv_head = attention_head / attention_heads_per_kv;
                let q_head_offset = attention_head * self.head_size;
                let k_head_ptr = k_batch_ptr.add(kv_head * k_head_stride);
                let v_head_ptr = v_batch_ptr.add(kv_head * v_head_stride);
                self.visit_blocks_for_head(
                    q_slice_ptr.add(q_head_offset),
                    output_slice_ptr.add(q_head_offset),
                    k_head_ptr,
                    v_head_ptr,
                    thread_id,
                    sequence_index,
                    col_end,
                    RowVisitPlan {
                        main: Some((row_begin, row_end)),
                        tail: None,
                    },
                );
            }
        }
    }

    /// Main entry point for running attention computation
    pub fn run(
        &self,
        _prefill_size: usize,
        _decode_size: usize,
        attention_list: &[SequenceSlice],
        thread_num: usize,
        thread_id: usize,
    ) {
        unsafe {
            let k_ptr = self.k_ptr.ptr;
            let v_ptr = self.v_ptr.ptr;
            let output_ptr = self.output_ptr.ptr;
            let q_ptr = self.q_ptr.ptr;
            let q_token_stride = self.attention_head_num * self.head_size;
            let attention_heads_per_kv = self.attention_head_num / self.kv_head_num;

            for slice in attention_list {
                if slice.batch_index >= self.batch_size {
                    continue;
                }

                let q_slice_ptr = q_ptr.add(slice.token_start_index * q_token_stride);
                let output_slice_ptr = output_ptr.add(slice.token_start_index * q_token_stride);
                let k_batch_ptr = k_ptr.add(slice.batch_index * self.k_batch_stride);
                let v_batch_ptr = v_ptr.add(slice.batch_index * self.v_batch_stride);
                let col_end = slice.sequence_index + slice.length;
                let aligned_len = slice.length / self.row_step * self.row_step;
                let use_head_split = self.force_head_split
                    || (slice.length > 0
                        && thread_num > 0
                        && slice.length.div_ceil(self.row_step.max(1)) < thread_num);
                let use_head_row_split =
                    self.head_row_split && self.brgemm_backend && slice.length > 1;

                let use_grouped_sequence_split = self.hybrid_split
                    && !use_head_split
                    && !use_head_row_split
                    && thread_num > self.kv_head_num
                    && self.kv_head_num > 1
                    && attention_heads_per_kv > 1
                    && slice.length >= thread_num;
                if self.profile_split && thread_id == 0 {
                    let mode = if use_head_row_split {
                        "head-row"
                    } else if use_head_split {
                        "head"
                    } else if use_grouped_sequence_split {
                        "hybrid"
                    } else {
                        "sequence"
                    };
                    let row_parts_per_kv = thread_num.div_ceil(self.kv_head_num.max(1)).max(1);
                    eprintln!(
                        "attention_split mode={mode} slice_len={} aligned_len={} thread_num={} q_heads={} kv_heads={} q_per_kv={} row_step={} col_step={} row_parts_per_kv={} hybrid_enabled={}",
                        slice.length,
                        aligned_len,
                        thread_num,
                        self.attention_head_num,
                        self.kv_head_num,
                        attention_heads_per_kv,
                        self.row_step,
                        self.col_step,
                        row_parts_per_kv,
                        self.hybrid_split
                    );
                }

                if use_head_row_split {
                    self.run_head_row_split(
                        q_slice_ptr,
                        output_slice_ptr,
                        k_batch_ptr,
                        v_batch_ptr,
                        slice.sequence_index,
                        col_end,
                        slice.length,
                        thread_num,
                        thread_id,
                        attention_heads_per_kv,
                        self.k_head_stride,
                        self.v_head_stride,
                    );
                } else if use_head_split {
                    self.run_head_split(
                        q_slice_ptr,
                        output_slice_ptr,
                        k_batch_ptr,
                        v_batch_ptr,
                        slice.sequence_index,
                        col_end,
                        slice.length,
                        aligned_len,
                        thread_num,
                        thread_id,
                        attention_heads_per_kv,
                        self.k_head_stride,
                        self.v_head_stride,
                    );
                } else if use_grouped_sequence_split {
                    self.run_grouped_sequence_split(
                        q_slice_ptr,
                        output_slice_ptr,
                        k_batch_ptr,
                        v_batch_ptr,
                        slice.sequence_index,
                        col_end,
                        slice.length,
                        aligned_len,
                        thread_num,
                        thread_id,
                        self.k_head_stride,
                        self.v_head_stride,
                        attention_heads_per_kv,
                    );
                } else {
                    self.run_sequence_split(
                        q_slice_ptr,
                        output_slice_ptr,
                        k_batch_ptr,
                        v_batch_ptr,
                        slice.sequence_index,
                        col_end,
                        slice.length,
                        aligned_len,
                        thread_num,
                        thread_id,
                        self.k_head_stride,
                        self.v_head_stride,
                        attention_heads_per_kv,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Attention;
    use crate::runtime::scheduling::SequenceSlice;

    #[test]
    fn head_row_split_assigns_every_task_once() {
        let query_heads = 32;
        let row_blocks = 10;
        let thread_num = 48;
        let total_tasks = query_heads * row_blocks;
        let mut visits = vec![0usize; total_tasks];

        for thread_id in 0..thread_num {
            if let Some((begin, end)) =
                Attention::<f32>::split_contiguous_range(total_tasks, thread_num, thread_id)
            {
                for task_id in begin..end {
                    visits[task_id] += 1;
                }
            }
        }

        assert!(visits.iter().all(|&count| count == 1));
    }

    fn naive_attention_row(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        head_size: usize,
        visible_col_end: usize,
        inverse_sqrt_head: f32,
    ) -> Vec<f32> {
        let mut max_score = f32::NEG_INFINITY;
        let mut scores = vec![0.0; visible_col_end];

        for col in 0..visible_col_end {
            let mut score = 0.0;
            for index in 0..head_size {
                score += q[index] * k[col * head_size + index];
            }
            score *= inverse_sqrt_head;
            scores[col] = score;
            if score > max_score {
                max_score = score;
            }
        }

        let mut denom = 0.0;
        for score in &mut scores {
            *score = (*score - max_score).exp();
            denom += *score;
        }

        let mut output = vec![0.0; head_size];
        for col in 0..visible_col_end {
            let weight = scores[col] / denom;
            for index in 0..head_size {
                output[index] += weight * v[col * head_size + index];
            }
        }
        output
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual_value, expected_value) in actual.iter().zip(expected.iter()) {
            assert!((actual_value - expected_value).abs() < 1e-5);
        }
    }

    #[test]
    fn sequence_split_computes_tail_rows() {
        let head_size = 2;
        let inverse_sqrt_head = 1.0;
        let q = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = vec![-1.0; q.len()];

        let mut attention = Attention::new(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            output.as_mut_ptr(),
            3,
            1,
            1,
            1,
            3 * head_size,
            3 * head_size,
            head_size,
            3 * head_size,
            3 * head_size,
            head_size,
            head_size,
            1,
            1,
            2,
            inverse_sqrt_head,
            false,
            1,
        );
        attention.causal_col_prune = true;

        let slices = [SequenceSlice {
            token_start_index: 0,
            batch_index: 0,
            sequence_index: 0,
            length: 3,
            last_token_flag: false,
        }];

        attention.run(0, 0, &slices, 1, 0);

        for row in 0..3 {
            let expected = naive_attention_row(
                &q[row * head_size..(row + 1) * head_size],
                &k,
                &v,
                head_size,
                row + 1,
                inverse_sqrt_head,
            );
            let actual = &output[row * head_size..(row + 1) * head_size];
            assert_close(actual, &expected);
        }
    }

    #[test]
    fn attention_reads_permuted_kv_with_strides() {
        let head_size = 2;
        let batch_size = 2;
        let seq_len = 3;
        let kv_heads = 1;
        let inverse_sqrt_head = 1.0;
        let q = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut k = vec![0.0; seq_len * batch_size * kv_heads * head_size];
        let mut v = vec![0.0; k.len()];

        for seq in 0..seq_len {
            for batch in 0..batch_size {
                let base = ((seq * batch_size + batch) * kv_heads) * head_size;
                k[base] = 10.0 * batch as f32 + seq as f32 + 1.0;
                k[base + 1] = 10.0 * batch as f32 + seq as f32 + 2.0;
                v[base] = 100.0 * batch as f32 + 10.0 * seq as f32 + 1.0;
                v[base + 1] = 100.0 * batch as f32 + 10.0 * seq as f32 + 2.0;
            }
        }

        let mut output = vec![-1.0; q.len()];
        let attention = Attention::new(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            output.as_mut_ptr(),
            seq_len,
            batch_size,
            1,
            kv_heads,
            kv_heads * head_size,
            head_size,
            batch_size * kv_heads * head_size,
            kv_heads * head_size,
            head_size,
            batch_size * kv_heads * head_size,
            head_size,
            1,
            2,
            head_size,
            inverse_sqrt_head,
            false,
            1,
        );

        let slice = [SequenceSlice {
            token_start_index: 0,
            batch_index: 1,
            sequence_index: 0,
            length: seq_len,
            last_token_flag: false,
        }];

        attention.run(0, 0, &slice, 1, 0);

        let batch1_k_base = kv_heads * head_size;
        let batch1_k = [
            k[batch1_k_base],
            k[batch1_k_base + 1],
            k[batch1_k_base + batch_size * kv_heads * head_size],
            k[batch1_k_base + batch_size * kv_heads * head_size + 1],
            k[batch1_k_base + 2 * batch_size * kv_heads * head_size],
            k[batch1_k_base + 2 * batch_size * kv_heads * head_size + 1],
        ];
        let batch1_v = [
            v[batch1_k_base],
            v[batch1_k_base + 1],
            v[batch1_k_base + batch_size * kv_heads * head_size],
            v[batch1_k_base + batch_size * kv_heads * head_size + 1],
            v[batch1_k_base + 2 * batch_size * kv_heads * head_size],
            v[batch1_k_base + 2 * batch_size * kv_heads * head_size + 1],
        ];

        for row in 0..seq_len {
            let expected = naive_attention_row(
                &q[row * head_size..(row + 1) * head_size],
                &batch1_k,
                &batch1_v,
                head_size,
                row + 1,
                inverse_sqrt_head,
            );
            let actual = &output[row * head_size..(row + 1) * head_size];
            assert_close(actual, &expected);
        }
    }
}
