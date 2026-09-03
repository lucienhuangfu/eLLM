use std::f16;
// use num::{complex::ComplexFloat, Zero};
use std::arch::x86_64::{
    __m512, __m512h, _MM_HINT_T2, _mm_prefetch, _mm512_add_ps, _mm512_castph_si512,
    _mm512_castsi512_si256, _mm512_cvtph_ps, _mm512_div_ph, _mm512_extracti64x4_epi64,
    _mm512_fmadd_ph, _mm512_fmadd_ps, _mm512_load_ph, _mm512_loadu_ph, _mm512_mul_ph,
    _mm512_reduce_add_ph, _mm512_reduce_add_ps, _mm512_set1_ph, _mm512_setzero_ph,
    _mm512_setzero_ps, _mm512_store_ph, _mm512_storeu_ph, _mm512_sub_ph,
}; // Add this line
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::kernel::x86_64::f16_512::activation::exp512;

use super::dot_product::_dot_product;

const GQA8_COL_BLOCK: usize = 32;

struct AttentionKernelProfile {
    regular_qk_ns: AtomicU64,
    regular_softmax_ns: AtomicU64,
    regular_value_ns: AtomicU64,
    regular_blocks: AtomicU64,
    regular_rows: AtomicU64,
    gqa8_clear_ns: AtomicU64,
    gqa8_qk_ns: AtomicU64,
    gqa8_softmax_ns: AtomicU64,
    gqa8_value_ns: AtomicU64,
    gqa8_blocks: AtomicU64,
    gqa8_rows: AtomicU64,
}

impl AttentionKernelProfile {
    const fn new() -> Self {
        Self {
            regular_qk_ns: AtomicU64::new(0),
            regular_softmax_ns: AtomicU64::new(0),
            regular_value_ns: AtomicU64::new(0),
            regular_blocks: AtomicU64::new(0),
            regular_rows: AtomicU64::new(0),
            gqa8_clear_ns: AtomicU64::new(0),
            gqa8_qk_ns: AtomicU64::new(0),
            gqa8_softmax_ns: AtomicU64::new(0),
            gqa8_value_ns: AtomicU64::new(0),
            gqa8_blocks: AtomicU64::new(0),
            gqa8_rows: AtomicU64::new(0),
        }
    }

    fn reset(&self) {
        self.regular_qk_ns.store(0, Ordering::Relaxed);
        self.regular_softmax_ns.store(0, Ordering::Relaxed);
        self.regular_value_ns.store(0, Ordering::Relaxed);
        self.regular_blocks.store(0, Ordering::Relaxed);
        self.regular_rows.store(0, Ordering::Relaxed);
        self.gqa8_clear_ns.store(0, Ordering::Relaxed);
        self.gqa8_qk_ns.store(0, Ordering::Relaxed);
        self.gqa8_softmax_ns.store(0, Ordering::Relaxed);
        self.gqa8_value_ns.store(0, Ordering::Relaxed);
        self.gqa8_blocks.store(0, Ordering::Relaxed);
        self.gqa8_rows.store(0, Ordering::Relaxed);
    }
}

static ATTENTION_KERNEL_PROFILE: AttentionKernelProfile = AttentionKernelProfile::new();
static ATTENTION_KERNEL_PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();

#[inline(always)]
fn attention_kernel_profile_enabled() -> bool {
    *ATTENTION_KERNEL_PROFILE_ENABLED.get_or_init(|| {
        std::env::var("ELLM_PROFILE_ATTENTION_KERNEL")
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false)
    })
}

#[inline(always)]
fn add_elapsed_ns(counter: &AtomicU64, start: Instant) {
    counter.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

pub fn reset_attention_kernel_profile() {
    if attention_kernel_profile_enabled() {
        ATTENTION_KERNEL_PROFILE.reset();
    }
}

pub fn print_attention_kernel_profile(label: &str) {
    if !attention_kernel_profile_enabled() {
        return;
    }

    let regular_qk = ATTENTION_KERNEL_PROFILE
        .regular_qk_ns
        .load(Ordering::Relaxed) as f64
        / 1e9;
    let regular_softmax = ATTENTION_KERNEL_PROFILE
        .regular_softmax_ns
        .load(Ordering::Relaxed) as f64
        / 1e9;
    let regular_value = ATTENTION_KERNEL_PROFILE
        .regular_value_ns
        .load(Ordering::Relaxed) as f64
        / 1e9;
    let gqa8_clear = ATTENTION_KERNEL_PROFILE
        .gqa8_clear_ns
        .load(Ordering::Relaxed) as f64
        / 1e9;
    let gqa8_qk = ATTENTION_KERNEL_PROFILE.gqa8_qk_ns.load(Ordering::Relaxed) as f64 / 1e9;
    let gqa8_softmax = ATTENTION_KERNEL_PROFILE
        .gqa8_softmax_ns
        .load(Ordering::Relaxed) as f64
        / 1e9;
    let gqa8_value = ATTENTION_KERNEL_PROFILE
        .gqa8_value_ns
        .load(Ordering::Relaxed) as f64
        / 1e9;
    let regular_total = regular_qk + regular_softmax + regular_value;
    let gqa8_total = gqa8_clear + gqa8_qk + gqa8_softmax + gqa8_value;

    eprintln!(
        "attention_kernel_profile label={label} regular_total={regular_total:.6}s regular_qk={regular_qk:.6}s regular_softmax={regular_softmax:.6}s regular_value={regular_value:.6}s regular_blocks={} regular_rows={} gqa8_total={gqa8_total:.6}s gqa8_clear={gqa8_clear:.6}s gqa8_qk={gqa8_qk:.6}s gqa8_softmax={gqa8_softmax:.6}s gqa8_value={gqa8_value:.6}s gqa8_blocks={} gqa8_rows={}",
        ATTENTION_KERNEL_PROFILE
            .regular_blocks
            .load(Ordering::Relaxed),
        ATTENTION_KERNEL_PROFILE
            .regular_rows
            .load(Ordering::Relaxed),
        ATTENTION_KERNEL_PROFILE.gqa8_blocks.load(Ordering::Relaxed),
        ATTENTION_KERNEL_PROFILE.gqa8_rows.load(Ordering::Relaxed)
    );
}

// 标量和向量相乘并累加到结果向量
#[inline(always)]
unsafe fn scalar_vector_mul_and_acc(
    scalar1: f16,
    o: &mut [__m512h; 4],
    scalar2: f16,
    v: *const f16,
) {
    let scalar1_chunk = _mm512_set1_ph(scalar1);
    let scalar2_chunk = _mm512_set1_ph(scalar2);

    for (offset, res_chunk) in (0..128).step_by(32).zip(o.iter_mut()) {
        let v_chunk = _mm512_load_ph(v.add(offset));
        // let res1_chunk = _mm512_mul_ph(scalar1_chunk, *res_chunk);
        let res2_chunk = _mm512_mul_ph(scalar2_chunk, v_chunk);
        // *res_chunk = _mm512_add_ph(res1_chunk, res2_chunk);
        *res_chunk = _mm512_fmadd_ph(scalar1_chunk, *res_chunk, res2_chunk);
    }
}
#[inline(always)]
pub fn flash_attention(
    q: *const f16,
    K: *const f16,
    V: *const f16,
    o: *mut f16,
    inverse_sqrt_head: f16,
    length: usize,
    stride: usize,
    position: usize,
) {
    unsafe {
        // 预取下一块数据 (4 cache lines ahead)
        _mm_prefetch(K as *const i8, _MM_HINT_T2);
        _mm_prefetch(K.add(64) as *const i8, _MM_HINT_T2);
        _mm_prefetch(V as *const i8, _MM_HINT_T2);
        _mm_prefetch(V.add(64) as *const i8, _MM_HINT_T2);

        let mut m_i_1 = f16::NEG_INFINITY;
        let mut d_i_1: f16 = 0.0;

        // 初始化结果寄存器为0
        let mut o_chunks = [
            _mm512_setzero_ph(),
            _mm512_setzero_ph(),
            _mm512_setzero_ph(),
            _mm512_setzero_ph(),
        ];

        let end = position * stride;
        for offset in (0..=end).step_by(stride) {
            // 提前预取下一次循环的数据
            if offset < end {
                let next_offset = offset + stride;
                _mm_prefetch(K.add(next_offset) as *const i8, _MM_HINT_T2);
                _mm_prefetch(K.add(next_offset + 64) as *const i8, _MM_HINT_T2);
                _mm_prefetch(V.add(next_offset) as *const i8, _MM_HINT_T2);
                _mm_prefetch(V.add(next_offset + 64) as *const i8, _MM_HINT_T2);
            }

            // dot product
            let x_i = _dot_product(q, K.add(offset), length);
            // 计算缩放后的 x_i
            let x_i = x_i * inverse_sqrt_head;
            let m_i = if x_i > m_i_1 { x_i } else { m_i_1 };

            // 合并 update_exp 和 add_exp 的计算
            let update_exp = d_i_1 * f16::exp(m_i_1 - m_i);
            let add_exp = f16::exp(x_i - m_i);
            let d_i = update_exp + add_exp;
            let u = update_exp / d_i;
            let a = add_exp / d_i;

            scalar_vector_mul_and_acc(u, &mut o_chunks, a, V.add(offset));

            m_i_1 = m_i;
            d_i_1 = d_i;
        }

        for (step, res_chunk) in (0..128).step_by(32).zip(o_chunks.into_iter()) {
            _mm512_store_ph(o.add(step), res_chunk);
        }
    }
}

#[inline(always)]
unsafe fn dot_product_avx512(q: *const f16, k: *const f16, head_size: usize) -> f16 {
    let simd_end = head_size / 32 * 32;
    let mut acc_lower = _mm512_setzero_ps();
    let mut acc_upper = _mm512_setzero_ps();

    for index in (0..simd_end).step_by(32) {
        let q_chunk = _mm512_loadu_ph(q.add(index) as *const _);
        let k_chunk = _mm512_loadu_ph(k.add(index) as *const _);
        let q_i = _mm512_castph_si512(q_chunk);
        let k_i = _mm512_castph_si512(k_chunk);

        let q_lower = _mm512_cvtph_ps(_mm512_castsi512_si256(q_i));
        let q_upper = _mm512_cvtph_ps(_mm512_extracti64x4_epi64::<1>(q_i));
        let k_lower = _mm512_cvtph_ps(_mm512_castsi512_si256(k_i));
        let k_upper = _mm512_cvtph_ps(_mm512_extracti64x4_epi64::<1>(k_i));

        acc_lower = _mm512_fmadd_ps(q_lower, k_lower, acc_lower);
        acc_upper = _mm512_fmadd_ps(q_upper, k_upper, acc_upper);
    }

    let mut sum = _mm512_reduce_add_ps(_mm512_add_ps(acc_lower, acc_upper));

    for index in simd_end..head_size {
        sum += (*q.add(index) as f32) * (*k.add(index) as f32);
    }
    sum as f16
}

#[inline(always)]
unsafe fn dot_product_gqa8_avx512(
    q_group: *const f16,
    key_row: *const f16,
    head_size: usize,
) -> [f16; 8] {
    let simd_end = head_size / 32 * 32;
    let mut acc_lower = [_mm512_setzero_ps(); 8];
    let mut acc_upper = [_mm512_setzero_ps(); 8];

    for index in (0..simd_end).step_by(32) {
        let k_chunk = _mm512_loadu_ph(key_row.add(index) as *const _);
        let k_i = _mm512_castph_si512(k_chunk);
        let k_lower = _mm512_cvtph_ps(_mm512_castsi512_si256(k_i));
        let k_upper = _mm512_cvtph_ps(_mm512_extracti64x4_epi64::<1>(k_i));

        for local_head in 0..8 {
            let q = q_group.add(local_head * head_size + index);
            let q_chunk = _mm512_loadu_ph(q as *const _);
            let q_i = _mm512_castph_si512(q_chunk);
            let q_lower = _mm512_cvtph_ps(_mm512_castsi512_si256(q_i));
            let q_upper = _mm512_cvtph_ps(_mm512_extracti64x4_epi64::<1>(q_i));

            acc_lower[local_head] = _mm512_fmadd_ps(q_lower, k_lower, acc_lower[local_head]);
            acc_upper[local_head] = _mm512_fmadd_ps(q_upper, k_upper, acc_upper[local_head]);
        }
    }

    let mut sums = [0.0f32; 8];
    for local_head in 0..8 {
        sums[local_head] =
            _mm512_reduce_add_ps(_mm512_add_ps(acc_lower[local_head], acc_upper[local_head]));
    }

    for index in simd_end..head_size {
        let k_item = *key_row.add(index) as f32;
        for local_head in 0..8 {
            sums[local_head] += (*q_group.add(local_head * head_size + index) as f32) * k_item;
        }
    }

    [
        sums[0] as f16,
        sums[1] as f16,
        sums[2] as f16,
        sums[3] as f16,
        sums[4] as f16,
        sums[5] as f16,
        sums[6] as f16,
        sums[7] as f16,
    ]
}

#[inline(always)]
unsafe fn preload_q_gqa8_head128(q_group: *const f16) -> ([[__m512; 4]; 8], [[__m512; 4]; 8]) {
    let mut q_lower = [[_mm512_setzero_ps(); 4]; 8];
    let mut q_upper = [[_mm512_setzero_ps(); 4]; 8];

    for local_head in 0..8 {
        for chunk in 0..4 {
            let q = q_group.add(local_head * 128 + chunk * 32);
            let q_chunk = _mm512_loadu_ph(q as *const _);
            let q_i = _mm512_castph_si512(q_chunk);
            q_lower[local_head][chunk] = _mm512_cvtph_ps(_mm512_castsi512_si256(q_i));
            q_upper[local_head][chunk] = _mm512_cvtph_ps(_mm512_extracti64x4_epi64::<1>(q_i));
        }
    }

    (q_lower, q_upper)
}

#[inline(always)]
unsafe fn dot_product_gqa8_preloaded_q128_avx512(
    q_lower: &[[__m512; 4]; 8],
    q_upper: &[[__m512; 4]; 8],
    key_row: *const f16,
) -> [f16; 8] {
    let mut acc_lower = [_mm512_setzero_ps(); 8];
    let mut acc_upper = [_mm512_setzero_ps(); 8];

    for chunk in 0..4 {
        let k_chunk = _mm512_loadu_ph(key_row.add(chunk * 32) as *const _);
        let k_i = _mm512_castph_si512(k_chunk);
        let k_lower = _mm512_cvtph_ps(_mm512_castsi512_si256(k_i));
        let k_upper = _mm512_cvtph_ps(_mm512_extracti64x4_epi64::<1>(k_i));

        for local_head in 0..8 {
            acc_lower[local_head] =
                _mm512_fmadd_ps(q_lower[local_head][chunk], k_lower, acc_lower[local_head]);
            acc_upper[local_head] =
                _mm512_fmadd_ps(q_upper[local_head][chunk], k_upper, acc_upper[local_head]);
        }
    }

    let mut sums = [0.0f32; 8];
    for local_head in 0..8 {
        sums[local_head] =
            _mm512_reduce_add_ps(_mm512_add_ps(acc_lower[local_head], acc_upper[local_head]));
    }

    [
        sums[0] as f16,
        sums[1] as f16,
        sums[2] as f16,
        sums[3] as f16,
        sums[4] as f16,
        sums[5] as f16,
        sums[6] as f16,
        sums[7] as f16,
    ]
}

#[inline(always)]
unsafe fn scale_output_avx512(output: *mut f16, head_size: usize, scale: f16) {
    let simd_end = head_size / 32 * 32;
    let scale_chunk = _mm512_set1_ph(scale);

    for offset in (0..simd_end).step_by(32) {
        let output_chunk = _mm512_loadu_ph(output.add(offset));
        _mm512_storeu_ph(output.add(offset), _mm512_mul_ph(output_chunk, scale_chunk));
    }

    for index in simd_end..head_size {
        *output.add(index) *= scale;
    }
}

#[inline(always)]
unsafe fn add_weighted_value_avx512(
    output: *mut f16,
    value: *const f16,
    head_size: usize,
    weight: f16,
) {
    let simd_end = head_size / 32 * 32;
    let weight_chunk = _mm512_set1_ph(weight);

    for offset in (0..simd_end).step_by(32) {
        let output_chunk = _mm512_loadu_ph(output.add(offset));
        let value_chunk = _mm512_loadu_ph(value.add(offset));
        let next_output = _mm512_fmadd_ph(value_chunk, weight_chunk, output_chunk);
        _mm512_storeu_ph(output.add(offset), next_output);
    }

    for index in simd_end..head_size {
        *output.add(index) += *value.add(index) * weight;
    }
}

#[inline(always)]
unsafe fn clear_output_avx512(output: *mut f16, head_size: usize) {
    let simd_end = head_size / 32 * 32;
    let zero = _mm512_setzero_ph();

    for offset in (0..simd_end).step_by(32) {
        _mm512_storeu_ph(output.add(offset), zero);
    }

    for index in simd_end..head_size {
        *output.add(index) = 0.0;
    }
}

#[inline(always)]
unsafe fn add_weighted_value_gqa8_avx512(
    output_group: *mut f16,
    value: *const f16,
    head_size: usize,
    weights: &[f16; 8],
) {
    let simd_end = head_size / 32 * 32;
    let weight_chunks = [
        _mm512_set1_ph(weights[0]),
        _mm512_set1_ph(weights[1]),
        _mm512_set1_ph(weights[2]),
        _mm512_set1_ph(weights[3]),
        _mm512_set1_ph(weights[4]),
        _mm512_set1_ph(weights[5]),
        _mm512_set1_ph(weights[6]),
        _mm512_set1_ph(weights[7]),
    ];

    for offset in (0..simd_end).step_by(32) {
        let value_chunk = _mm512_loadu_ph(value.add(offset));
        for local_head in 0..8 {
            let output = output_group.add(local_head * head_size + offset);
            let output_chunk = _mm512_loadu_ph(output);
            let next_output = _mm512_fmadd_ph(value_chunk, weight_chunks[local_head], output_chunk);
            _mm512_storeu_ph(output, next_output);
        }
    }

    for index in simd_end..head_size {
        let value_item = *value.add(index);
        for local_head in 0..8 {
            *output_group.add(local_head * head_size + index) += value_item * weights[local_head];
        }
    }
}

#[inline(always)]
unsafe fn add_weighted_values_gqa8_head128_avx512(
    output_group: *mut f16,
    v_head_ptr: *const f16,
    col_begin: usize,
    block_len: usize,
    v_seq_stride: usize,
    weights_by_head: &[[f16; GQA8_COL_BLOCK]; 8],
) {
    for chunk in 0..4 {
        let offset = chunk * 32;
        let mut acc = [
            _mm512_loadu_ph(output_group.add(offset)),
            _mm512_loadu_ph(output_group.add(128 + offset)),
            _mm512_loadu_ph(output_group.add(256 + offset)),
            _mm512_loadu_ph(output_group.add(384 + offset)),
            _mm512_loadu_ph(output_group.add(512 + offset)),
            _mm512_loadu_ph(output_group.add(640 + offset)),
            _mm512_loadu_ph(output_group.add(768 + offset)),
            _mm512_loadu_ph(output_group.add(896 + offset)),
        ];

        for row_offset in 0..block_len {
            let value =
                _mm512_loadu_ph(v_head_ptr.add((col_begin + row_offset) * v_seq_stride + offset));
            for local_head in 0..8 {
                let weight = _mm512_set1_ph(weights_by_head[local_head][row_offset]);
                acc[local_head] = _mm512_fmadd_ph(value, weight, acc[local_head]);
            }
        }

        _mm512_storeu_ph(output_group.add(offset), acc[0]);
        _mm512_storeu_ph(output_group.add(128 + offset), acc[1]);
        _mm512_storeu_ph(output_group.add(256 + offset), acc[2]);
        _mm512_storeu_ph(output_group.add(384 + offset), acc[3]);
        _mm512_storeu_ph(output_group.add(512 + offset), acc[4]);
        _mm512_storeu_ph(output_group.add(640 + offset), acc[5]);
        _mm512_storeu_ph(output_group.add(768 + offset), acc[6]);
        _mm512_storeu_ph(output_group.add(896 + offset), acc[7]);
    }
}

#[inline(always)]
pub unsafe fn block_flash_attention_gqa8(
    q_group_ptr: *const f16,
    output_group_ptr: *mut f16,
    row_begin: usize,
    row_end: usize,
    total_col_end: usize,
    k_head_ptr: *const f16,
    v_head_ptr: *const f16,
    k_seq_stride: usize,
    v_seq_stride: usize,
    q_seq_stride: usize,
    head_size: usize,
    inverse_sqrt_head: f16,
    sequence_index: usize,
) {
    let profile = attention_kernel_profile_enabled();
    for row in row_begin..row_end {
        let visible_col_end = (sequence_index + row + 1).min(total_col_end);
        if visible_col_end == 0 {
            continue;
        }
        if profile {
            ATTENTION_KERNEL_PROFILE
                .gqa8_rows
                .fetch_add(1, Ordering::Relaxed);
        }

        let q_row_group_ptr = q_group_ptr.add(row * q_seq_stride);
        let output_row_group_ptr = output_group_ptr.add(row * q_seq_stride);
        let profile_start = profile.then(Instant::now);
        for local_head in 0..8 {
            clear_output_avx512(output_row_group_ptr.add(local_head * head_size), head_size);
        }
        if let Some(start) = profile_start {
            add_elapsed_ns(&ATTENTION_KERNEL_PROFILE.gqa8_clear_ns, start);
        }

        let mut running_max = [f16::NEG_INFINITY; 8];
        let mut running_denom = [0.0f16; 8];
        let mut scores = [[0.0f16; GQA8_COL_BLOCK]; 8];
        let preloaded_q = (head_size == 128).then(|| preload_q_gqa8_head128(q_row_group_ptr));

        for col_begin in (0..visible_col_end).step_by(GQA8_COL_BLOCK) {
            if profile {
                ATTENTION_KERNEL_PROFILE
                    .gqa8_blocks
                    .fetch_add(1, Ordering::Relaxed);
            }
            let row_col_end = (col_begin + GQA8_COL_BLOCK).min(visible_col_end);
            let block_len = row_col_end - col_begin;
            let mut block_max = [f16::NEG_INFINITY; 8];

            let profile_start = profile.then(Instant::now);
            for offset in 0..block_len {
                let key_row_ptr = k_head_ptr.add((col_begin + offset) * k_seq_stride);
                let score_group = if let Some((q_lower, q_upper)) = preloaded_q.as_ref() {
                    dot_product_gqa8_preloaded_q128_avx512(q_lower, q_upper, key_row_ptr)
                } else {
                    dot_product_gqa8_avx512(q_row_group_ptr, key_row_ptr, head_size)
                };
                for local_head in 0..8 {
                    let score = score_group[local_head] * inverse_sqrt_head;
                    scores[local_head][offset] = score;
                    if score > block_max[local_head] {
                        block_max[local_head] = score;
                    }
                }
            }
            if let Some(start) = profile_start {
                add_elapsed_ns(&ATTENTION_KERNEL_PROFILE.gqa8_qk_ns, start);
            }
            for offset in block_len..GQA8_COL_BLOCK {
                for local_head in 0..8 {
                    scores[local_head][offset] = f16::NEG_INFINITY;
                }
            }

            let mut weights_by_head = [[0.0f16; GQA8_COL_BLOCK]; 8];
            let profile_start = profile.then(Instant::now);
            for local_head in 0..8 {
                let next_max = if block_max[local_head] > running_max[local_head] {
                    block_max[local_head]
                } else {
                    running_max[local_head]
                };

                let carry =
                    running_denom[local_head] * f16::exp(running_max[local_head] - next_max);
                let shifted = _mm512_sub_ph(
                    _mm512_loadu_ph(scores[local_head].as_ptr()),
                    _mm512_set1_ph(next_max),
                );
                let exp_scores = exp512(shifted);
                let next_denom = carry + _mm512_reduce_add_ph(exp_scores);

                let previous_weight = carry / next_denom;
                scale_output_avx512(
                    output_row_group_ptr.add(local_head * head_size),
                    head_size,
                    previous_weight,
                );

                let normalized = _mm512_div_ph(exp_scores, _mm512_set1_ph(next_denom));
                _mm512_storeu_ph(weights_by_head[local_head].as_mut_ptr(), normalized);

                running_max[local_head] = next_max;
                running_denom[local_head] = next_denom;
            }
            if let Some(start) = profile_start {
                add_elapsed_ns(&ATTENTION_KERNEL_PROFILE.gqa8_softmax_ns, start);
            }

            let profile_start = profile.then(Instant::now);
            if head_size == 128 {
                add_weighted_values_gqa8_head128_avx512(
                    output_row_group_ptr,
                    v_head_ptr,
                    col_begin,
                    block_len,
                    v_seq_stride,
                    &weights_by_head,
                );
            } else {
                for offset in 0..block_len {
                    let value_row_ptr = v_head_ptr.add((col_begin + offset) * v_seq_stride);
                    let weights = [
                        weights_by_head[0][offset],
                        weights_by_head[1][offset],
                        weights_by_head[2][offset],
                        weights_by_head[3][offset],
                        weights_by_head[4][offset],
                        weights_by_head[5][offset],
                        weights_by_head[6][offset],
                        weights_by_head[7][offset],
                    ];
                    add_weighted_value_gqa8_avx512(
                        output_row_group_ptr,
                        value_row_ptr,
                        head_size,
                        &weights,
                    );
                }
            }
            if let Some(start) = profile_start {
                add_elapsed_ns(&ATTENTION_KERNEL_PROFILE.gqa8_value_ns, start);
            }
        }
    }
}

#[inline(always)]
pub unsafe fn block_flash_attention(
    q_head_ptr: *const f16,
    output_head_ptr: *mut f16,
    row_begin: usize,
    row_end: usize,
    col_begin: usize,
    col_end: usize,
    total_col_end: usize,
    k_head_ptr: *const f16,
    v_head_ptr: *const f16,
    k_seq_stride: usize,
    v_seq_stride: usize,
    q_seq_stride: usize,
    head_size: usize,
    inverse_sqrt_head: f16,
    sequence_index: usize,
    running_max: &mut [f16],
    running_denom: &mut [f16],
    scores: &mut [f16],
) {
    let profile = attention_kernel_profile_enabled();
    for (row_offset, row) in (row_begin..row_end).enumerate() {
        let visible_col_end = (sequence_index + row + 1).min(total_col_end);
        let row_col_end = col_end.min(visible_col_end);
        if col_begin >= row_col_end {
            continue;
        }
        if profile {
            ATTENTION_KERNEL_PROFILE
                .regular_rows
                .fetch_add(1, Ordering::Relaxed);
            ATTENTION_KERNEL_PROFILE
                .regular_blocks
                .fetch_add(1, Ordering::Relaxed);
        }

        let q_offset = row * q_seq_stride;
        let q_row_ptr = q_head_ptr.add(q_offset);
        let output_row_ptr = output_head_ptr.add(q_offset);
        let block_len = row_col_end - col_begin;
        let mut block_max = f16::NEG_INFINITY;

        let profile_start = profile.then(Instant::now);
        for offset in 0..block_len {
            let col = col_begin + offset;
            let key_row_ptr = k_head_ptr.add(col * k_seq_stride);
            let score = dot_product_avx512(q_row_ptr, key_row_ptr, head_size) * inverse_sqrt_head;
            scores[offset] = score;
            if score > block_max {
                block_max = score;
            }
        }
        if let Some(start) = profile_start {
            add_elapsed_ns(&ATTENTION_KERNEL_PROFILE.regular_qk_ns, start);
        }
        let next_max = if block_max > running_max[row_offset] {
            block_max
        } else {
            running_max[row_offset]
        };

        let carry = running_denom[row_offset] * f16::exp(running_max[row_offset] - next_max);
        let next_denom = if scores.len() >= GQA8_COL_BLOCK {
            let profile_start = profile.then(Instant::now);
            for offset in block_len..GQA8_COL_BLOCK {
                scores[offset] = f16::NEG_INFINITY;
            }
            let shifted = _mm512_sub_ph(_mm512_loadu_ph(scores.as_ptr()), _mm512_set1_ph(next_max));
            let exp_scores = exp512(shifted);
            let next_denom = carry + _mm512_reduce_add_ph(exp_scores);

            let previous_weight = carry / next_denom;
            scale_output_avx512(output_row_ptr, head_size, previous_weight);

            let normalized = _mm512_div_ph(exp_scores, _mm512_set1_ph(next_denom));
            let mut normalized_scores = [0.0f16; GQA8_COL_BLOCK];
            _mm512_storeu_ph(normalized_scores.as_mut_ptr(), normalized);
            if let Some(start) = profile_start {
                add_elapsed_ns(&ATTENTION_KERNEL_PROFILE.regular_softmax_ns, start);
            }

            let profile_start = profile.then(Instant::now);
            for offset in 0..block_len {
                let col = col_begin + offset;
                let value_row_ptr = v_head_ptr.add(col * v_seq_stride);
                add_weighted_value_avx512(
                    output_row_ptr,
                    value_row_ptr,
                    head_size,
                    normalized_scores[offset],
                );
            }
            if let Some(start) = profile_start {
                add_elapsed_ns(&ATTENTION_KERNEL_PROFILE.regular_value_ns, start);
            }
            next_denom
        } else {
            let profile_start = profile.then(Instant::now);
            let mut next_denom = carry;
            for offset in 0..block_len {
                next_denom += f16::exp(scores[offset] - next_max);
            }

            let previous_weight = carry / next_denom;
            scale_output_avx512(output_row_ptr, head_size, previous_weight);
            if let Some(start) = profile_start {
                add_elapsed_ns(&ATTENTION_KERNEL_PROFILE.regular_softmax_ns, start);
            }

            let profile_start = profile.then(Instant::now);
            for offset in 0..block_len {
                let col = col_begin + offset;
                let value_row_ptr = v_head_ptr.add(col * v_seq_stride);
                let weight = f16::exp(scores[offset] - next_max) / next_denom;
                add_weighted_value_avx512(output_row_ptr, value_row_ptr, head_size, weight);
            }
            if let Some(start) = profile_start {
                add_elapsed_ns(&ATTENTION_KERNEL_PROFILE.regular_value_ns, start);
            }
            next_denom
        };

        running_max[row_offset] = next_max;
        running_denom[row_offset] = next_denom;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::slice;
    // use std::f16;
    use crate::mem_mgr::allocator::AlignedBox;

    #[test]
    fn test_scalar_vector_mul_and_acc() {
        let scalar1 = 1.0;
        let scalar2 = 2.0;
        // let v: Vec<f16> = vec![2.0; 128];
        let length = 128;
        let v = AlignedBox::allocate_init(length, 2.0f16);
        let mut o = unsafe {
            [
                _mm512_set1_ph(1.0),
                _mm512_set1_ph(1.0),
                _mm512_set1_ph(1.0),
                _mm512_set1_ph(1.0),
            ]
        };
        unsafe {
            scalar_vector_mul_and_acc(scalar1, &mut o, scalar2, v.as_ptr());
        }
        for i in 0..4 {
            // let mut res: Vec<f16> = [0.0; 32].into_iter().map(|x| x).collect();
            let mut res = AlignedBox::allocate_init(length, 0.0f16);
            let res_slice = unsafe { slice::from_raw_parts(res.as_ptr(), 32) };
            unsafe {
                _mm512_store_ph(res.as_mut_ptr(), o[i]);
            }
            let mut exp: Vec<f16> = vec![5.0; 32];
            // unsafe {
            //    _mm512_store_ph(exp.as_mut_ptr(), expected[i]);
            // }
            for j in 0..32 {
                assert!(f16::abs(res_slice[j] - exp[j]) < 1e-6);
            }
        }
    }

    #[test]
    fn test_flash_attention() {
        let row_size = 8;
        let length = 128;

        // let q: Vec<f16> = vec![1.0; length];
        // let K: Vec<f16> = vec![1.0; row_size * length];
        // let V: Vec<f16> = vec![1.0; row_size * length];
        // let mut o: Vec<f16> = vec![0.0; length];

        let q = AlignedBox::allocate_init(length, 1.0f16);
        let K = AlignedBox::allocate_init(row_size * length, 1.0f16);
        let V = AlignedBox::allocate_init(row_size * length, 1.0f16);
        let mut o = AlignedBox::allocate_init(length, 0.0f16);
        let o_slice = unsafe { slice::from_raw_parts(o.as_ptr(), length) };
        flash_attention(
            q.as_ptr(),
            K.as_ptr(),
            V.as_ptr(),
            o.as_mut_ptr(),
            1.0,
            length,
            length,
            row_size - 1,
        );
        println!("Result: {:?}", o);

        let expected: Vec<f16> = vec![1.0; length];
        for i in 0..length {
            assert!((o_slice[i] - expected[i]).abs() < 1e-6);
        }
    }
}
