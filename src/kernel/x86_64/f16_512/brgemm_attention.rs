//! Blocked fp16 attention using LibTorch's low-level CPU BRGEMM primitive.
//!
//! This is an eLLM kernel: it owns packing, causal masking, online softmax and
//! scheduling.  LibTorch is loaded dynamically only for its half GEMM microkernel.

use std::arch::x86_64::{
    __m512, _mm256_storeu_si256, _mm512_add_ps, _mm512_castsi512_ps, _mm512_cmp_ps_mask,
    _mm512_cvtps_ph, _mm512_cvttps_epi32, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_mask_mov_epi32,
    _mm512_max_ps, _mm512_mul_ps, _mm512_reduce_add_ps, _mm512_reduce_max_ps, _mm512_roundscale_ps,
    _mm512_set1_epi32, _mm512_set1_ps, _mm512_setzero_epi32, _mm512_setzero_ps, _mm512_storeu_ps,
    _mm512_sub_ps, _CMP_GT_OS, _CMP_LT_OS, _MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT,
    _MM_FROUND_TO_NEG_INF,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::f16;
use std::ffi::{c_char, c_void, CString};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

type BrgemmFn = unsafe extern "C" fn(
    i64,
    i64,
    i64,
    i64,
    i64,
    i64,
    bool,
    *const f16,
    *const f16,
    *mut f32,
    bool,
);

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

const RTLD_NOW: i32 = 2;
const BRGEMM_SYMBOL: &str = "_ZN2at6native7cpublas6brgemmEllllllbPKN3c104HalfES5_Pfb";
const DEFAULT_LIBTORCH_CPU: &str =
    "/usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cpu.so";

static BRGEMM: OnceLock<Option<BrgemmFn>> = OnceLock::new();

fn python_libtorch_candidates(root: &Path, candidates: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    let mut python_dirs = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("python"))
        })
        .collect::<Vec<_>>();
    python_dirs.sort();

    for python_dir in python_dirs.into_iter().rev() {
        for packages_dir in ["site-packages", "dist-packages"] {
            let candidate = python_dir
                .join(packages_dir)
                .join("torch/lib/libtorch_cpu.so");
            if candidate.is_file() {
                candidates.push(candidate);
            }
        }
    }
}

fn libtorch_candidates() -> Vec<PathBuf> {
    if let Some(path) = std::env::var_os("ELLM_LIBTORCH_CPU_PATH") {
        return vec![PathBuf::from(path)];
    }

    let mut candidates = vec![PathBuf::from(DEFAULT_LIBTORCH_CPU)];
    if let Some(home_dir) = std::env::var_os("HOME") {
        python_libtorch_candidates(&PathBuf::from(home_dir).join(".local/lib"), &mut candidates);
    }
    for root in ["/usr/local/lib", "/usr/lib", "/opt"] {
        python_libtorch_candidates(Path::new(root), &mut candidates);
    }
    candidates.push(PathBuf::from("libtorch_cpu.so"));
    candidates.dedup();
    candidates
}

fn load_brgemm() -> Option<BrgemmFn> {
    *BRGEMM.get_or_init(|| unsafe {
        let symbol = CString::new(BRGEMM_SYMBOL).unwrap();
        for path in libtorch_candidates() {
            let Ok(path) = CString::new(path.as_os_str().as_encoded_bytes()) else {
                continue;
            };
            let handle = dlopen(path.as_ptr(), RTLD_NOW);
            if handle.is_null() {
                continue;
            }
            let address = dlsym(handle, symbol.as_ptr());
            if !address.is_null() {
                return Some(std::mem::transmute::<*mut c_void, BrgemmFn>(address));
            }
        }
        None
    })
}

pub fn available() -> bool {
    let isa_limited = ["ONEDNN_MAX_CPU_ISA", "DNNL_MAX_CPU_ISA"]
        .iter()
        .any(|name| std::env::var_os(name).is_some());
    !isa_limited && std::is_x86_feature_detected!("amx-fp16") && load_brgemm().is_some()
}

#[derive(Default)]
struct Workspace {
    scores: Vec<f32>,
    probabilities: Vec<f16>,
    output_acc: Vec<f32>,
    running_max: Vec<f32>,
    running_denom: Vec<f32>,
}

thread_local! {
    static WORKSPACE: RefCell<Workspace> = RefCell::new(Workspace::default());
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PackedKvKey {
    k_ptr: usize,
    v_ptr: usize,
    k_stride: usize,
    v_stride: usize,
    total_cols: usize,
    head_size: usize,
    fingerprint: [u16; 8],
}

struct PackedKv {
    k: Vec<f16>,
    v: Vec<f16>,
    total_cols: usize,
}

type PackedKvSlot = OnceLock<Arc<PackedKv>>;

/// Shared by all cloned worker-side `Attention` operators.  A GQA group has
/// eight Q heads but only one K/V head, so this avoids packing the same K/V
/// span independently on eight worker threads.
#[derive(Default)]
pub struct SharedCache {
    slots: Mutex<HashMap<PackedKvKey, Arc<PackedKvSlot>>>,
}

#[inline(always)]
unsafe fn fast_exp_f32(values: __m512) -> __m512 {
    let c0 = _mm512_set1_ps(0.00010703434948458272);
    let c1 = _mm512_set1_ps(0.30354260500649682);
    let c2 = _mm512_set1_ps(-0.22433836478672356);
    let c3 = _mm512_set1_ps(-0.07920424021977324);
    let log2ef = _mm512_castsi512_ps(_mm512_set1_epi32(0x3fb8aa3b));
    let a = _mm512_set1_ps(2f32.powi(23));
    let b = _mm512_set1_ps(2f32.powi(23) * 127.0);
    let min = _mm512_castsi512_ps(_mm512_set1_epi32(0xc2aeac50u32 as i32));
    let max = _mm512_castsi512_ps(_mm512_set1_epi32(0x42b17218));
    let min_mask = _mm512_cmp_ps_mask(values, min, _CMP_LT_OS);
    let max_mask = _mm512_cmp_ps_mask(values, max, _CMP_GT_OS);
    let mut src = _mm512_mul_ps(values, log2ef);
    let fractional = _mm512_sub_ps(
        src,
        _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC }>(src),
    );
    let mut result = _mm512_fmadd_ps(fractional, c3, c2);
    result = _mm512_fmadd_ps(fractional, result, c1);
    result = _mm512_fmadd_ps(fractional, result, c0);
    src = _mm512_sub_ps(src, result);
    let bits = _mm512_cvttps_epi32(_mm512_fmadd_ps(a, src, b));
    let bits = _mm512_mask_mov_epi32(bits, min_mask, _mm512_setzero_epi32());
    let bits = _mm512_mask_mov_epi32(bits, max_mask, _mm512_set1_epi32(0x7f800000));
    _mm512_castsi512_ps(bits)
}

#[inline(always)]
unsafe fn softmax_row(
    scores: &mut [f32],
    probabilities: &mut [f16],
    acc: &mut [f32],
    valid_cols: usize,
    scale: f32,
    old_max: f32,
    old_denom: f32,
) -> (f32, f32) {
    let simd_end = valid_cols / 16 * 16;
    let scale_vec = _mm512_set1_ps(scale);
    let mut max_vec = _mm512_set1_ps(f32::NEG_INFINITY);
    for index in (0..simd_end).step_by(16) {
        let scaled = _mm512_mul_ps(_mm512_loadu_ps(scores.as_ptr().add(index)), scale_vec);
        _mm512_storeu_ps(scores.as_mut_ptr().add(index), scaled);
        max_vec = _mm512_max_ps(max_vec, scaled);
    }
    let mut block_max = if simd_end == 0 {
        f32::NEG_INFINITY
    } else {
        _mm512_reduce_max_ps(max_vec)
    };
    for score in &mut scores[simd_end..valid_cols] {
        *score *= scale;
        block_max = block_max.max(*score);
    }

    let next_max = old_max.max(block_max);
    let carry = (old_max - next_max).exp();
    let carry_vec = _mm512_set1_ps(carry);
    let acc_simd_end = acc.len() / 16 * 16;
    for index in (0..acc_simd_end).step_by(16) {
        let value = _mm512_mul_ps(_mm512_loadu_ps(acc.as_ptr().add(index)), carry_vec);
        _mm512_storeu_ps(acc.as_mut_ptr().add(index), value);
    }
    for value in &mut acc[acc_simd_end..] {
        *value *= carry;
    }

    let max_vec = _mm512_set1_ps(next_max);
    let mut sum_vec = _mm512_setzero_ps();
    for index in (0..simd_end).step_by(16) {
        let values = fast_exp_f32(_mm512_sub_ps(
            _mm512_loadu_ps(scores.as_ptr().add(index)),
            max_vec,
        ));
        sum_vec = _mm512_add_ps(sum_vec, values);
        let half = _mm512_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(values);
        _mm256_storeu_si256(probabilities.as_mut_ptr().add(index) as *mut _, half);
    }
    let mut denom = old_denom * carry + _mm512_reduce_add_ps(sum_vec);
    for index in simd_end..valid_cols {
        let probability = (scores[index] - next_max).exp();
        denom += probability;
        probabilities[index] = probability as f16;
    }
    probabilities[valid_cols..].fill(0.0);
    (next_max, denom)
}

#[inline(always)]
unsafe fn pack_keys(
    dst: *mut f16,
    src: *const f16,
    rows: usize,
    packed_rows: usize,
    cols: usize,
    src_stride: usize,
) {
    for k in (0..cols).step_by(2) {
        for row in 0..rows {
            let dst_offset = (k / 2) * packed_rows * 2 + row * 2;
            let src_offset = row * src_stride + k;
            *dst.add(dst_offset) = *src.add(src_offset);
            *dst.add(dst_offset + 1) = *src.add(src_offset + 1);
        }
    }
}

#[inline(always)]
unsafe fn pack_values(
    dst: *mut f16,
    src: *const f16,
    rows: usize,
    _packed_rows: usize,
    cols: usize,
    src_stride: usize,
) {
    for row in (0..rows).step_by(2) {
        for col in 0..cols {
            let dst_offset = (row / 2) * cols * 2 + col * 2;
            *dst.add(dst_offset) = *src.add(row * src_stride + col);
            if row + 1 < rows {
                *dst.add(dst_offset + 1) = *src.add((row + 1) * src_stride + col);
            }
        }
    }
}

impl SharedCache {
    #[inline]
    unsafe fn fingerprint(
        k: *const f16,
        v: *const f16,
        total_cols: usize,
        k_stride: usize,
        v_stride: usize,
        head_size: usize,
    ) -> [u16; 8] {
        let last_row = total_cols.saturating_sub(1);
        let last_dim = head_size.saturating_sub(1);
        [
            (*k).to_bits(),
            (*k.add(last_dim)).to_bits(),
            (*k.add(last_row * k_stride)).to_bits(),
            (*k.add(last_row * k_stride + last_dim)).to_bits(),
            (*v).to_bits(),
            (*v.add(last_dim)).to_bits(),
            (*v.add(last_row * v_stride)).to_bits(),
            (*v.add(last_row * v_stride + last_dim)).to_bits(),
        ]
    }

    #[inline]
    unsafe fn get_or_pack(
        &self,
        k: *const f16,
        v: *const f16,
        total_cols: usize,
        k_stride: usize,
        v_stride: usize,
        head_size: usize,
    ) -> Arc<PackedKv> {
        let packed_total_cols = total_cols.next_multiple_of(2);
        let key = PackedKvKey {
            k_ptr: k as usize,
            v_ptr: v as usize,
            k_stride,
            v_stride,
            total_cols,
            head_size,
            fingerprint: Self::fingerprint(k, v, total_cols, k_stride, v_stride, head_size),
        };
        let slot = {
            let mut slots = self.slots.lock().unwrap_or_else(|error| error.into_inner());
            if !slots.contains_key(&key) && slots.len() >= 8 {
                slots.clear();
            }
            slots
                .entry(key)
                .or_insert_with(|| Arc::new(PackedKvSlot::new()))
                .clone()
        };
        slot.get_or_init(|| {
            let packed_len = packed_total_cols * head_size;
            let mut packed_k = vec![0.0; packed_len];
            let mut packed_v = vec![0.0; packed_len];
            unsafe {
                pack_keys(
                    packed_k.as_mut_ptr(),
                    k,
                    total_cols,
                    packed_total_cols,
                    head_size,
                    k_stride,
                );
                pack_values(
                    packed_v.as_mut_ptr(),
                    v,
                    total_cols,
                    packed_total_cols,
                    head_size,
                    v_stride,
                );
            }
            Arc::new(PackedKv {
                k: packed_k,
                v: packed_v,
                total_cols: packed_total_cols,
            })
        })
        .clone()
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn block_attention(
    shared_cache: &SharedCache,
    q: *const f16,
    k: *const f16,
    v: *const f16,
    output: *mut f16,
    row_begin: usize,
    row_end: usize,
    col_begin: usize,
    col_end: usize,
    total_col_end: usize,
    sequence_index: usize,
    q_stride: usize,
    k_stride: usize,
    v_stride: usize,
    head_size: usize,
    scale: f32,
) -> bool {
    let Some(brgemm) = load_brgemm() else {
        return false;
    };
    let rows = row_end - row_begin;
    let cols = col_end - col_begin;
    let gemm_cols = cols.next_multiple_of(2);
    if rows == 0 || cols == 0 || head_size % 2 != 0 {
        return false;
    }

    WORKSPACE.with(|cell| {
        let mut workspace = cell.borrow_mut();
        workspace.scores.resize(rows * gemm_cols, 0.0);
        workspace.probabilities.resize(rows * gemm_cols, 0.0);
        workspace.output_acc.resize(rows * head_size, 0.0);
        workspace.running_max.resize(rows, f32::NEG_INFINITY);
        workspace.running_denom.resize(rows, 0.0);
        let packed = shared_cache.get_or_pack(k, v, total_col_end, k_stride, v_stride, head_size);

        if col_begin == 0 {
            workspace.output_acc.fill(0.0);
            workspace.running_max.fill(f32::NEG_INFINITY);
            workspace.running_denom.fill(0.0);
        }
        let packed_k_block = packed.k.as_ptr().add(col_begin * 2);
        brgemm(
            rows as i64,
            gemm_cols as i64,
            head_size as i64,
            q_stride as i64,
            packed.total_cols as i64,
            gemm_cols as i64,
            false,
            q.add(row_begin * q_stride),
            packed_k_block,
            workspace.scores.as_mut_ptr(),
            true,
        );

        let Workspace {
            scores,
            probabilities,
            output_acc,
            running_max,
            running_denom,
        } = &mut *workspace;

        for row_offset in 0..rows {
            let row = row_begin + row_offset;
            let visible = (sequence_index + row + 1).min(total_col_end);
            let valid_cols = visible.saturating_sub(col_begin).min(cols);
            let score_row = &mut scores[row_offset * gemm_cols..(row_offset + 1) * gemm_cols];
            let acc = &mut output_acc[row_offset * head_size..(row_offset + 1) * head_size];
            let probability_row =
                &mut probabilities[row_offset * gemm_cols..(row_offset + 1) * gemm_cols];
            let (next_max, denom) = softmax_row(
                score_row,
                probability_row,
                acc,
                valid_cols,
                scale,
                running_max[row_offset],
                running_denom[row_offset],
            );
            running_max[row_offset] = next_max;
            running_denom[row_offset] = denom;
        }

        let packed_v_block = packed.v.as_ptr().add(col_begin * head_size);
        brgemm(
            rows as i64,
            head_size as i64,
            gemm_cols as i64,
            gemm_cols as i64,
            head_size as i64,
            head_size as i64,
            true,
            probabilities.as_ptr(),
            packed_v_block,
            output_acc.as_mut_ptr(),
            true,
        );

        for row_offset in 0..rows {
            let row = row_begin + row_offset;
            let visible = (sequence_index + row + 1).min(total_col_end);
            if col_end < visible {
                continue;
            }
            let inverse_denom = running_denom[row_offset].recip();
            let output_row = output.add(row * q_stride);
            let acc = &output_acc[row_offset * head_size..(row_offset + 1) * head_size];
            for index in 0..head_size {
                *output_row.add(index) = (acc[index] * inverse_denom) as f16;
            }
        }
    });
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocked_attention_matches_reference_with_causal_tail() {
        if !available() {
            return;
        }
        let rows = 5;
        let total_cols = 35;
        let head_size = 128;
        let q_stride = 256;
        let kv_stride = 192;
        let sequence_index = total_cols - rows;
        let scale = 1.0 / (head_size as f32).sqrt();
        let mut q = vec![0.0f16; rows * q_stride];
        let mut k = vec![0.0f16; total_cols * kv_stride];
        let mut v = vec![0.0f16; total_cols * kv_stride];
        let mut output = vec![0.0f16; rows * q_stride];
        for row in 0..rows {
            for d in 0..head_size {
                q[row * q_stride + d] = (((row * 17 + d * 7) % 29) as f32 * 0.01 - 0.14) as f16;
            }
        }
        for col in 0..total_cols {
            for d in 0..head_size {
                k[col * kv_stride + d] = (((col * 13 + d * 3) % 31) as f32 * 0.01 - 0.15) as f16;
                v[col * kv_stride + d] = (((col * 11 + d * 5) % 37) as f32 * 0.01 - 0.18) as f16;
            }
        }

        for col_begin in (0..total_cols).step_by(16) {
            unsafe {
                assert!(block_attention(
                    &SharedCache::default(),
                    q.as_ptr(),
                    k.as_ptr(),
                    v.as_ptr(),
                    output.as_mut_ptr(),
                    0,
                    rows,
                    col_begin,
                    (col_begin + 16).min(total_cols),
                    total_cols,
                    sequence_index,
                    q_stride,
                    kv_stride,
                    kv_stride,
                    head_size,
                    scale,
                ));
            }
        }

        for row in 0..rows {
            let visible = sequence_index + row + 1;
            let mut scores = vec![0.0f32; visible];
            for col in 0..visible {
                scores[col] = (0..head_size)
                    .map(|d| (q[row * q_stride + d] as f32) * (k[col * kv_stride + d] as f32))
                    .sum::<f32>()
                    * scale;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denom: f32 = scores.iter().map(|score| (*score - max).exp()).sum();
            for d in 0..head_size {
                let expected = (0..visible)
                    .map(|col| (scores[col] - max).exp() * (v[col * kv_stride + d] as f32))
                    .sum::<f32>()
                    / denom;
                let actual = output[row * q_stride + d] as f32;
                assert!(
                    (actual - expected).abs() < 8e-4,
                    "row={row} d={d} actual={actual} expected={expected}"
                );
            }
        }
    }
}
