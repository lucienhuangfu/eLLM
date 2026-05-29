use super::*;
use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::operators::expert::expert_routing::routing_from_dense;
use crate::tensor::GlobalOperatorQueue;
use approx::{assert_abs_diff_eq, assert_ulps_eq};
use std::collections::HashMap;
use std::f16;
use std::mem;

// ============================================================
// helpers
// ============================================================

pub fn avail_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

pub fn init_f16_tensor_test_runtime() {
    f16::init_global(HashMap::new());
    f16::init_operator_queue();
}

#[inline]
pub fn f32_from_f16(x: f16) -> f32 {
    // bitcast based f16->f32 (与你原实现一致)
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

pub fn run_operator_all_threads(
    op: &Operator<f16>,
    prefill_size: usize,
    decode_size: usize,
    cpu_num: usize,
) {
    for tid in 0..cpu_num {
        op.run(
            prefill_size,
            decode_size,
            cpu_num,
            tid,
            &[],
            &[],
            &mut Vec::new(),
        );
    }
}

pub fn take_single_f16_operator<F>(matches_expected: F) -> Operator<f16>
where
    F: FnOnce(&Operator<f16>) -> bool,
{
    let queue = f16::take_operator_queue();
    assert_eq!(queue.len(), 1);
    assert!(matches_expected(&queue[0]));
    queue.into_iter().next().unwrap()
}

pub fn run_f16_queue(prefill_size: usize, decode_size: usize, cpu_num: usize) {
    let queue = f16::take_operator_queue();
    assert_eq!(queue.len(), 1);
    for op in queue.iter() {
        run_operator_all_threads(op, prefill_size, decode_size, cpu_num);
    }
}

pub fn rope_identity(head_dim: usize) -> Vec<f16> {
    let mut rope = vec![0.0f16; head_dim];
    for i in (0..head_dim).step_by(2) {
        rope[i] = 1.0f16;
    }
    rope
}

pub fn rms_norm_f32_in_place(row: &mut [f32]) {
    let sum_sq: f32 = row.iter().map(|v| v * v).sum();
    let rrms = 1.0f32 / (sum_sq / row.len() as f32 + 1e-6).sqrt();
    for v in row {
        *v *= rrms;
    }
}

pub fn apply_qk_post_process_ref_f16(
    data: &mut [f32],
    rows: usize,
    cols: usize,
    head_dim: usize,
    rope: &[f16],
) {
    for row in 0..rows {
        for head_base in (0..cols).step_by(head_dim) {
            let head = &mut data[row * cols + head_base..row * cols + head_base + head_dim];
            rms_norm_f32_in_place(head);
            for i in (0..head_dim).step_by(2) {
                let a = head[i];
                let b = head[i + 1];
                let c = rope[i] as f32;
                let d = rope[i + 1] as f32;
                head[i] = a * c - b * d;
                head[i + 1] = a * d + b * c;
            }
        }
    }
}

#[inline]
pub fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// NT layout helpers:
// B_nt is N×K row-major => b_nt[j*K + kk]
#[inline]
pub fn idx_b_nt(j: usize, kk: usize, k: usize) -> usize {
    j * k + kk
}

// ============================================================
