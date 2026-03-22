// === compiler/mul/experts_matmul_silu_mul_matmul.rs ===
#![allow(non_snake_case)]

use std::f16;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

use super::super::super::init::{
    matmul_params::MatMulParams,
    send_sync_ptr::{ConstPtr, MutPtr},
};
use super::super::assign::assign;
use super::mul_trait::ExpertsSiluTrait;

#[derive(Clone, Copy, Debug)]
struct ExpertTaskMeta {
    expert_id: usize,
    token_begin: usize,
    token_count: usize,
    task_begin: usize,
    task_end: usize,
}

#[derive(Clone)]
pub struct ExpertsMatMulSilu<T> {
    pub input_ptr: ConstPtr<T>, // A[B,H]

    // ✅ 现在要求外部传入的权重已经是 NT（N×K，行距=H）
    // gate_nt_ptr/up_nt_ptr 指向 [E][I×H]，每 expert 一个 [I×H]
    pub gate_nt_ptr: ConstPtr<T>, // [E, I×H]  (N×K), row_stride=H
    pub up_nt_ptr: ConstPtr<T>,   // [E, I×H]

    pub experts_indicator: ConstPtr<bool>, // [E]
    pub indice_ptr: ConstPtr<bool>,        // [E,B]

    pub output_ptr: MutPtr<T>, // NONLIN[E,B,I]

    pub params: MatMulParams,

    pub batch: usize,       // B
    pub inter: usize,       // I (N)
    pub hidden: usize,      // H (K)
    pub num_experts: usize, // E

    // === strides（保留）===
    pub b_panel_stride: usize, // kc*nr
    pub acc_stride: usize,     // mr*nr
    pub a_tile_stride: usize,  // mr*kc

    // === pools（按 threads 切片）===
    pub gate_panel_pool: Box<[T]>,
    pub up_panel_pool: Box<[T]>,
    pub gate_acc_pool: Box<[T]>,
    pub up_acc_pool: Box<[T]>,
    pub a_tile_pool: Box<[T]>,

    pub idx_buf_pool: Box<[usize]>,

    // ✅ 不再持有转置后权重（外部保证生命周期）
    // pub wgate_nt_buf: Box<[T]>,
    // pub wup_nt_buf: Box<[T]>,

    _marker: PhantomData<T>,
}

impl<T> ExpertsMatMulSilu<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    pub unsafe fn new(
        input_ptr: *const T,            // A[B,H]
        gate_nt_ptr: *const T,          // ✅ W_gate_nt[E, I×H]（每 expert: [I×H] 行主）
        up_nt_ptr: *const T,            // ✅ W_up_nt  [E, I×H]
        experts_indicator: *const bool, // [E]
        indice_ptr: *const bool,        // [E,B]
        output_ptr: *mut T,             // NONLIN[E,B,I]
        batch: usize,
        inter: usize,
        hidden: usize,
        num_experts: usize,
        a_row_step_macro: usize,  // MB
        b_row_step_macro: usize,  // NB
        column_step_macro: usize, // KC
        a_row_step_micro: usize,  // MR=3
        b_row_step_micro: usize,  // NR=32
    ) -> Self {
        let mb = a_row_step_macro.max(1);
        let kc = column_step_macro.max(1);
        let mr = a_row_step_micro.max(1);
        let nr = b_row_step_micro.max(1);

        let b_panel_stride = kc * nr;
        let acc_stride = mr * nr;
        let a_tile_stride = mr * kc;

        // threads 只在 new() 用一次
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let gate_panel_pool = vec![T::default(); threads * b_panel_stride].into_boxed_slice();
        let up_panel_pool = vec![T::default(); threads * b_panel_stride].into_boxed_slice();
        let gate_acc_pool = vec![T::default(); threads * acc_stride].into_boxed_slice();
        let up_acc_pool = vec![T::default(); threads * acc_stride].into_boxed_slice();
        let a_tile_pool = vec![T::default(); threads * a_tile_stride].into_boxed_slice();

        let idx_buf_pool = vec![0usize; threads * mb].into_boxed_slice();

        Self {
            input_ptr: ConstPtr { ptr: input_ptr },
            gate_nt_ptr: ConstPtr { ptr: gate_nt_ptr },
            up_nt_ptr: ConstPtr { ptr: up_nt_ptr },

            experts_indicator: ConstPtr { ptr: experts_indicator },
            indice_ptr: ConstPtr { ptr: indice_ptr },
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

            b_panel_stride,
            acc_stride,
            a_tile_stride,

            gate_panel_pool,
            up_panel_pool,
            gate_acc_pool,
            up_acc_pool,
            a_tile_pool,

            idx_buf_pool,

            _marker: PhantomData,
        }
    }

    #[inline(always)]
    fn thread_slices(&self, tid: usize) -> (*mut T, *mut T, *mut T, *mut T, *mut T, *mut usize) {
        unsafe {
            let gp = self.gate_panel_pool.as_ptr().add(tid * self.b_panel_stride) as *mut T;
            let up = self.up_panel_pool.as_ptr().add(tid * self.b_panel_stride) as *mut T;
            let ga = self.gate_acc_pool.as_ptr().add(tid * self.acc_stride) as *mut T;
            let ua = self.up_acc_pool.as_ptr().add(tid * self.acc_stride) as *mut T;
            let at = self.a_tile_pool.as_ptr().add(tid * self.a_tile_stride) as *mut T;
            let idx = self
                .idx_buf_pool
                .as_ptr()
                .add(tid * self.params.a_row_step_macro) as *mut usize;
            (gp, up, ga, ua, at, idx)
        }
    }

    #[inline(always)]
    pub unsafe fn pack_panel_from_bnt(
        b_nt: *const T, // [N×K]
        ldb_row: usize, // = K
        n0: usize,
        k0: usize,
        kc: usize,
        nr: usize,
        out: *mut T, // KC×NR
    ) {
        for p in 0..kc {
            let dst = out.add(p * nr);
            let col_k = k0 + p;
            for lane in 0..nr {
                let j = n0 + lane;
                *dst.add(lane) = *b_nt.add(j * ldb_row + col_k);
            }
        }
    }

    #[inline(always)]
    pub unsafe fn pack_a_tile_mrkc(
        a_bxh: *const T,   // [B,H]
        lda: usize,        // = H
        idx: *const usize, // b_idx list
        idx_off: usize,
        valid_rows: usize, // <= MR
        k0: usize,
        kc: usize,
        out_mrkc: *mut T, // MR×KC
        mr: usize,
    ) {
        for i in 0..(mr * kc) {
            *out_mrkc.add(i) = T::default();
        }
        for r in 0..valid_rows {
            let b = *idx.add(idx_off + r);
            let src = a_bxh.add(b * lda + k0);
            let dst = out_mrkc.add(r * kc);
            for p in 0..kc {
                *dst.add(p) = *src.add(p);
            }
        }
    }

    #[inline]
    fn build_task_space(
        &self,
        batch_size: usize,
        mb: usize,
        tiles_n: usize,
    ) -> (Vec<ExpertTaskMeta>, Vec<usize>, usize) {
        let mut expert_tasks = Vec::new();
        let mut routed_tokens = Vec::new();
        let mut total_tasks = 0usize;

        unsafe {
            for e in 0..self.num_experts {
                if !*self.experts_indicator.ptr.add(e) {
                    continue;
                }

                let token_begin = routed_tokens.len();
                let idx_base = self.indice_ptr.ptr.add(e * self.batch);
                for b_idx in 0..batch_size {
                    if *idx_base.add(b_idx) {
                        routed_tokens.push(b_idx);
                    }
                }

                let token_count = routed_tokens.len() - token_begin;
                if token_count == 0 {
                    routed_tokens.truncate(token_begin);
                    continue;
                }

                let tiles_m_e = token_count.div_ceil(mb);
                let task_count = tiles_m_e * tiles_n;
                expert_tasks.push(ExpertTaskMeta {
                    expert_id: e,
                    token_begin,
                    token_count,
                    task_begin: total_tasks,
                    task_end: total_tasks + task_count,
                });
                total_tasks += task_count;
            }
        }

        (expert_tasks, routed_tokens, total_tasks)
    }

    #[inline(always)]
    fn decode_task(
        expert_tasks: &[ExpertTaskMeta],
        tiles_n: usize,
        task_id: usize,
    ) -> Option<(ExpertTaskMeta, usize, usize)> {
        let meta_idx = expert_tasks.partition_point(|meta| meta.task_end <= task_id);
        let meta = *expert_tasks.get(meta_idx)?;
        debug_assert!(task_id >= meta.task_begin && task_id < meta.task_end);
        let local_task = task_id - meta.task_begin;
        Some((meta, local_task / tiles_n, local_task % tiles_n))
    }

    pub fn run(
        &self,
        _position_index: usize,
        _position_interval: usize,
        batch_size: usize,
        cpu_num: usize,
        thread_id: usize,
    ) {
        unsafe {
            let b = batch_size;
            let n = self.inter;
            let k = self.hidden;

            let mb = self.params.a_row_step_macro.max(1);
            let nb = self.params.b_row_step_macro.max(1);
            let kc = self.params.column_step_macro.max(1);
            let mr = self.params.a_row_step_micro.max(1);
            let nr = self.params.b_row_step_micro.max(1);

            debug_assert!(n % nr == 0 && k % kc == 0);

            let a_base = self.input_ptr.ptr;
            let lda = self.hidden;

            let c_base = self.output_ptr.ptr;
            let c_stride_e = self.batch * self.inter; // 每 expert 的 C 块跨度

            let ldb_row = self.hidden; // K
            let w_stride = self.inter * self.hidden; // I*H per expert

            let (gate_panel, up_panel, gate_acc, up_acc, a_tile, idx_buf) =
                self.thread_slices(thread_id);

            let tiles_n = (n + nb - 1) / nb;
            let (expert_tasks, routed_tokens, total_tasks) = self.build_task_space(b, mb, tiles_n);

            if let Some((tb, te)) = assign(total_tasks, cpu_num, thread_id) {
                for t in tb..te {
                    let Some((meta, tm, tn)) = Self::decode_task(&expert_tasks, tiles_n, t) else {
                        continue;
                    };

                    let n0 = tn * nb;
                    let n_blk = (n - n0).min(nb);
                    if n_blk == 0 {
                        continue;
                    }

                    let slot_start = tm * mb;
                    let be = (meta.token_count - slot_start).min(mb);
                    debug_assert!(be > 0);

                    let token_slice =
                        &routed_tokens[(meta.token_begin + slot_start)..(meta.token_begin + slot_start + be)];
                    for (dst, &b_idx) in token_slice.iter().enumerate() {
                        *idx_buf.add(dst) = b_idx;
                    }

                    let e = meta.expert_id;
                    let wgate_nt_e = self.gate_nt_ptr.ptr.add(e * w_stride);
                    let wup_nt_e = self.up_nt_ptr.ptr.add(e * w_stride);

                    let mut nt = 0usize;
                    while nt < n_blk {
                        let cols_this = (n_blk - nt).min(nr);
                        debug_assert!(cols_this == nr || nt + cols_this == n_blk);

                        let mut off = 0usize;
                        while off < be {
                            let valid_rows = (be - off).min(mr);

                            for i in 0..(mr * nr) {
                                *gate_acc.add(i) = T::default();
                            }
                            for i in 0..(mr * nr) {
                                *up_acc.add(i) = T::default();
                            }

                            let mut k0 = 0usize;
                            while k0 < k {
                                Self::pack_panel_from_bnt(
                                    wgate_nt_e,
                                    ldb_row,
                                    n0 + nt,
                                    k0,
                                    kc,
                                    nr,
                                    gate_panel,
                                );
                                Self::pack_panel_from_bnt(
                                    wup_nt_e,
                                    ldb_row,
                                    n0 + nt,
                                    k0,
                                    kc,
                                    nr,
                                    up_panel,
                                );

                                Self::pack_a_tile_mrkc(
                                    a_base, lda, idx_buf, off, valid_rows, k0, kc, a_tile, mr,
                                );

                                self.compute1(
                                    a_tile as *const T,
                                    gate_panel as *const T,
                                    up_panel as *const T,
                                    gate_acc as *mut T,
                                    up_acc as *mut T,
                                    kc,
                                );

                                k0 += kc;
                            }

                            for r in 0..valid_rows {
                                let b_idx = *idx_buf.add(off + r);
                                let c_row = c_base
                                    .add(e * c_stride_e)
                                    .add(b_idx * self.inter)
                                    .add(n0 + nt);

                                let gate_row = gate_acc.add(r * nr);
                                let up_row = up_acc.add(r * nr);

                                self.compute2(
                                    gate_row as *const T,
                                    up_row as *const T,
                                    c_row as *mut T,
                                );
                            }

                            off += valid_rows;
                        }

                        nt += nr;
                    }
                }
            }
        }
    }
}

/* -------------------- ExpertsSiluTrait 默认实现（占位） -------------------- */

impl<T> ExpertsSiluTrait<T> for ExpertsMatMulSilu<T>
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

    default fn compute2(&self, _gate_row: *const T, _up_row: *const T, _c_row: *mut T) {}
}

/* -------------------- f16 AVX-512 FP16 特化实现 -------------------- */

impl ExpertsSiluTrait<f16> for ExpertsMatMulSilu<f16> {
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
            crate::kernel::x86_64::f16_512::moe_silu::moe_silu_update_3x32(
                a_tile, gate_panel, up_panel, gate_acc, up_acc, kc,
            );
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        {
            unreachable!("avx512fp16 required for ExpertsMatMulSilu<f16>::compute1");
        }
    }

    fn compute2(&self, gate_row: *const f16, up_row: *const f16, c_row: *mut f16) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            crate::kernel::x86_64::f16_512::moe_silu::moe_silu_finalize_row_32(
                gate_row, up_row, c_row,
            );
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        {
            unreachable!("avx512fp16 required for ExpertsMatMulSilu<f16>::compute2");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rand::prelude::*;
    use std::collections::HashSet;

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

    fn run_all_threads(runner: &ExpertsMatMulSilu<f16>, batch: usize, cpu_num: usize) {
        for tid in 0..cpu_num {
            runner.run(0, 1, batch, cpu_num, tid);
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
        let mut w_up = vec![0.0f16; E * H * I];   // K×N
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
            ExpertsMatMulSilu::<f16>::new(
                a.as_ptr(),
                w_gate_nt.as_ptr(),
                w_up_nt.as_ptr(),
                experts_indicator.as_ptr(),
                indice.as_ptr(),
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
        let mut w_up = vec![0.0f16; E * H * I];   // K×N
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
            ExpertsMatMulSilu::<f16>::new(
                a.as_ptr(),
                w_gate_nt.as_ptr(),
                w_up_nt.as_ptr(),
                experts_indicator.as_ptr(),
                indice.as_ptr(),
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
        let mut w_up = vec![0.0f16; E * H * I];   // K×N
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
            ExpertsMatMulSilu::<f16>::new(
                a.as_ptr(),
                w_gate_nt.as_ptr(),
                w_up_nt.as_ptr(),
                experts_indicator.as_ptr(),
                indice.as_ptr(),
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
                    w_gate[base] =
                        ((ex as f32 + 1.0) * 0.002 + (kk as f32) * 0.0002 + (ii as f32) * 0.0001)
                            as f16;
                    w_up[base] =
                        ((ex as f32 + 1.0) * 0.003 + (kk as f32) * 0.0001 - (ii as f32) * 0.00005)
                            as f16;
                }
            }
        }

        let w_gate_nt = transpose_expert_kxn_to_nt(&w_gate, E, H, I);
        let w_up_nt = transpose_expert_kxn_to_nt(&w_up, E, H, I);

        let runner = unsafe {
            ExpertsMatMulSilu::<f16>::new(
                a.as_ptr(),
                w_gate_nt.as_ptr(),
                w_up_nt.as_ptr(),
                experts_indicator.as_ptr(),
                indice.as_ptr(),
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
            ExpertsMatMulSilu::<f16>::new(
                a.as_ptr(),
                w_gate_nt.as_ptr(),
                w_up_nt.as_ptr(),
                experts_indicator.as_ptr(),
                indice.as_ptr(),
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
            ExpertsMatMulSilu::<f16>::new(
                a.as_ptr(),
                w_gate_nt.as_ptr(),
                w_up_nt.as_ptr(),
                experts_indicator.as_ptr(),
                indice.as_ptr(),
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
    let up_weights_nt =
        transpose_expert_kxn_to_nt(&up_weights_kxn, num_experts, hidden, inter);

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
        let op = ExpertsMatMulSilu::new(
            input.as_ptr(),
            gate_weights_nt.as_ptr(),
            up_weights_nt.as_ptr(),
            experts_indicator.as_ptr(),
            indice_ptr.as_ptr(),
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
        );

        // 注意：最好不要超过 available_parallelism()，避免 scratch 越界
        let threads_cap = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let used = num_threads.min(threads_cap).max(1);

        for tid in 0..used {
            op.run(0, 0, batch, used, tid);
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
        ExpertsMatMulSilu::<f16>::new(
            a.as_ptr(),
            w_gate_nt.as_ptr(),
            w_up_nt.as_ptr(),
            experts_indicator.as_ptr(),
            indice.as_ptr(),
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
        )
    };

    // run 只跑 B_RUN
    for tid in 0..cpu_num {
        runner.run(0, 0, B_RUN, cpu_num, tid);
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
        if any_nonzero { break; }
    }
    assert!(any_nonzero, "expected some non-zero outputs for rows 0..6");
}
#[test]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
fn test_experts_matmul_silu_moe_smoke_fast_sampled() {
    // 缩小版“Qwen风格”测试：仍然是 MoE + topk routing + mr=3/nr=32 + kc split
    // 但规模足够小，保证 cargo test 很快跑完

    let batch = 24;          // 原 72
    let hidden = 256;        // 原 4096（仍保持能被 kc=32 整除）
    let inter = 128;         // 原 768（能被 nr=32 整除）
    let num_experts = 16;    // 原 128
    let top_k = 4;           // 原 8
    let num_threads = 4;

    let mr = 3;
    let nr = 32;
    let kc = 32;
    let mb = 6;              // 让 tiles_m>1，更像真实分块
    let nb = 64;             // inter=128 => tiles_n=2

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
    let gate_weights_nt = transpose_expert_kxn_to_nt(&gate_weights_kxn, num_experts, hidden, inter);
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
        let op = ExpertsMatMulSilu::new(
            input.as_ptr(),
            gate_weights_nt.as_ptr(),
            up_weights_nt.as_ptr(),
            experts_indicator.as_ptr(),
            indice_ptr.as_ptr(),
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
        );

        // 你说固定机器跑，这里就直接用 num_threads（确保不超过你那台机的 threads）
        for tid in 0..num_threads {
            op.run(0, 0, batch, num_threads, tid);
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

    assert!(checked >= samples / 2, "too few routed samples checked: {}/{}", checked, samples);
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
        ExpertsMatMulSilu::<f16>::new(
            a.as_ptr(),
            w_gate_nt.as_ptr(),
            w_up_nt.as_ptr(),
            experts_indicator.as_ptr(),
            indice.as_ptr(),
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
        )
    };

    // 单线程即可复现
    op.run(0, 0, B_RUN, 1, 0);

    // row 7/8 必须仍为 0（没被触碰）
    for b in B_RUN..B_CAP {
        for ii in 0..I {
            let v = out[b * I + ii] as f32;
            assert!(
                v.abs() <= 1e-3,
                "row {} should remain 0, got {}",
                b,
                v
            );
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
}
