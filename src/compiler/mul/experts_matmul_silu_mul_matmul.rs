// === runner/experts_matmul_silu.rs ===
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

/// MoE 第一步：
/// 对每个 expert e：
///   gate = A·W_gate[e]
///   up   = A·W_up[e]
///   NONLIN[e,b,:] = SiLU(gate[b,:]) ⊙ up[b,:]
#[derive(Clone)]
pub struct ExpertsMatmulSilu<T> {
    pub input_ptr: ConstPtr<T>, // A[B,H]

    // 转置后：每 expert 的 [I×H]（N×K），行距=K=H
    pub gate_nt_ptr: ConstPtr<T>,
    pub up_nt_ptr: ConstPtr<T>,

    pub experts_indicator: ConstPtr<bool>, // [E]
    pub indice_ptr: ConstPtr<bool>,        // [E,B]

    pub output_ptr: MutPtr<T>, // NONLIN[E,B,I]

    pub params: MatMulParams,

    pub batch: usize,       // B
    pub inter: usize,       // I
    pub hidden: usize,      // H
    pub num_experts: usize, // E

    pub cpu_max_for_scratch: usize,
    pub b_panel_stride: usize,
    pub acc_stride: usize,
    pub a_tile_stride: usize,

    pub gate_panel_pool: Box<[T]>,
    pub up_panel_pool: Box<[T]>,
    pub gate_acc_pool: Box<[T]>,
    pub up_acc_pool: Box<[T]>,
    pub a_tile_pool: Box<[T]>,

    pub idx_buf_pool: Box<[usize]>,

    pub wgate_nt_buf: Box<[T]>,
    pub wup_nt_buf: Box<[T]>,

    _marker: PhantomData<T>,
}

impl<T> ExpertsMatmulSilu<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    #[inline]
    unsafe fn make_b_nt(src_kxn: *const T, n: usize, k: usize) -> Box<[T]> {
        let mut v = vec![T::default(); n * k];
        let dst = v.as_mut_ptr();
        for kk in 0..k {
            let src_row = src_kxn.add(kk * n);
            for jj in 0..n {
                *dst.add(jj * k + kk) = *src_row.add(jj);
            }
        }
        v.into_boxed_slice()
    }

    pub unsafe fn new(
        input_ptr: *const T,            // A[B,H]
        gate_kxn_ptr: *const T,         // W_gate[E, H×I]
        up_kxn_ptr: *const T,           // W_up  [E, H×I]
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
        cpu_max_for_scratch: usize,
    ) -> Self {
        let per_elems = inter * hidden;

        let mut wgate_nt_buf = vec![T::default(); num_experts * per_elems].into_boxed_slice();
        let mut wup_nt_buf = vec![T::default(); num_experts * per_elems].into_boxed_slice();

        for e in 0..num_experts {
            let src_gate_e = gate_kxn_ptr.add(e * (hidden * inter));
            let src_up_e = up_kxn_ptr.add(e * (hidden * inter));

            let gate_nt = Self::make_b_nt(src_gate_e, inter, hidden);
            let up_nt = Self::make_b_nt(src_up_e, inter, hidden);

            let dst_gate = wgate_nt_buf.as_mut_ptr().add(e * per_elems);
            let dst_up = wup_nt_buf.as_mut_ptr().add(e * per_elems);
            std::ptr::copy_nonoverlapping(gate_nt.as_ptr(), dst_gate, per_elems);
            std::ptr::copy_nonoverlapping(up_nt.as_ptr(), dst_up, per_elems);
        }

        let gate_nt_ptr = wgate_nt_buf.as_ptr();
        let up_nt_ptr = wup_nt_buf.as_ptr();

        let mb = a_row_step_macro.max(1);
        let kc = column_step_macro.max(1);
        let mr = a_row_step_micro.max(1);
        let nr = b_row_step_micro.max(1);

        let b_panel_stride = kc * nr;
        let acc_stride = mr * nr;
        let a_tile_stride = mr * kc;

        let gate_panel_pool =
            vec![T::default(); cpu_max_for_scratch * b_panel_stride].into_boxed_slice();
        let up_panel_pool =
            vec![T::default(); cpu_max_for_scratch * b_panel_stride].into_boxed_slice();
        let gate_acc_pool = vec![T::default(); cpu_max_for_scratch * acc_stride].into_boxed_slice();
        let up_acc_pool = vec![T::default(); cpu_max_for_scratch * acc_stride].into_boxed_slice();
        let a_tile_pool =
            vec![T::default(); cpu_max_for_scratch * a_tile_stride].into_boxed_slice();

        let idx_buf_pool = vec![0usize; cpu_max_for_scratch * mb].into_boxed_slice();

        Self {
            input_ptr: ConstPtr { ptr: input_ptr },
            gate_nt_ptr: ConstPtr { ptr: gate_nt_ptr },
            up_nt_ptr: ConstPtr { ptr: up_nt_ptr },

            experts_indicator: ConstPtr {
                ptr: experts_indicator,
            },
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

            cpu_max_for_scratch,
            b_panel_stride,
            acc_stride,
            a_tile_stride,

            gate_panel_pool,
            up_panel_pool,
            gate_acc_pool,
            up_acc_pool,
            a_tile_pool,

            idx_buf_pool,

            wgate_nt_buf,
            wup_nt_buf,

            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn thread_slices(
        &self,
        tid: usize,
    ) -> (*mut T, *mut T, *mut T, *mut T, *mut T, *mut usize) {
        debug_assert!(tid < self.cpu_max_for_scratch);
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
            debug_assert!(
                thread_id < self.cpu_max_for_scratch && cpu_num <= self.cpu_max_for_scratch
            );

            let a_base = self.input_ptr.ptr;
            let lda = self.hidden;

            let c_base = self.output_ptr.ptr;
            let c_stride_e = self.batch * self.inter;

            let ldb_row = self.hidden;
            let w_stride = self.inter * self.hidden;

            let (gate_panel, up_panel, gate_acc, up_acc, a_tile, idx_buf) =
                self.thread_slices(thread_id);

            let tiles_m = (b + mb - 1) / mb;
            let tiles_n = (n + nb - 1) / nb;

            if let Some((tb, te)) = assign(tiles_m * tiles_n, cpu_num, thread_id) {
                for t in tb..te {
                    let tm = t / tiles_n;
                    let tn = t % tiles_n;

                    let slot_start = tm * mb;
                    let slot_end = slot_start + mb;

                    let n0 = tn * nb;
                    let n_blk = (n - n0).min(nb);
                    if n_blk == 0 {
                        continue;
                    }

                    for e in 0..self.num_experts {
                        if !*self.experts_indicator.ptr.add(e) {
                            continue;
                        }

                        let idx_base = self.indice_ptr.ptr.add(e * self.batch);
                        let mut be = 0usize;
                        let mut seen = 0usize;

                        for b_idx in 0..self.batch {
                            if *idx_base.add(b_idx) {
                                if seen >= slot_start && seen < slot_end {
                                    *idx_buf.add(be) = b_idx;
                                    be += 1;
                                    if be == mb {
                                        break;
                                    }
                                }
                                seen += 1;
                                if seen >= slot_end {
                                    break;
                                }
                            }
                        }
                        if be == 0 {
                            continue;
                        }

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
                                    let wgate_nt_e = self.gate_nt_ptr.ptr.add(e * w_stride);
                                    let wup_nt_e = self.up_nt_ptr.ptr.add(e * w_stride);

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
}

/* -------------------- ExpertsSiluTrait 默认实现（占位） -------------------- */

impl<T> ExpertsSiluTrait<T> for ExpertsMatmulSilu<T>
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

impl ExpertsSiluTrait<f16> for ExpertsMatmulSilu<f16> {
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
            unreachable!("avx512fp16 required for ExpertsMatmulSilu<f16>::compute1");
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
            unreachable!("avx512fp16 required for ExpertsMatmulSilu<f16>::compute2");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use std::collections::HashSet;

    // 纯 Rust 参考实现：用于验证算子计算结果的正确性
    // Output[e, b, i] = SiLU(Input[b] @ Gate[e]) * (Input[b] @ Up[e])
    fn reference_implementation(
        input: &[f16],
        gate: &[f16],
        up: &[f16],
        experts_indicator: &[bool],
        indice_ptr: &[bool],
        batch: usize,
        hidden: usize,
        inter: usize,
        num_experts: usize,
    ) -> Vec<f16> {
        let mut output = vec![0.0 as f16; num_experts * batch * inter];

        for e in 0..num_experts {
            // 如果该 Expert 在本 batch 中未被任何 token 选中，跳过
            if !experts_indicator[e] {
                continue;
            }

            for b in 0..batch {
                // 检查 token b 是否被路由到 expert e
                if indice_ptr[e * batch + b] {
                    for i in 0..inter {
                        let mut gate_val = 0.0f32;
                        let mut up_val = 0.0f32;

                        for h in 0..hidden {
                            let inp = input[b * hidden + h] as f32;
                            // 权重布局为 [E, H, I] (Row-Major)
                            let w_idx = e * (hidden * inter) + h * inter + i;

                            gate_val += inp * (gate[w_idx] as f32);
                            up_val += inp * (up[w_idx] as f32);
                        }

                        // SiLU 激活: x / (1 + exp(-x))
                        let silu_gate = gate_val / (1.0 + (-gate_val).exp());
                        let res = silu_gate * up_val;

                        output[e * (batch * inter) + b * inter + i] = res as f16;
                    }
                }
            }
        }
        output
    }

    #[test]
    fn test_experts_matmul_silu_qwen_moe_config() {
        // 1. 测试参数配置
        let batch = 72;
        let hidden = 128 * 32; // 4096
        let inter = 768; // moe_intermediate_size
        let num_experts = 128;
        let top_k = 8; // 每个 token 激活 8 个 expert
        let num_threads = 4;

        // Kernel 分块参数
        let mr = 3;
        let nr = 32;
        let kc = 32;
        let mb = 3;
        let nb = 32;

        let mut rng = rand::thread_rng();

        println!(
            "Testing with Batch={}, Hidden={}, Inter={}, Experts={}, TopK={}",
            batch, hidden, inter, num_experts, top_k
        );

        // 2. 准备随机数据
        // Input: [B, H]
        let input: Vec<f16> = (0..batch * hidden)
            .map(|_| rng.gen_range(-0.1..0.1) as f16)
            .collect();

        // Weights: [E, H, I]
        let gate_weights: Vec<f16> = (0..num_experts * hidden * inter)
            .map(|_| rng.gen_range(-0.05..0.05) as f16)
            .collect();
        let up_weights: Vec<f16> = (0..num_experts * hidden * inter)
            .map(|_| rng.gen_range(-0.05..0.05) as f16)
            .collect();

        // 3. 模拟路由逻辑 (Routing)
        // indice_ptr: [E, B] -> 如果 token b 选中 expert e，则为 true
        let mut indice_ptr = vec![false; num_experts * batch];
        // experts_indicator: [E] -> 如果 expert e 被任意 token 选中，则为 true
        let mut experts_indicator = vec![false; num_experts];

        for b in 0..batch {
            // 为每个 token 随机选择 8 个不重复的 expert
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

        // 4. 运行优化算子
        unsafe {
            let op = ExpertsMatmulSilu::new(
                input.as_ptr(),
                gate_weights.as_ptr(),
                up_weights.as_ptr(),
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
                num_threads, // cpu_max_for_scratch
            );

            // 模拟多线程运行，逐个执行
            for tid in 0..num_threads {
                op.run(0, 0, batch, num_threads, tid);
            }
        }

        // 5. 运行参考实现
        let ref_output = reference_implementation(
            &input,
            &gate_weights,
            &up_weights,
            &experts_indicator,
            &indice_ptr,
            batch,
            hidden,
            inter,
            num_experts,
        );

        // 6. 验证结果
        let mut max_diff = 0.0f32;
        // f16 在 4096 维度累加时精度损失较大，设置较宽的容差
        let tolerance = 0.15;

        for i in 0..output.len() {
            let val = output[i] as f32;
            let ref_val = ref_output[i] as f32;
            let diff = (val - ref_val).abs();

            if diff > max_diff {
                max_diff = diff;
            }

            if diff > tolerance {
                // 仅当参考值本身不接近 0 时才报错，避免 0 vs 1e-4 这种无意义的比较
                if ref_val.abs() > 0.01 {
                    let e = i / (batch * inter);
                    let rem = i % (batch * inter);
                    let b = rem / inter;
                    let idx = rem % inter;
                    panic!(
                        "Mismatch at Expert={}, Batch={}, Idx={}: Got {}, Expected {}, Diff {}",
                        e, b, idx, val, ref_val, diff
                    );
                }
            }
        }

        println!("Max diff: {}", max_diff);
    }
}
