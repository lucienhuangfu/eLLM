// === runner/experts_matmul_silu.rs ===
#![allow(non_snake_case)]

use std::marker::PhantomData;
use std::ops::{Add, Mul};

use super::super::super::init::{
    matmul_params::MatMulParams,
    send_sync_ptr::{ConstPtr, MutPtr},
};
use super::super::assign::assign;
use super::mul_trait::MatMul3Trait; // compute1: GEMM 累加，compute2: SiLU⊙ 写回

/// MoE 第一步：
/// 对每个 expert e，把 A[B,H] 和该 expert 的 W_gate[e] / W_up[e] 做
///     gate = A·W_gate[e]
///     up   = A·W_up[e]
/// 然后输出 NONLIN[e,b,:] = SiLU(gate[b,:]) ⊙ up[b,:]
///
/// 这版按照「expert 内第 k 个命中 token 的 slot」做宏块：
/// - 对 expert e，我们有一个 bitvector indice_ptr[e,B]（to ken 是否命中该 expert）
/// - 把所有 b 上的 true 按顺序编号：slot 0,1,2,...
/// - M 方向 tile_m = 0,1,2,... 表示 slot 区间：
///     tile_m = 0 -> slot [0, MB)
///     tile_m = 1 -> slot [MB, 2MB)
///   对于每个 tile_m，我们在 expert 的 slot 序列里找到这一段的 token，
///   把对应 A[b,:] 行挤成一个致密小块来做 3×32 GEMM。
#[derive(Clone)]
pub struct ExpertsMatmulSilu<T> {
    // 左矩阵 A[B,H]
    pub input_ptr: ConstPtr<T>,

    // 右矩阵（已转置）：每 expert 的 W_gate_nt / W_up_nt 为 [N×K=I×H] 行主（行距=K）
    // 多个 expert 顺序拼接：第 e 个块偏移 = e * (N*K)
    pub gate_nt_ptr: ConstPtr<T>,
    pub up_nt_ptr:   ConstPtr<T>,

    // 路由 bitvector：
    //  - experts_indicator[e]：该 expert 在整个 batch 内是否有命中（粗粒度跳过用）
    //  - indice_ptr[e,b]：token b 是否命中 expert e（作为 expert 的 slot 序列的源）
    pub experts_indicator: ConstPtr<bool>, // [E]
    pub indice_ptr:        ConstPtr<bool>, // [E,B]

    // 输出 NONLIN[e,b,i]（行主：先 e，再 b，再 i）
    pub output_ptr: MutPtr<T>,

    /// 仅承载 step 形状（MB/NB/KC/MR/NR）
    pub params: MatMulParams,

    // 形状信息
    pub batch: usize,       // B
    pub inter: usize,       // I（中间维度）
    pub hidden: usize,      // H
    pub num_experts: usize, // E

    // 线程私有池（一次分配，run 零分配）
    // - gate_panel / up_panel：KC×NR（打 B_nt 面板用）
    // - gate_acc / up_acc：MR×NR（累加 3×32）
    // - a_tile：MR×KC（从 A[B,H] 抽出的 3 行压紧 tile）
    // - idx_buf：每个线程最多 MB 个 slot 对应的 token 行号
    pub cpu_max_for_scratch: usize,
    pub b_panel_stride: usize,
    pub acc_stride: usize,
    pub a_tile_stride: usize,

    pub gate_panel_pool: Box<[T]>,
    pub up_panel_pool:   Box<[T]>,
    pub gate_acc_pool:   Box<[T]>,
    pub up_acc_pool:     Box<[T]>,
    pub a_tile_pool:     Box<[T]>,

    pub idx_buf_pool: Box<[usize]>,

    // 构造期转置后的持久化缓冲
    pub wgate_nt_buf: Box<[T]>, // [E, N×K]
    pub wup_nt_buf:   Box<[T]>, // [E, N×K]

    _marker: PhantomData<T>,
}

impl<T> ExpertsMatmulSilu<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    /// 单 expert：把 B[K×N]（行主，行距=N）转为 B_nt[N×K]（行主，行距=K）
    #[inline]
    unsafe fn make_b_nt(src_kxn: *const T, n: usize, k: usize) -> Box<[T]> {
        let mut v = vec![T::default(); n * k];
        let dst = v.as_mut_ptr();
        for kk in 0..k {
            let src_row = src_kxn.add(kk * n); // B[kk, :]
            for jj in 0..n {
                *dst.add(jj * k + kk) = *src_row.add(jj); // B_nt[jj, kk] = B[kk, jj]
            }
        }
        v.into_boxed_slice()
    }

    /// 构造：传入 gate/up 权重为 K×N（= H×I），逐 expert 转成 N×K（= I×H），并分配线程池。
    ///
    /// Safety：所有裸指针应在整个 Runner 生命周期内有效；cpu_num 不得超过 cpu_max_for_scratch。
    pub unsafe fn new(
        input_ptr: *const T,            // A[B,H]
        gate_kxn_ptr: *const T,         // W_gate[E, K=H, N=I]
        up_kxn_ptr:   *const T,         // W_up  [E, K=H, N=I]
        experts_indicator: *const bool, // [E]
        indice_ptr: *const bool,        // [E,B]
        output_ptr: *mut T,             // NONLIN[E,B,I]
        // 形状
        batch: usize,   // B
        inter: usize,   // I
        hidden: usize,  // H
        num_experts: usize,
        // 分块参数
        a_row_step_macro: usize, // MB
        b_row_step_macro: usize, // NB
        column_step_macro: usize, // KC
        a_row_step_micro: usize, // MR=3
        b_row_step_micro: usize, // NR=32
        cpu_max_for_scratch: usize,
    ) -> Self {
        // === (1) 逐 expert 转置 W_gate / W_up（K×N → N×K） ===
        let per_elems = inter * hidden; // N×K = I×H

        let mut wgate_nt_buf = vec![T::default(); num_experts * per_elems].into_boxed_slice();
        let mut wup_nt_buf   = vec![T::default(); num_experts * per_elems].into_boxed_slice();

        for e in 0..num_experts {
            let src_gate_e = gate_kxn_ptr.add(e * (hidden * inter)); // [H×I]
            let src_up_e   = up_kxn_ptr  .add(e * (hidden * inter));

            let gate_nt = Self::make_b_nt(src_gate_e, inter, hidden); // [I×H]
            let up_nt   = Self::make_b_nt(src_up_e,   inter, hidden);

            let dst_gate = wgate_nt_buf.as_mut_ptr().add(e * per_elems);
            let dst_up   = wup_nt_buf  .as_mut_ptr().add(e * per_elems);
            std::ptr::copy_nonoverlapping(gate_nt.as_ptr(), dst_gate, per_elems);
            std::ptr::copy_nonoverlapping(up_nt  .as_ptr(), dst_up,   per_elems);
        }

        let gate_nt_ptr = wgate_nt_buf.as_ptr();
        let up_nt_ptr   = wup_nt_buf  .as_ptr();

        // === (2) 分配线程本地池（panel/acc/A_tile/idx_buf） ===
        let mb = a_row_step_macro.max(1);
        let kc = column_step_macro.max(1);
        let mr = a_row_step_micro.max(1);
        let nr = b_row_step_micro.max(1);

        let b_panel_stride = kc * nr;
        let acc_stride     = mr * nr;
        let a_tile_stride  = mr * kc;

        let gate_panel_pool = vec![T::default(); cpu_max_for_scratch * b_panel_stride].into_boxed_slice();
        let up_panel_pool   = vec![T::default(); cpu_max_for_scratch * b_panel_stride].into_boxed_slice();
        let gate_acc_pool   = vec![T::default(); cpu_max_for_scratch * acc_stride    ].into_boxed_slice();
        let up_acc_pool     = vec![T::default(); cpu_max_for_scratch * acc_stride    ].into_boxed_slice();
        let a_tile_pool     = vec![T::default(); cpu_max_for_scratch * a_tile_stride ].into_boxed_slice();

        let idx_buf_pool    = vec![0usize; cpu_max_for_scratch * mb].into_boxed_slice();

        Self {
            input_ptr: ConstPtr { ptr: input_ptr },
            gate_nt_ptr: ConstPtr { ptr: gate_nt_ptr },
            up_nt_ptr:   ConstPtr { ptr: up_nt_ptr },

            experts_indicator: ConstPtr { ptr: experts_indicator },
            indice_ptr:        ConstPtr { ptr: indice_ptr },
            output_ptr:        MutPtr   { ptr: output_ptr },

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

    /// 线程切片：返回 (gate_panel, up_panel, gate_acc, up_acc, a_tile, idx_buf)
    #[inline(always)]
    pub fn thread_slices(&self, tid: usize) -> (*mut T, *mut T, *mut T, *mut T, *mut T, *mut usize) {
        debug_assert!(tid < self.cpu_max_for_scratch);
        unsafe {
            let gp  = self.gate_panel_pool.as_ptr().add(tid * self.b_panel_stride) as *mut T;
            let up  = self.up_panel_pool  .as_ptr().add(tid * self.b_panel_stride) as *mut T;
            let ga  = self.gate_acc_pool  .as_ptr().add(tid * self.acc_stride)     as *mut T;
            let ua  = self.up_acc_pool    .as_ptr().add(tid * self.acc_stride)     as *mut T;
            let at  = self.a_tile_pool    .as_ptr().add(tid * self.a_tile_stride)  as *mut T;
            let idx = self.idx_buf_pool   .as_ptr().add(tid * self.params.a_row_step_macro) as *mut usize;
            (gp, up, ga, ua, at, idx)
        }
    }

    /// 从 B_nt[N×K]（行主；行距=K）打 kc×nr 面板（输出 KC×NR 行主）
    #[inline(always)]
    pub unsafe fn pack_panel_from_bnt(
        b_nt: *const T,  // [N×K]
        ldb_row: usize,  // = K
        n0: usize,       // N 起点
        k0: usize,       // K 起点
        kc: usize,
        nr: usize,
        out: *mut T,     // KC×NR
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

    /// 从 A[B,H] 拷 MR×KC 小 tile（不足 MR 的行补 0），配合“expert slot 序列”的行索引
    #[inline(always)]
    pub unsafe fn pack_a_tile_mrkc(
        a_bxh: *const T,   // [B,H]
        lda: usize,        // = H
        idx: *const usize, // 压紧后的 token 行号数组（存的是 b_idx）
        idx_off: usize,    // 当前批在 idx 中的起点（对应 slot_start+offset）
        valid_rows: usize, // <= MR
        k0: usize,         // K 起点
        kc: usize,
        out_mrkc: *mut T,  // MR×KC
        mr: usize,
    ) {
        // 先清零整个 MR×KC（尾部无效行）
        for i in 0..(mr * kc) {
            *out_mrkc.add(i) = T::default();
        }
        // 再填充 valid_rows 行
        for r in 0..valid_rows {
            let b = *idx.add(idx_off + r); // 原始 batch 行号
            let src = a_bxh.add(b * lda + k0);
            let dst = out_mrkc.add(r * kc);
            for p in 0..kc {
                *dst.add(p) = *src.add(p);
            }
        }
    }

    /// 主流程：
    /// - 外层仍然按 M×N tiles 切；M 方向的 tile_m 被解释为 "expert 内的 slot 范围"
    /// - 对每个 expert e：
    ///     1) 用 indice_ptr[e,B] 按顺序遍历 batch，找出 slot ∈ [slot_start, slot_end) 的 token 行号
    ///     2) 把这些行打包成致密的 A_tile（MR×KC 小块）
    ///     3) 调用 compute1 做 A·W_gate / A·W_up 的 GEMM 累加
    ///     4) 调用 compute2 做 SiLU(gate) ⊙ up 并写入 NONLIN[e,b,:]
    ///
    /// 注意：这里“没有 sequence 维度”，position_* 参数保留只是为了接口兼容，不参与计算。
    pub fn run(
        &self,
        _position_index: usize,
        _position_interval: usize,
        batch_size: usize,
        cpu_num: usize,
        thread_id: usize,
    ) {
        unsafe {
            let b = batch_size;       // 最大可能的 slot 数（expert 的命中 token 不会超过 B）
            let n = self.inter;       // N = I
            let k = self.hidden;      // K = H

            // 分块参数
            let mb = self.params.a_row_step_macro.max(1); // 每个宏核最多 MB 行（slot）
            let nb = self.params.b_row_step_macro.max(1);
            let kc = self.params.column_step_macro.max(1);
            let mr = self.params.a_row_step_micro.max(1); // 3
            let nr = self.params.b_row_step_micro.max(1); // 32

            debug_assert!(n % nr == 0 && k % kc == 0);
            debug_assert!(thread_id < self.cpu_max_for_scratch && cpu_num <= self.cpu_max_for_scratch);

            // A / C 布局
            let a_base = self.input_ptr.ptr;     // [B,H]
            let lda    = self.hidden;            // 行距 = H

            let c_base = self.output_ptr.ptr;    // [E,B,I]
            let c_stride_e = self.batch * self.inter; // 每个 expert 的块大小

            // B（权重）布局：每 expert 段大小 = N×K = I×H，行距 ldb_row = K=H
            let ldb_row = self.hidden;
            let w_stride = self.inter * self.hidden;

            // 线程缓冲
            let (gate_panel, up_panel, gate_acc, up_acc, a_tile, idx_buf) = self.thread_slices(thread_id);

            // tile 个数：
            // - M 方向沿着“slot 索引”切：0..B，以 MB 为粒度
            // - N 方向沿着中间维 I 切
            let tiles_m = (b + mb - 1) / mb;
            let tiles_n = (n + nb - 1) / nb;

            if let Some((tb, te)) = assign(tiles_m * tiles_n, cpu_num, thread_id) {
                for t in tb..te {
                    let tm = t / tiles_n;
                    let tn = t % tiles_n;

                    // slot 范围：[slot_start, slot_end)
                    let slot_start = tm * mb;
                    let slot_end   = slot_start + mb;

                    let n0 = tn * nb;
                    let n_blk = (n - n0).min(nb);
                    if n_blk == 0 {
                        continue;
                    }

                    // 对每个 expert e 单独处理该 slot 范围
                    for e in 0..self.num_experts {
                        if *self.experts_indicator.ptr.add(e) == false {
                            continue;
                        }

                        // ==== (1) 在 expert e 的命中 token 序列中，找出 slot ∈ [slot_start, slot_end) 的 token ====
                        let idx_base = self.indice_ptr.ptr.add(e * self.batch);
                        let mut be = 0usize;  // 当前宏核内有效行数（<= MB）
                        let mut seen = 0usize; // 已数到的命中 token 数（slot index）

                        for b_idx in 0..self.batch {
                            if *idx_base.add(b_idx) {
                                // 当前命中的 token slot = seen
                                if seen >= slot_start && seen < slot_end {
                                    *idx_buf.add(be) = b_idx; // 存原始 batch 行号
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
                            // 该 expert 在这个 slot 范围内没有命中 token
                            continue;
                        }

                        // N 方向：按 NR 切列段
                        let mut nt = 0usize;
                        while nt < n_blk {
                            let cols_this = (n_blk - nt).min(nr);
                            debug_assert!(cols_this == nr || nt + cols_this == n_blk);

                            // 行方向：按 MR 处理压紧后的行
                            let mut off = 0usize;
                            while off < be {
                                let valid_rows = (be - off).min(mr);

                                // 清零 gate_acc / up_acc（MR×NR）
                                for i in 0..(mr * nr) {
                                    *gate_acc.add(i) = T::default();
                                    *up_acc  .add(i) = T::default();
                                }

                                // K 方向按 kc 面板累加
                                let mut k0 = 0usize;
                                while k0 < k {
                                    // 打 gate / up 的 KC×NR 面板（来自 expert e 的 W_nt）
                                    let wgate_nt_e = self.gate_nt_ptr.ptr.add(e * w_stride);
                                    let wup_nt_e   = self.up_nt_ptr  .ptr.add(e * w_stride);

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

                                    // 从 A[B,H] 抽取 MR×KC，小 tile 紧凑存放在 a_tile
                                    Self::pack_a_tile_mrkc(
                                        a_base,
                                        lda,
                                        idx_buf,
                                        off,
                                        valid_rows,
                                        k0,
                                        kc,
                                        a_tile,
                                        mr,
                                    );

                                    // per-kc 累加：A_tile × gate_panel / up_panel → gate_acc / up_acc
                                    self.compute1(
                                        a_tile   as *const T,
                                        gate_panel as *const T,
                                        up_panel   as *const T,
                                        gate_acc  as *mut   T,
                                        up_acc    as *mut   T,
                                    );

                                    k0 += kc;
                                }

                                // finalize：对 valid_rows 行做 SiLU(gate) ⊙ up，并写回 NONLIN[e,b_idx,n 段]
                                for r in 0..valid_rows {
                                    let b_idx = *idx_buf.add(off + r);
                                    let c_row = c_base
                                        .add(e * c_stride_e)
                                        .add(b_idx * self.inter)
                                        .add(n0 + nt);
                                    let gate_row = gate_acc.add(r * nr);
                                    let up_row   = up_acc  .add(r * nr);

                                    self.compute2(
                                        gate_row as *const T,
                                        up_row   as *const T,
                                        c_row    as *mut   T,
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

/* -------------------- MatMul3Trait：默认实现留空，具体 ISA 在 kernel 里实现 -------------------- */

impl<T> MatMul3Trait<T> for ExpertsMatmulSilu<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    // compute1：每个 kc 面板，A_tile(MR×KC) × gate_panel/up_panel(KC×NR) → gate_acc/up_acc(MR×NR)
    default fn compute1(
        &self,
        _a_tile: *const T,
        _gate_panel: *const T,
        _up_panel: *const T,
        _gate_acc: *mut T,
        _up_acc: *mut T,
    ) {
        // 留空：由 generic / avx2 / avx512 等具体内核实现
    }

    // compute2：对某个 3×32 tile 的前 valid_rows 行做 SiLU(gate) ⊙ up，并写到 C
    // 这里我们在 run 里已经按“行”传入 gate_row / up_row / c_row
    default fn compute2(
        &self,
        _gate_row: *const T,
        _up_row: *const T,
        _c_row: *mut T,
    ) {
        // 留空：由具体内核实现（可以是 1×NR 的 row-wise 内核）
    }
}