// === runner/matmul_silu.rs ===
#![allow(non_snake_case)]

use std::f16;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

use super::super::super::init::{
    matmul_params::MatmulParams,
    send_sync_ptr::{ConstPtr, MutPtr},
};
use super::super::super::kernel;
use super::super::super::memory::allocator::allocate_init;
use super::super::assign::assign;
use super::mul_trait::SiluMatmulTrait; // ← 新 trait 名称

/// 构造期：把 B[K×N] 转置为 B_nt[N×K]（行主，行距=K）
#[inline]
unsafe fn make_b_nt<T: Copy + Default>(ptr2: *const T, n: usize, k: usize) -> Box<[T]> {
    let mut v = vec![T::default(); n * k];
    for kk in 0..k {
        let src_row = ptr2.add(kk * n); // B[kk, 0]
        for jj in 0..n {
            *v.as_mut_ptr().add(jj * k + kk) = *src_row.add(jj);
        }
    }
    v.into_boxed_slice()
}

/// Runner：A·W_gate 与 A·W_up 的 fused 计算；最终输出 SiLU(gate) ⊙ up
#[derive(Clone)]
pub struct MatMulSilu<T> {
    pub ptr1: ConstPtr<T>,     // A[M×K]
    pub ptr2: ConstPtr<T>,     // W_gate[N×K] —— 已转置后的首地址
    pub ptr3: ConstPtr<T>,     // W_up  [N×K] —— 已转置后的首地址
    pub output_ptr: MutPtr<T>, // C[M×N]
    /// params 只承载 step 形状（MB/NB/KC/MR/NR）
    pub params: MatmulParams,
    pub m_max: usize,
    pub n_max: usize,
    pub k_max: usize,
    _marker: PhantomData<T>,

    // 转置缓存（与 ptr2/ptr3 同寿命；用于持有内存）
    pub wgate_buf: Box<[T]>, // [N×K]
    pub wup_buf:   Box<[T]>, // [N×K]

    // 线程私有池（一次分配，run 零分配）
    // - B 面板：KC×NR（行主，每行 NR 连续）
    // - 累加器：MR×NR（3×32）
    cpu_max_for_scratch: usize,
    b_panel_stride: usize, // kc*nr
    acc_stride: usize,     // mr*nr
    gate_panel_pool: Box<[T]>,
    up_panel_pool:   Box<[T]>,
    gate_acc_pool:   Box<[T]>,
    up_acc_pool:     Box<[T]>,
}

impl<T> MatMulSilu<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    /// Safety：传入的裸指针需保证后续访问安全
    pub fn new(
        a_ptr: *const T,
        wgate_ptr: *const T, // 输入为 K×N
        wup_ptr: *const T,   // 输入为 K×N
        out_ptr: *mut T,
        m_max: usize,
        n_max: usize,
        k_max: usize,
        mb: usize,
        nb: usize,
        kc: usize,
        mr: usize, // 3
        nr: usize, // 32
        cpu_max_for_scratch: usize,
    ) -> Self {
        // 1) 构造期一次性转置（K×N -> N×K）
        let wgate_buf = unsafe { make_b_nt::<T>(wgate_ptr, n_max, k_max) };
        let wup_buf   = unsafe { make_b_nt::<T>(wup_ptr,   n_max, k_max) };

        // 2) 线程私有池
        let b_panel_stride = kc.max(1) * nr.max(1);
        let acc_stride     = mr.max(1) * nr.max(1);
        let pool_elems_b   = cpu_max_for_scratch * b_panel_stride;
        let pool_elems_acc = cpu_max_for_scratch * acc_stride;

        let gate_panel_pool = vec![T::default(); pool_elems_b].into_boxed_slice();
        let up_panel_pool   = vec![T::default(); pool_elems_b].into_boxed_slice();
        let gate_acc_pool   = vec![T::default(); pool_elems_acc].into_boxed_slice();
        let up_acc_pool     = vec![T::default(); pool_elems_acc].into_boxed_slice();

        // 用转置后的缓冲覆盖 ptr2/ptr3（后续一律按 N×K 使用）
        let ptr2 = ConstPtr { ptr: wgate_buf.as_ptr() };
        let ptr3 = ConstPtr { ptr: wup_buf.as_ptr() };

        Self {
            ptr1: ConstPtr { ptr: a_ptr },
            ptr2,
            ptr3,
            output_ptr: MutPtr { ptr: out_ptr },
            params: MatmulParams {
                a_row_step_macro: mb,
                b_row_step_macro: nb,
                column_step_macro: kc,
                a_row_step_micro: mr,
                b_row_step_micro: nr,
            },
            m_max,
            n_max,
            k_max,
            _marker: PhantomData,

            wgate_buf,
            wup_buf,

            cpu_max_for_scratch,
            b_panel_stride,
            acc_stride,
            gate_panel_pool,
            up_panel_pool,
            gate_acc_pool,
            up_acc_pool,
        }
    }

    #[inline(always)]
    fn thread_slices(&self, tid: usize) -> (*mut T, *mut T, *mut T, *mut T) {
        unsafe {
            let gp = self.gate_panel_pool.as_ptr().add(tid * self.b_panel_stride) as *mut T;
            let up = self.up_panel_pool  .as_ptr().add(tid * self.b_panel_stride) as *mut T;
            let ga = self.gate_acc_pool  .as_ptr().add(tid * self.acc_stride)     as *mut T;
            let ua = self.up_acc_pool    .as_ptr().add(tid * self.acc_stride)     as *mut T;
            (gp, up, ga, ua)
        }
    }

    /// 从 B_nt[N×K]（行主；行距=K）打 kc×nr 连续面板
    #[inline(always)]
    unsafe fn pack_panel_from_bnt<U: Copy>(
        b_nt: *const U,  // [N×K] 行主
        ldb_row: usize,  // = K
        n0: usize,
        k0: usize,
        kc: usize,
        nr: usize,
        out: *mut U,     // 输出 KC×NR 行主
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

    /// 执行：按 tiles_m × tiles_n 分块；每微块（MR×NR=3×32）：
    /// - 循环 K 方向，反复打 KC×NR 面板并调用 compute1 累加到 gate_acc/up_acc
    /// - 整个 K 完成后，compute2 把 SiLU(gate) ⊙ up 写回 C
    pub fn run(
        &self,
        position_index: usize,
        position_interval: usize,
        batch_size: usize,
        cpu_num: usize,
        thread_id: usize,
    ) {
        unsafe {
            // 维度
            let m = batch_size;
            let n = self.n_max;
            let k = self.k_max;

            // 形状
            let mb = self.params.a_row_step_macro.max(1);
            let nb = self.params.b_row_step_macro.max(1);
            let kc = self.params.column_step_macro.max(1);
            let mr = self.params.a_row_step_micro.max(1);
            let nr = self.params.b_row_step_micro.max(1);

            debug_assert!(m % mr == 0 && n % nr == 0 && k % kc == 0);
            debug_assert!(thread_id < self.cpu_max_for_scratch && cpu_num <= self.cpu_max_for_scratch);

            // 基址/行距
            let a_base = self.ptr1.ptr;       // A
            let c_base = self.output_ptr.ptr; // C
            let lda = k;  // A 行距
            let ldc = n;  // C 行距

            // 转置后的权重（N×K；行距=K）
            let wgate_nt_ptr = self.ptr2.ptr;
            let wup_nt_ptr   = self.ptr3.ptr;
            let ldb_row = k;

            // 每线程缓冲
            let (gate_panel, up_panel, gate_acc, up_acc) = self.thread_slices(thread_id);

            // tiles
            let tiles_m = (m + mb - 1) / mb;
            let tiles_n = (n + nb - 1) / nb;

            if let Some((tb, te)) = assign(tiles_m * tiles_n, cpu_num, thread_id) {
                for t in tb..te {
                    let tm = t / tiles_n;
                    let tn = t % tiles_n;

                    let m0 = tm * mb;
                    let n0 = tn * nb;

                    let m_blk = (m - m0).min(mb);
                    let n_blk = (n - n0).min(nb);
                    debug_assert!(m_blk % mr == 0 && n_blk % nr == 0);

                    // NB 内按 NR 走列块；MB 内按 MR 走行块
                    let mut nt = 0;
                    while nt < n_blk {
                        let mut mi = 0;
                        while mi < m_blk {
                            // 清零 MR×NR 累加缓冲（线程本地）
                            for i in 0..(mr * nr) { *gate_acc.add(i) = T::default(); }
                            for i in 0..(mr * nr) { *up_acc  .add(i) = T::default(); }

                            // K 方向按 kc 面板累加
                            let mut k0 = 0;
                            while k0 < k {
                                // 打 gate / up 的 KC×NR 面板（从 N×K 权重中抽取）
                                Self::pack_panel_from_bnt::<T>(
                                    wgate_nt_ptr,
                                    ldb_row,
                                    n0 + nt,
                                    k0,
                                    kc,
                                    nr,
                                    gate_panel,
                                );
                                Self::pack_panel_from_bnt::<T>(
                                    wup_nt_ptr,
                                    ldb_row,
                                    n0 + nt,
                                    k0,
                                    kc,
                                    nr,
                                    up_panel,
                                );

                                // 当前微块 A 起点（3×kc）
                                let a_tile = a_base.add((m0 + mi) * lda + k0);

                                // —— per-kc 累加 —— //
                                // 传入：A_tile、两份面板；输出：两份 acc（3×32）
                                self.compute1(
                                    a_tile,
                                    gate_panel as *const T,
                                    up_panel   as *const T,
                                    gate_acc,
                                    up_acc,
                                );

                                k0 += kc;
                            }

                            // —— finalize：SiLU(gate) ⊙ up → C —— //
                            let c_tile = c_base.add((m0 + mi) * ldc + (n0 + nt));
                            self.compute2(gate_acc as *const T, up_acc as *const T, c_tile);

                            mi += mr;
                        }
                        nt += nr;
                    }
                }
            }
        }
    }
}

/* -------------------- SiluMatmulTrait 默认实现 -------------------- */

impl<T> SiluMatmulTrait<T> for MatMulSilu<T>
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
    ) {
        // 默认空实现（generic T 不做任何事）
    }

    default fn compute2(
        &self,
        _gate_acc: *const T,
        _up_acc: *const T,
        _c_tile: *mut T,
    ) {
        // 默认空实现
    }
}

/* -------------------------- f16 专用实现（AVX-512 FP16） -------------------------- */

impl SiluMatmulTrait<f16> for MatMulSilu<f16> {
    /// per-kc 累加：把 A×W_gate 与 A×W_up 的部分和累加进两个 3×32 的 acc
    fn compute1(
        &self,
        a_tile: *const f16,
        gate_panel: *const f16,
        up_panel: *const f16,
        gate_acc: *mut f16,
        up_acc: *mut f16,
    ) {
        // 调用期参数映射（update 阶段）：
        // - lda = K                -> a_row_step_macro = self.k_max
        // - ldc_acc = 32 (NR)      -> b_row_step_macro = self.params.b_row_step_micro
        // - kc = self.params.column_step_macro
        // - mr/nr 来自 params
        let call_param = MatMulParams {
            a_row_step_macro: self.k_max,                     // lda (=K)
            b_row_step_macro: self.params.b_row_step_micro,   // ldc_acc (=NR=32)
            column_step_macro: self.params.column_step_macro, // kc
            a_row_step_micro: self.params.a_row_step_micro,   // mr (=3)
            b_row_step_micro: self.params.b_row_step_micro,   // nr (=32)
        };

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::silu_ffn::silu_update_3x32(
                a_tile,
                gate_panel,
                up_panel,
                gate_acc,
                up_acc,
                &call_param,
            );
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        {
            unreachable!("avx512fp16 required for f16 SiluMatmulTrait::compute1");
        }
    }

    /// finalize：C = SiLU(gate_acc) ⊙ up_acc
    fn compute2(&self, gate_acc: *const f16, up_acc: *const f16, c_tile: *mut f16) {
        // 调用期参数映射（finalize 阶段）：
        // - ldc_out = N           -> b_row_step_macro = self.n_max
        // - mr/nr 来自 params
        let call_param = MatMulParams {
            a_row_step_macro: self.k_max,                     // unused here
            b_row_step_macro: self.n_max,                     // ldc_out (=N)
            column_step_macro: self.params.column_step_macro, // unused here
            a_row_step_micro: self.params.a_row_step_micro,   // mr
            b_row_step_micro: self.params.b_row_step_micro,   // nr
        };

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::silu_ffn::silu_finish_3x32(
                gate_acc,
                up_acc,
                c_tile,
                &call_param,
            );
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        {
            unreachable!("avx512fp16 required for f16 SiluMatmulTrait::compute2");
        }
    }
}
