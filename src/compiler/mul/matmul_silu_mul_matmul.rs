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
use super::super::assign::assign;
use super::mul_trait::Matmul3Trait; // 直接使用你的 trait（已改为 compute1/compute2 签名）

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
pub struct MatmulSilu<T> {
    pub ptr1: ConstPtr<T>,     // A[M×K]
    pub ptr2: ConstPtr<T>,     // W_gate[K×N]
    pub ptr3: ConstPtr<T>,     // W_up  [K×N]
    pub output_ptr: MutPtr<T>, // C[M×N]
    /// params 只承载 step 形状（MB/NB/KC/MR/NR）
    pub params: MatmulParams,
    pub m_max: usize,
    pub n_max: usize,
    pub k_max: usize,
    _marker: PhantomData<T>,

    // 构造期转置缓存：N×K，行主，行距=K
    pub wgate_nt: Option<Box<[T]>>,
    pub wup_nt: Option<Box<[T]>>,
}

impl<T> MatmulSilu<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    /// Safety：传入的裸指针需保证后续访问安全
    pub fn new(
        a_ptr: *const T,
        wgate_ptr: *const T,
        wup_ptr: *const T,
        out_ptr: *mut T,
        m_max: usize,
        n_max: usize,
        k_max: usize,
        mb: usize,
        nb: usize,
        kc: usize,
        mr: usize,
        nr: usize,
    ) -> Self {
        // 构造期一次性转置，避免 run() 或多线程重复转置
        let wgate_nt = unsafe { make_b_nt::<T>(wgate_ptr, n_max, k_max) };
        let wup_nt = unsafe { make_b_nt::<T>(wup_ptr, n_max, k_max) };

        Self {
            ptr1: ConstPtr { ptr: a_ptr },
            ptr2: ConstPtr { ptr: wgate_ptr },
            ptr3: ConstPtr { ptr: wup_ptr },
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
            wgate_nt: Some(wgate_nt),
            wup_nt: Some(wup_nt),
        }
    }

    #[inline(always)]
    unsafe fn pack_panel<U: Copy>(
        b_nt: *const U,
        ldb_row: usize,
        n0: usize,
        k0: usize,
        kc: usize,
        nr: usize,
        out: *mut U, // 输出 KC×NR 行主，每行 NR 连续
    ) {
        for p in 0..kc {
            let src_col = k0 + p;
            let dst_row = out.add(p * nr);
            for lane in 0..nr {
                let j = n0 + lane;
                let src = b_nt.add(j * ldb_row + src_col); // b_nt[j, src_col]
                *dst_row.add(lane) = *src;
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
        /*
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

            // 基址/行距
            let a_base = self.ptr1.ptr; // A
            let c_base = self.output_ptr.ptr; // C
            let lda = k; // A 行距
            let ldc = n; // C 行距

            // 转置缓存
            let wgate_nt = self.wgate_nt.as_ref().expect("wgate_nt not initialized");
            let wup_nt = self.wup_nt.as_ref().expect("wup_nt not initialized");
            let (wgate_nt_ptr, wup_nt_ptr, ldb_row) = (wgate_nt.as_ptr(), wup_nt.as_ptr(), k);

            // 瓦片计数
            let tiles_m = (m + mb - 1) / mb;
            let tiles_n = (n + nb - 1) / nb;

            if let Some((tb, te)) = assign(tiles_m * tiles_n, cpu_num, thread_id) {
                // 线程私有 KC×NR 面板
                let mut gate_panel: Vec<T> = vec![T::default(); kc * nr];
                let mut up_panel: Vec<T> = vec![T::default(); kc * nr];

                // 微块累加缓冲：固定 MR×NR（3×32）
                let mut gate_acc: Vec<T> = vec![T::default(); mr * nr];
                let mut up_acc: Vec<T> = vec![T::default(); mr * nr];

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
                            // 清零微块累加缓冲
                            for x in gate_acc.iter_mut() {
                                *x = T::default();
                            }
                            for x in up_acc.iter_mut() {
                                *x = T::default();
                            }

                            // K 方向按 kc 面板累加
                            let mut k0 = 0;
                            while k0 < k {
                                // 打 gate / up 的 KC×NR 面板
                                Self::pack_panel::<T>(
                                    wgate_nt_ptr,
                                    ldb_row,
                                    n0 + nt,
                                    k0,
                                    kc,
                                    nr,
                                    gate_panel.as_mut_ptr(),
                                );
                                Self::pack_panel::<T>(
                                    wup_nt_ptr,
                                    ldb_row,
                                    n0 + nt,
                                    k0,
                                    kc,
                                    nr,
                                    up_panel.as_mut_ptr(),
                                );

                                // 当前微块 A 起点（3×kc）
                                let a_tile = a_base.add((m0 + mi) * lda + k0);

                                // —— per-kc 累加 —— //
                                // 传入：A_tile、两份面板；输出：两份 acc（3×32）
                                self.compute1(
                                    a_tile,
                                    gate_panel.as_ptr(),
                                    up_panel.as_ptr(),
                                    gate_acc.as_mut_ptr(),
                                    up_acc.as_mut_ptr(),
                                );

                                k0 += kc;
                            }

                            // —— finalize：SiLU(gate) ⊙ up → C —— //
                            let c_tile = c_base.add((m0 + mi) * ldc + (n0 + nt));
                            self.compute2(gate_acc.as_ptr(), up_acc.as_ptr(), c_tile);

                            mi += mr;
                        }
                        nt += nr;
                    }
                }
            }
        }*/
    } 
}

/* -------------------- 默认实现（占位，保持与 matmul 一致的模式） -------------------- */

impl<T> Matmul3Trait<T> for MatmulSilu<T>
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
        // 默认空实现（初始版本只提供 f16 专用实现）
    }

    default fn compute2(&self, _gate_acc: *const T, _up_acc: *const T, _c_tile: *mut T) {
        // 默认空实现
    }
}

/* -------------------------- f16 专用实现（AVX-512 FP16） -------------------------- */

impl Matmul3Trait<f16> for MatmulSilu<f16> {
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
        // - ldc_acc = 32 (NR)     -> b_row_step_macro = self.params.b_row_step_micro
        // - kc = self.params.column_step_macro
        // - mr/nr 来自 params
        let call_param = MatmulParams {
            a_row_step_macro: self.k_max,                     // lda (=K)
            b_row_step_macro: self.params.b_row_step_micro,   // ldc_acc (=NR=32)
            column_step_macro: self.params.column_step_macro, // kc
            a_row_step_micro: self.params.a_row_step_micro,   // mr (=3)
            b_row_step_micro: self.params.b_row_step_micro,   // nr (=32)
        };

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::fused_gate_up_silu_mul_block::fused_update_gate_up_acc_block(
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
            unreachable!("avx512fp16 required for f16 path");
        }
    }

    /// finalize：C = SiLU(gate_acc) ⊙ up_acc
    fn compute2(&self, gate_acc: *const f16, up_acc: *const f16, c_tile: *mut f16) {
        // 调用期参数映射（finalize 阶段）：
        // - ldc_out = N           -> b_row_step_macro = self.n_max
        // - mr/nr 来自 params
        let call_param = MatmulParams {
            a_row_step_macro: self.k_max,                     // unused here
            b_row_step_macro: self.n_max,                     // ldc_out (=N)
            column_step_macro: self.params.column_step_macro, // unused here
            a_row_step_micro: self.params.a_row_step_micro,   // mr
            b_row_step_micro: self.params.b_row_step_micro,   // nr
        };

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::fused_gate_up_silu_mul_block::fused_finalize_gate_up_silu_mul_block(
                gate_acc, up_acc, c_tile, &call_param,
            );
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        {
            unreachable!("avx512fp16 required for f16 path");
        }
    }
}
