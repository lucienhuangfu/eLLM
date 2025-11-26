// === runner/matmul.rs ===
#![allow(non_snake_case)]

use std::f16;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

use super::super::super::init::{
    matmul_params::MatMulParams,
    send_sync_ptr::{ConstPtr, MutPtr},
};
use super::super::super::kernel;
use super::super::assign::assign;
use super::mul_trait::MatMulTrait;

#[derive(Clone)]
pub struct MatMul<T> {
    pub ptr1: ConstPtr<T>,       // A[S×M×K] 首地址
    pub ptr2: ConstPtr<T>,       // 构造后即指向 B_nt[N×K]
    pub output_ptr: MutPtr<T>,   // C[S×M×N] 首地址
    pub output_to_kv: bool,      // 保持兼容（你的旧逻辑）

    /// 仅承载 step 形状（MB/NB/KC/MR/NR）
    pub params: MatMulParams,
    pub _marker: PhantomData<T>,

    // “最大维度” M/N/K（替代旧 params.a_row/b_row/column）
    pub m_max: usize,
    pub n_max: usize,
    pub k_max: usize,

    // 构造期转置得到的 B_nt（N×K，行主；行距=K）
    b_nt_buf: Box<[T]>,

    // —— 线程私有的 KC×NR 面板池（连续大块，按线程切片）——
    // 布局： [ cpu_max_for_scratch ][ kc * nr ]
    b_panel_pool: Box<[T]>,
    b_panel_stride_elems: usize,   // = kc * nr
    cpu_max_for_scratch: usize,    // 允许的最大线程数
}

impl<T> MatMul<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    /// 构造函数：一次性完成
    /// 1) B[K×N] → B_nt[N×K] 转置，并让 `ptr2` 直接指向 B_nt
    /// 2) 为每个线程预分配 KC×NR 面板（运行期不再分配）
    ///
    /// Safety：假定传入裸指针的可读写范围满足后续访问
    pub unsafe fn new(
        ptr1: *const T,              // A
        ptr2_b_kxn: *const T,        // B[K×N]（只在构造期使用一次）
        output_ptr: *mut T,          // C
        output_to_kv: bool,
        params: MatMulParams,        // 仅 step 形状：MB/NB/KC/MR/NR
        m_max: usize,
        n_max: usize,
        k_max: usize,
        cpu_max_for_scratch: usize,  // 运行期线程数上限（保证不再分配）
    ) -> Self {
        // === (1) 构造期完成 B → B_nt，全量转置 ===
        // 原 B：K×N（行主，行距=N）
        // 目标：N×K（行主，行距=K）
        let mut b_nt_vec: Vec<T> = vec![T::default(); n_max * k_max];
        let b_nt_ptr = b_nt_vec.as_mut_ptr();

        for kk in 0..k_max {
            let b_row = ptr2_b_kxn.add(kk * n_max); // B[kk, :]
            for jj in 0..n_max {
                // B_nt[jj, kk] = B[kk, jj]
                *b_nt_ptr.add(jj * k_max + kk) = *b_row.add(jj);
            }
        }
        let b_nt_box = b_nt_vec.into_boxed_slice();
        let b_nt_base = b_nt_box.as_ptr();

        // === (2) 预分配“线程私有 KC×NR 面板池” ===
        let kc = params.column_step_macro.max(1);
        let nr = params.b_row_step_micro.max(1);
        let b_panel_stride_elems = kc * nr; // 每线程一段就是这么大
        let pool_len = cpu_max_for_scratch * b_panel_stride_elems;
        let b_panel_pool: Vec<T> = vec![T::default(); pool_len];

        Self {
            ptr1: ConstPtr { ptr: ptr1 },
            ptr2: ConstPtr { ptr: b_nt_base }, // 覆盖为 B_nt
            output_ptr: MutPtr { ptr: output_ptr },
            output_to_kv,
            params,
            _marker: PhantomData,
            m_max,
            n_max,
            k_max,
            b_nt_buf: b_nt_box,
            b_panel_pool: b_panel_pool.into_boxed_slice(),
            b_panel_stride_elems,
            cpu_max_for_scratch,
        }
    }

    /// 取得本线程的 KC×NR 面板指针（不分配）
    #[inline(always)]
    pub fn thread_b_panel_ptr(&self, thread_id: usize) -> *mut T {
        debug_assert!(thread_id < self.cpu_max_for_scratch);
        unsafe {
            self.b_panel_pool
                .as_ptr()
                .add(thread_id * self.b_panel_stride_elems) as *mut T
        }
    }

    /// 执行：S×M×N 的三分块调度 + 使用预分配的线程面板
    pub fn run(
        &self,
        position_index: usize,
        position_interval: usize,
        batch_size: usize,
        cpu_num: usize,
        thread_id: usize,
    ) {
        unsafe {
            // ===== 维度 =====
            let m = batch_size;     // 本次 M
            let n = self.n_max;     // N
            let k = self.k_max;     // K

            // ===== 分块参数（来自 params，仅形状）=====
            let mb = self.params.a_row_step_macro.max(1);
            let nb = self.params.b_row_step_macro.max(1);
            let kc = self.params.column_step_macro.max(1);
            let mr = self.params.a_row_step_micro.max(1);
            let nr = self.params.b_row_step_micro.max(1);

            debug_assert!(m % mr == 0);
            debug_assert!(n % nr == 0);
            debug_assert!(k % kc == 0);
            debug_assert!(cpu_num <= self.cpu_max_for_scratch);
            debug_assert!(thread_id < cpu_num);

            // ===== 基址与行距（元素计）=====
            let a_base = self.ptr1.ptr;       // A[S×M×K]
            let c_base = self.output_ptr.ptr; // C[S×M×N]
            let lda = k; // A 每行跨度
            let ldc = n; // C 每行跨度

            // ===== 序列范围 =====
            let s_begin = position_index;
            let s_end = position_index + position_interval;
            let s_len = s_end - s_begin;
            let a_seq_stride = m * k;
            let c_seq_stride = m * n;

            // ===== 使用构造期转置的 B_nt（N×K，行主；行距=K）=====
            let b_nt_ptr = self.ptr2.ptr; // 已经是 B_nt
            let ldb_row = k;

            // ===== 取本线程的 KC×NR 面板切片（不分配）=====
            let b_panel_ptr = self.thread_b_panel_ptr(thread_id);

            #[inline(always)]
            unsafe fn pack_b_panel<T: Copy>(
                b_nt: *const T, // [N×K] 行主
                ldb_row: usize, // = K
                n0: usize,      // N 起点
                k0: usize,      // K 起点
                kc: usize,
                nr: usize,
                out: *mut T,    // 输出：KC×NR 行主（长度 = kc*nr）
            ) {
                for p in 0..kc {
                    let src_col = k0 + p;
                    let dst_row = out.add(p * nr);
                    for lane in 0..nr {
                        let j = n0 + lane;
                        let src = b_nt.add(j * ldb_row + src_col);
                        *dst_row.add(lane) = *src;
                    }
                }
            }

            // ===== 任务切分：S × tiles_m × tiles_n =====
            let tiles_m = (m + mb - 1) / mb;
            let tiles_n = (n + nb - 1) / nb;
            let tiles_sn = s_len * tiles_m * tiles_n;

            if let Some((tb, te)) = assign(tiles_sn, cpu_num, thread_id) {
                for t in tb..te {
                    let s_rel = t / (tiles_m * tiles_n);
                    let rem   = t % (tiles_m * tiles_n);
                    let tm    = rem / tiles_n;
                    let tn    = rem % tiles_n;

                    let s  = s_begin + s_rel;
                    let m0 = tm * mb;
                    let n0 = tn * nb;

                    let m_blk = (m - m0).min(mb);
                    let n_blk = (n - n0).min(nb);
                    debug_assert!(m_blk % mr == 0 && n_blk % nr == 0);

                    let a_base_s = a_base.add(s * a_seq_stride);
                    let c_base_s = c_base.add(s * c_seq_stride);

                    // Kc 循环
                    let mut k0 = 0;
                    while k0 < k {
                        // NB 内分 NR 小块：为当前 (k0..k0+kc, n0+nt..+nr) 打面板
                        let mut nt = 0;
                        while nt < n_blk {
                            pack_b_panel::<T>(
                                b_nt_ptr, ldb_row,
                                n0 + nt, k0,
                                kc, nr,
                                b_panel_ptr,
                            );

                            // M 方向按 MR 行组走微核
                            let mut mi = 0;
                            while mi < m_blk {
                                let a_tile = a_base_s.add((m0 + mi) * lda + k0);
                                let c_tile = c_base_s.add((m0 + mi) * ldc + (n0 + nt));

                                self.compute(a_tile, b_panel_ptr as *const T, c_tile);
                                mi += mr;
                            }
                            nt += nr;
                        }
                        k0 += kc;
                    }
                }
            }
        }
    }
}

/* ------------------ compute/compute2：保持你的调用风格 ------------------ */

impl<T> MatMulTrait<T> for MatMul<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    default fn compute(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T) {
        let call_param = MatMulParams {
            a_row_step_macro: self.k_max,                     // lda = K
            b_row_step_macro: self.n_max,                     // ldc = N
            column_step_macro: self.params.column_step_macro, // kc
            a_row_step_micro: self.params.a_row_step_micro,   // mr
            b_row_step_micro: self.params.b_row_step_micro,   // nr
        };

        kernel::generic::matmul_block::matmul_block(
            input_ptr1,
            input_ptr2,
            output_ptr,
            &call_param,
        );
    }

    default fn compute2(
        &self,
        input_ptr1: *const T,
        input_ptr2: *const T,
        output_ptr: *mut T,
        length: usize,
    ) {
        kernel::generic::dot_product::dot_product(input_ptr1, input_ptr2, output_ptr, length);
    }
}

impl MatMulTrait<f16> for MatMul<f16> {
    fn compute(&self, input_ptr1: *const f16, input_ptr2: *const f16, output_ptr: *mut f16) {
        let call_param = MatMulParams {
            a_row_step_macro: self.k_max,                     // lda = K
            b_row_step_macro: self.n_max,                     // ldc = N
            column_step_macro: self.params.column_step_macro, // kc
            a_row_step_micro: self.params.a_row_step_micro,   // mr (=3)
            b_row_step_micro: self.params.b_row_step_micro,   // nr (=32)
        };

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::matmul_block::matmul_block(
                input_ptr1,
                input_ptr2,
                output_ptr,
                &call_param,
            );
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::generic::matmul_block::matmul_block(
            input_ptr1,
            input_ptr2,
            output_ptr,
            &call_param,
        );
    }

    fn compute2(
        &self,
        input_ptr1: *const f16,
        input_ptr2: *const f16,
        output_ptr: *mut f16,
        length: usize,
    ) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::dot_product::dot_product(
                input_ptr1, input_ptr2, output_ptr, length,
            );
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::generic::dot_product::dot_product(input_ptr1, input_ptr2, output_ptr, length);
    }
}

impl MatMulTrait<f32> for MatMul<f32> {
    fn compute(&self, _a: *const f32, _b: *const f32, _c: *mut f32) { /* TODO */ }
    fn compute2(
        &self, a: *const f32, b: *const f32, c: *mut f32, length: usize,
    ) { kernel::generic::dot_product::dot_product(a, b, c, length); }
}