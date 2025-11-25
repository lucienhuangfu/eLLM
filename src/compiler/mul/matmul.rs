// === runner/matmul.rs ===
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
use super::mul_trait::MatmulTrait;

#[derive(Clone)]
pub struct Matmul<T> {
    pub ptr1: ConstPtr<T>,     // A[S×M×K] 首地址
    pub ptr2: ConstPtr<T>,     // B[K×N]   首地址（保留原始指针）
    pub output_ptr: MutPtr<T>, // C[S×M×N] 首地址
    pub output_to_kv: bool,    // 保持兼容
    /// 注意：`params` 仅承载 **step 形状**（MB/NB/KC/MR/NR）
    pub params: MatmulParams,
    pub _marker: PhantomData<T>,

    // 保存“最大维度” M/N/K（替代旧 params.a_row/b_row/column）
    pub m_max: usize,
    pub n_max: usize,
    pub k_max: usize,

    // 构造期转置得到的 B_nt（N×K，行主；行距=K）
    pub b_nt: Option<Box<[T]>>,
}

impl<T> Matmul<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    /// 构造函数：在此完成 **一次性** B[K×N] → B_nt[N×K] 的全量转置
    ///
    /// Safety：假定传入裸指针的可读写范围满足后续访问
    pub unsafe fn new(
        ptr1: *const T,
        ptr2: *const T,
        output_ptr: *mut T,
        output_to_kv: bool,
        params: MatmulParams,
        m_max: usize,
        n_max: usize,
        k_max: usize,
    ) -> Self {
        println!(
            "Matmul::new called with m_max={}, n_max={}, k_max={}",
            m_max, n_max, k_max
        );
        // 直接在 new() 内完成转置，避免任何线程或 run() 内重复
        let mut b_nt_vec: Vec<T> = vec![T::default(); n_max * k_max];
        let b_nt_ptr = b_nt_vec.as_mut_ptr();

        println!("Transposing B matrix in Matmul::new");
        // 原始 B 为 K×N（行主，行距 = N）
        // 目标 B_nt 为 N×K（行主，行距 = K）
        for kk in 0..k_max {
            let b_row = ptr2.add(kk * n_max); // B[kk, 0]
            for jj in 0..n_max {
                // B_nt[jj, kk] = B[kk, jj]
                *b_nt_ptr.add(jj * k_max + kk) = *b_row.add(jj);
            }
        }
        println!("Finished transposing B matrix in Matmul::new");
        Self {
            ptr1: ConstPtr { ptr: ptr1 },
            ptr2: ConstPtr { ptr: ptr2 },
            output_ptr: MutPtr { ptr: output_ptr },
            output_to_kv,
            params,

            _marker: PhantomData,
            m_max,
            n_max,
            k_max,
            b_nt: Some(b_nt_vec.into_boxed_slice()),
        }
    }

    /// 执行：S×M×N 的三分块调度 + 线程私有 KC×NR 面板（仅 packing，不做转置）
    pub fn run(&self, batch_size: usize, cpu_num: usize, thread_id: usize) {
        unsafe {
            // ===== 维度 =====
            let m = batch_size; // 本次 M
            let n = self.n_max; // N
            let k = self.k_max; // K

            // ===== 分块参数（来自 params，仅形状）=====
            let mb = self.params.a_row_step_macro.max(1);
            let nb = self.params.b_row_step_macro.max(1);
            let kc = self.params.column_step_macro.max(1);
            let mr = self.params.a_row_step_micro.max(1);
            let nr = self.params.b_row_step_micro.max(1);

            // println!( "n = {}, nr = {}", n, nr);

            debug_assert!(m % mr == 0);
            debug_assert!(n % nr == 0);
            debug_assert!(k % kc == 0);

            // ===== 基址与行距（元素计）=====
            let a_base = self.ptr1.ptr; // A[M×K]
            let c_base = self.output_ptr.ptr; // C[M×N]
            let lda = k; // A 每行跨度
            let ldc = n; // C 每行跨度

            // ===== 使用构造期转置的 B_nt（N×K，行主；行距=K）=====
            let (b_nt_ptr, ldb_row) = {
                let bnt = self.b_nt.as_ref().expect("B_nt not initialized");
                (bnt.as_ptr(), k)
            };

            // ===== 任务切分：tiles_m × tiles_n =====
            let tiles_m = (m + mb - 1) / mb;
            let tiles_n = (n + nb - 1) / nb;
            let tiles_total = tiles_m * tiles_n;

            if let Some((tb, te)) = assign(tiles_total, cpu_num, thread_id) {
                // 线程私有 KC×NR 面板（packing；不是转置）
                let mut b_panel: Vec<T> = vec![T::default(); kc * nr];

                #[inline(always)]
                unsafe fn pack_b_panel<T: Copy>(
                    b_nt: *const T, // [N×K] 行主
                    ldb_row: usize, // = K
                    n0: usize,      // N 起点
                    k0: usize,      // K 起点
                    kc: usize,
                    nr: usize,
                    out: *mut T, // 输出：KC×NR 行主
                ) {
                    // 从 B_nt 的 (n0..n0+nr, k0..k0+kc) 抽取到连续的 KC×NR 面板
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

                for t in tb..te {
                    let tm = t / tiles_n;
                    let tn = t % tiles_n;

                    let m0 = tm * mb;
                    let n0 = tn * nb;

                    let m_blk = (m - m0).min(mb);
                    let n_blk = (n - n0).min(nb);
                    debug_assert!(m_blk % mr == 0 && n_blk % nr == 0);

                    // Kc 循环
                    let mut k0 = 0;
                    while k0 < k {
                        // NB 内分 NR 小块
                        let mut nt = 0;
                        while nt < n_blk {
                            // 打一块 KC×NR 面板（仅当前块所需）
                            pack_b_panel::<T>(
                                b_nt_ptr,
                                ldb_row,
                                n0 + nt,
                                k0,
                                kc,
                                nr,
                                b_panel.as_mut_ptr(),
                            );

                            // M 方向按 MR 行组走微核（保持 compute 调用）
                            let mut mi = 0;
                            while mi < m_blk {
                                let a_tile = a_base.add((m0 + mi) * lda + k0);
                                let c_tile = c_base.add((m0 + mi) * ldc + (n0 + nt));

                                // 只调用 compute；真实 lda/ldc/kc/mr/nr 在 compute 内组装
                                self.compute(a_tile, b_panel.as_ptr(), c_tile);

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

/* ------------------ 下面是 compute/compute2 的实现（仅此处改“调用 param”） ------------------ */

impl<T> MatmulTrait<T> for Matmul<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    /// generic 版本：在这里组装“调用 param”
    default fn compute(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T) {
        let call_param = MatmulParams {
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

impl MatmulTrait<f16> for Matmul<f16> {
    /// f16 专用：同样在这里组装“调用 param”，而不是用 self.params 直接透传
    fn compute(&self, input_ptr1: *const f16, input_ptr2: *const f16, output_ptr: *mut f16) {
        let call_param = MatmulParams {
            a_row_step_macro: self.k_max,                     // lda = K
            b_row_step_macro: self.n_max,                     // ldc = N
            column_step_macro: self.params.column_step_macro, // kc
            a_row_step_micro: self.params.a_row_step_micro,   // mr (=3)
            b_row_step_micro: self.params.b_row_step_micro,   // nr (=32)
        };

        // 平台选择：有 AVX512-FP16 则走 3x32 广播微核，否则 generic
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

impl MatmulTrait<f32> for Matmul<f32> {
    fn compute(&self, _a: *const f32, _b: *const f32, _c: *mut f32) { /* TODO */
    }
    fn compute2(&self, a: *const f32, b: *const f32, c: *mut f32, length: usize) {
        // kernel::generic::dot_product::dot_product(a, b, c, length);
    }
}

/* ---------------------------------- 测试 ---------------------------------- */

/*
#[cfg(test)]
mod innteg_tests {
    use super::*;
    use std::arch::is_x86_feature_detected;
    use std::f16;

    #[inline]
    fn f16v(v: f32) -> f16 {
        let h = half::f16::from_f32(v);
        f16::from_bits(h.to_bits())
    }
    #[inline]
    fn to_f32(x: f16) -> f32 {
        half::f16::from_bits(x.to_bits()).to_f32()
    }
    fn all_close(a: &[f16], b: &[f16], tol: f32) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| (to_f32(*x) - to_f32(*y)).abs() <= tol)
    }

    /// 单序列整除路径：验证 run + compute（compute 内部组装调用 param）
    #[test]
    fn test_run_internal_transpose_f16_m6_n64_k64() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("Skipping avx512fp16");
            return;
        }
        unsafe {
            let m = 6usize;
            let n = 64usize;
            let k = 64usize;
            let mb = 6usize;
            let nb = 64usize;
            let kc = 64usize;
            let mr = 3usize;
            let nr = 32usize;

            let a: Vec<f16> = (0..m * k).map(|_| f16v(1.0)).collect();
            let b_orig: Vec<f16> = (0..k * n).map(|_| f16v(1.0)).collect();
            let mut c: Vec<f16> = (0..m * n).map(|_| f16v(0.0)).collect();

            let params = MatmulParams {
                a_row_step_macro: mb,
                b_row_step_macro: nb,
                column_step_macro: kc,
                a_row_step_micro: mr,
                b_row_step_micro: nr,
            };

            // 用 new()，构造期一次性转置 B -> B_nt（不使用 Arc）
            let op = Matmul::<f16>::new(
                a.as_ptr(),
                b_orig.as_ptr(),
                c.as_mut_ptr(),
                false,
                params,
                m,
                n,
                k,
            );

            op.run(m, 1, 0);

            let expected: Vec<f16> = (0..m * n).map(|_| f16v(k as f32)).collect();
            assert!(all_close(&c, &expected, 1e-3));
        }
    }

    /// 多序列切片：只处理 s=1,2
    #[test]
    fn test_run_internal_transpose_f16_seqlen_gt1_slice() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("Skipping avx512fp16");
            return;
        }
        unsafe {
            let s_total = 4usize;
            let m = 6usize;
            let n = 64usize;
            let k = 64usize;
            let mb = 6usize;
            let nb = 64usize;
            let kc = 64usize;
            let mr = 3usize;
            let nr = 32usize;

            let a_len = s_total * m * k;
            let a: Vec<f16> = (0..a_len).map(|_| f16v(1.0)).collect();
            let b_orig: Vec<f16> = (0..k * n).map(|_| f16v(1.0)).collect();
            let mut c: Vec<f16> = (0..s_total * m * n).map(|_| f16v(0.0)).collect();

            let params = MatmulParams {
                a_row_step_macro: mb,
                b_row_step_macro: nb,
                column_step_macro: kc,
                a_row_step_micro: mr,
                b_row_step_micro: nr,
            };

            let op = Matmul::<f16>::new(
                a.as_ptr(),
                b_orig.as_ptr(),
                c.as_mut_ptr(),
                false,
                params,
                m,
                n,
                k,
            );

            // 只处理 s=1,2
            op.run(1, 2, m, 1, 0);

            let c_seq_stride = m * n;
            let view = |s: usize| -> &[f16] {
                let off = s * c_seq_stride;
                &c[off..off + c_seq_stride]
            };

            let expected: Vec<f16> = (0..m * n).map(|_| f16v(k as f32)).collect();
            // s=0、3 未处理
            assert!(view(0).iter().all(|&x| to_f32(x) == 0.0));
            assert!(view(3).iter().all(|&x| to_f32(x) == 0.0));
            // s=1、2 处理为 K
            assert!(all_close(view(1), &expected, 1e-3));
            assert!(all_close(view(2), &expected, 1e-3));
        }
    }
}*/
