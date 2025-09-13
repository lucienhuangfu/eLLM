use std::f16;
use std::marker::PhantomData;
use std::ops::{Add, Mul};
// use std::sync::{Arc, Barrier};

use super::super::super::init::{
    matmul_params::MatMulParams,
    send_sync_ptr::{ConstPtr, MutPtr},
};
use super::super::super::kernel;
use super::super::assign::assign;
use super::mul_trait::MatlMulTrait;

// there will be just one instance of this runner in the program
// this runner will be shared by many threads that together compute the matrix multiplication
#[derive(Clone)]
pub struct MatMul<T> {
    ptr1: ConstPtr<T>,
    ptr2: ConstPtr<T>,
    output_ptr: MutPtr<T>,
    // sequence_length: usize,
    output_to_kv: bool,
    pub params: MatMulParams,
    _marker: PhantomData<T>,
    // sequence_stride: usize,
    // batch_size: usize,
    // hidden_size: usize,
}
impl<T> MatMul<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    pub fn new(
        ptr1: *const T,
        ptr2: *const T,
        output_ptr: *mut T,
        // sequence_length: usize,
        output_to_kv: bool,

        // these are the parameters of the matrix multiplication, this matrix is a largest possible one
        // for later matrix multiplication, the actual size of the matrix will be smaller
        // so this is reserving enough spaces in memory, and later lay the data into a small portion of it
        // and as we compute, we just access and calculate with the data in the small portion
        // this is like we construct a big playground, and we only play in a small or big portion of it, depending on how many people there are
        // so these dimensions are the dimensions of the largest possible matrix
        a_row: usize,
        b_row: usize,
        column: usize,
        // these are the sizes of the macro kernels
        // how they are determined is not clear
        a_row_step_macro: usize,
        b_row_step_macro: usize,
        column_step_macro: usize,
        // these are the sizes of the micro kernels
        // how they are determined is not clear
        a_row_step_micro: usize,
        b_row_step_micro: usize,
    ) -> Self {
        Self {
            ptr1: ConstPtr { ptr: ptr1 },
            ptr2: ConstPtr { ptr: ptr2 },
            output_ptr: MutPtr { ptr: output_ptr },
            // sequence_length: sequence_length,
            output_to_kv: output_to_kv,
            params: MatMulParams {
                a_row,
                b_row,
                column,
                a_row_step_macro,
                b_row_step_macro,
                column_step_macro,
                a_row_step_micro,
                b_row_step_micro,
            },
            _marker: PhantomData,
        }
    }

    pub fn run(
        &self,
        position_index: usize,    // 序列起点 s_begin
        position_interval: usize, // 要处理的序列数（即 sequence_length 的这次切片）
        batch_size: usize,        // M
        cpu_num: usize,           // 线程总数
        thread_id: usize,         // 本线程 id
    ) {
        unsafe {
            // ----- 维度 -----
            let m = batch_size; // M
            let n = self.params.b_row; // N
            let k = self.params.column; // K

            // ----- 三分块参数 -----
            let mb = self.params.a_row_step_macro.max(1); // MB
            let nb = self.params.b_row_step_macro.max(1); // NB
            let kc = self.params.column_step_macro.max(1); // KC
            let mr = self.params.a_row_step_micro.max(1); // MR (建议=3)
            let nr = self.params.b_row_step_micro.max(1); // NR (建议=32)

            // 整除假设：无需尾块分支
            debug_assert!(m % mr == 0, "要求 M 能被 MR 整除");
            debug_assert!(n % nr == 0, "要求 N 能被 NR 整除");
            debug_assert!(k % kc == 0, "要求 K 能被 KC 整除");

            // ----- 基址/行距（元素计）-----
            // A: [S×M×K] 行主；B_orig: [K×N] 行主；C: [S×M×N] 行主
            let a_base = self.ptr1.ptr;
            let b_orig = self.ptr2.ptr; // 原始 B: [K×N] 行主
            let c_base = self.output_ptr.ptr;
            let lda = k; // A 每行跨度
            let ldc = n; // C 每行跨度

            // =====序列范围与 stride =====
            let s_begin = position_index;
            let s_end = position_index + position_interval;
            let s_len = s_end - s_begin;

            let a_seq_stride = m * k; // A[s+1] 相对 A[s] 的偏移
            let c_seq_stride = m * n; // C[s+1] 相对 C[s] 的偏移

            // ===== 每线程：把 B[K×N] → B_nt[N×K] 转置一次 =====
            // b_nt[j, k] = b_orig[k, j]
            let mut b_nt: Vec<T> = vec![T::default(); n * k];
            for kk in 0..k {
                let src_row = b_orig.add(kk * n); // b_orig[kk, 0]
                for jj in 0..n {
                    *b_nt.as_mut_ptr().add(jj * k + kk) = *src_row.add(jj);
                }
            }
            let b_nt_ptr = b_nt.as_ptr(); // 转置后基址（[N×K] 行主）
            let ldb_row = k; // b_nt 的行距（=K）

            // ===== 任务切分：S × tiles_m × tiles_n =====
            let tiles_m = (m + mb - 1) / mb;
            let tiles_n = (n + nb - 1) / nb;
            let tiles_sn = s_len * tiles_m * tiles_n;

            if let Some((tb, te)) = assign(tiles_sn, cpu_num, thread_id) {
                // 线程私有：KC×NR 面板缓存
                let mut b_panel: Vec<T> = vec![T::default(); kc * nr];

                // 打包函数：把 b_nt[n0..n0+NR, k0..k0+KC] 打成 (KC×NR) 行主
                #[inline(always)]
                unsafe fn pack_b_panel<T: Copy>(
                    b_nt: *const T, // [N×K] 行主
                    ldb_row: usize, // = K
                    n0: usize,      // 本 NR 块在 N 维起点
                    k0: usize,      // 本 KC 块在 K 维起点
                    kc: usize,      // KC
                    nr: usize,      // NR
                    out: *mut T,    // 输出面板
                ) {
                    for p in 0..kc {
                        let src_col = k0 + p; // K 维
                        let dst_row = out.add(p * nr); // 面板第 p 行
                        for lane in 0..nr {
                            let j = n0 + lane; // N 维
                            let src = b_nt.add(j * ldb_row + src_col); // b_nt[j, src_col]
                            *dst_row.add(lane) = *src;
                        }
                    }
                }

                // 线性任务号 t ∈ [tb, te) → (s, tm, tn)
                for t in tb..te {
                    let t_in = t; // [0, s_len*tiles_m*tiles_n)
                    let s_rel = t_in / (tiles_m * tiles_n);
                    let rem = t_in % (tiles_m * tiles_n);
                    let tm = rem / tiles_n;
                    let tn = rem % tiles_n;

                    let s = s_begin + s_rel;
                    let m0 = tm * mb;
                    let n0 = tn * nb;

                    // 当前宏块尺寸（和整除假设一致，无 remainder）
                    let m_blk = (m - m0).min(mb);
                    let n_blk = (n - n0).min(nb);
                    debug_assert!(m_blk % mr == 0 && n_blk % nr == 0);

                    // 本序列的 A/C 基址
                    let a_base_s = a_base.add(s * a_seq_stride);
                    let c_base_s = c_base.add(s * c_seq_stride);

                    // KC 循环（整除）
                    let mut k0 = 0;
                    while k0 < k {
                        // NB 内分 NR 小块
                        let mut nt = 0;
                        while nt < n_blk {
                            // 打一块 (KC×NR) 面板，复用给下面所有 MR 行组
                            pack_b_panel::<T>(
                                b_nt_ptr,
                                ldb_row,
                                n0 + nt,
                                k0,
                                kc,
                                nr,
                                b_panel.as_mut_ptr(),
                            );

                            // M 方向按 MR 行组：整块走微核
                            let mut mi = 0;
                            while mi < m_blk {
                                let a_tile = a_base_s.add((m0 + mi) * lda + k0); // A[(s,m0+mi), k0]
                                let c_tile = c_base_s.add((m0 + mi) * ldc + (n0 + nt)); // C[(s,m0+mi), (n0+nt)]
                                                                                        // 微核：内部对 k in 0..KC 广播 A 的3个标量，load b_panel[k,*] 做 FMA，+= 写回 C
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

impl<T> MatlMulTrait<T> for MatMul<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    default fn compute(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T) {
        //print!("generic runner\n");
        kernel::generic::matmul_block::matmul_block(
            input_ptr1,
            input_ptr2,
            output_ptr,
            &(self.params),
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

impl MatlMulTrait<f16> for MatMul<f16> {
    fn compute(&self, input_ptr1: *const f16, input_ptr2: *const f16, output_ptr: *mut f16) {
        // print!("f16 runner\n");

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::matmul_block::matmul_block(
                input_ptr1,
                input_ptr2,
                output_ptr,
                &self.params,
            );
        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::generic::matmul_block::matmul_block(
            input_ptr1,
            input_ptr2,
            output_ptr,
            &(self.params),
        );
    }

    fn compute2(
        &self,
        input_ptr1: *const f16,
        input_ptr2: *const f16,
        output_ptr: *mut f16,
        length: usize,
    ) {
        // print!("f16 runner\n");

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::dot_product::dot_product(
                input_ptr1, input_ptr2, output_ptr, length,
            );
        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::generic::dot_product::dot_product(input_ptr1, input_ptr2, output_ptr, length);
    }
}

impl MatlMulTrait<f32> for MatMul<f32> {
    fn compute(&self, input_ptr1: *const f32, input_ptr2: *const f32, output_ptr: *mut f32) {
        // print!("f32 runner\n");

        /*//implementation for f32 on platform with avx2
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            SIMD_f32_256_matmul_block(a, b, c, param, a_row_l, b_row_l, column_l);
        };
        // generic implementation for f32
        // #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]*/
        // generic_matmul_block(input_ptr1, input_ptr2, output_ptr, &(self.params));
    }

    fn compute2(
        &self,
        input_ptr1: *const f32,
        input_ptr2: *const f32,
        output_ptr: *mut f32,
        length: usize,
    ) {
        // print!("f32 runner\n");

        /*//implementation for f32 on platform with avx2
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            SIMD_f32_256_matmul_block(a, b, c, param, a_row_l, b_row_l, column_l);
        };
        // generic implementation for f32
        // #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]*/
        kernel::generic::dot_product::dot_product(input_ptr1, input_ptr2, output_ptr, length);
    }
}

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
        let h = half::f16::from_bits(x.to_bits());
        h.to_f32()
    }

    fn all_close_f16(a: &[f16], b: &[f16], tol: f32) -> bool {
        if a.len() != b.len() {
            return false;
        }
        for (x, y) in a.iter().zip(b.iter()) {
            if (to_f32(*x) - to_f32(*y)).abs() > tol {
                return false;
            }
        }
        true
    }

    /// 集成测试：run 内部完成 B 的预转置 + 外层打包 + 3x32 微核
    /// 假设整除：M%MR=0, N%NR=0, K%KC=0
    ///
    /// A: [M×K] 行主（全 1）
    /// B_orig: [K×N] 行主（全 1）  ← 注意：这里传原始布局
    /// 期望：C = A×B = K（每个元素都等于 K）
    #[test]
    fn test_run_internal_transpose_f16_m6_n64_k64() {
        // 微核依赖 avx512fp16；没有就跳过
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("Skipping: CPU lacks avx512fp16");
            return;
        }

        unsafe {
            // ---- 维度（整除） ----
            let m = 6usize; // M = batch_size
            let n = 64usize; // N = 输出列
            let k = 64usize; // K = hidden_size

            // ---- 分块（整除） ----
            let mb = 6usize; // a_row_step_macro
            let nb = 64usize; // b_row_step_macro
            let kc = 64usize; // column_step_macro
            let mr = 3usize; // a_row_step_micro（微核行）
            let nr = 32usize; // b_row_step_micro（微核列）

            // ---- 构造数据 ----
            // A: [M×K] 行主，全 1
            let a: Vec<f16> = (0..m * k).map(|_| f16v(1.0)).collect();
            // B_orig: [K×N] 行主，全 1   ← 传入 run 的 “原始” B
            let b_orig: Vec<f16> = (0..k * n).map(|_| f16v(1.0)).collect();
            // C: [M×N] 行主，初始化为 0
            let mut c: Vec<f16> = (0..m * n).map(|_| f16v(0.0)).collect();

            // ---- 参数（为什么这样填）----
            // a_row=m, b_row=n, column=k  对应本次 GEMM 的 M/N/K
            // 宏核：MB/NB/KC；微核：MR=3/NR=32（和微核实现一致）
            let params = MatMulParams {
                a_row: m,
                b_row: n,
                column: k,
                a_row_step_macro: mb,
                b_row_step_macro: nb,
                column_step_macro: kc,
                a_row_step_micro: mr,
                b_row_step_micro: nr,
            };

            // ---- 构造算子 ----
            let op = MatMul::<f16>::new(
                a.as_ptr(),      // A[M×K]
                b_orig.as_ptr(), // B_orig[K×N]（原始布局！）
                c.as_mut_ptr(),  // C[M×N]
                false,           // output_to_kv: 本测试不用
                params.a_row,
                params.b_row,
                params.column,
                params.a_row_step_macro,
                params.b_row_step_macro,
                params.column_step_macro,
                params.a_row_step_micro,
                params.b_row_step_micro,
            );

            // ---- 执行（单线程，简化路径）----
            op.run(0, 1, m, 1, 0);

            // ---- 验证：C 每个元素都应为 K ----
            let expected: Vec<f16> = (0..m * n).map(|_| f16v(k as f32)).collect();
            assert!(
                all_close_f16(&c, &expected, 1e-3),
                "C != {}, first few = {:?}",
                k,
                c.iter().take(8).map(|x| to_f32(*x)).collect::<Vec<_>>()
            );
        }
    }

    mod integ_tests_seq {
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
            let h = half::f16::from_bits(x.to_bits());
            h.to_f32()
        }

        fn all_close_f16(a: &[f16], b: &[f16], tol: f32) -> bool {
            if a.len() != b.len() {
                return false;
            }
            for (x, y) in a.iter().zip(b.iter()) {
                if (to_f32(*x) - to_f32(*y)).abs() > tol {
                    return false;
                }
            }
            true
        }

        /// 集成测试（sequence > 1）：
        /// - run 内部会把 B[K×N] 预转置为 B_nt[N×K]，然后做三分块 + 打 KC×NR 面板 + 3x32 微核；
        /// - 假设整除：M%MR=0, N%NR=0, K%KC=0。
        ///
        /// 设置：
        ///   S_total = 4；只处理 [position_index=1, position_interval=2] → 序列 s=1 和 s=2。
        ///   M=6, N=64, K=64；MB=6, NB=64, KC=64；MR=3, NR=32。
        /// 期望：
        ///   C[s=1] 与 C[s=2] 的每个元素 == K；
        ///   C[s=0] 与 C[s=3] 仍为 0。
        #[test]
        fn test_run_internal_transpose_f16_seqlen_gt1_slice() {
            if !is_x86_feature_detected!("avx512fp16") {
                eprintln!("Skipping: CPU lacks avx512fp16");
                return;
            }

            unsafe {
                // ---- 维度（整除） ----
                let s_total = 4usize; // 总序列
                let m = 6usize; // M = batch_size
                let n = 64usize; // N = 输出列
                let k = 64usize; // K = hidden_size

                // ---- 分块（整除） ----
                let mb = 6usize; // a_row_step_macro
                let nb = 64usize; // b_row_step_macro
                let kc = 64usize; // column_step_macro
                let mr = 3usize; // a_row_step_micro（微核行）
                let nr = 32usize; // b_row_step_micro（微核列）

                // ---- 序列切片 ----
                let position_index = 1usize; // 只处理 s=1 开始
                let position_interval = 2usize; // 处理 s=1,2 两帧

                // ---- 构造数据 ----
                // A: [S×M×K] 行主，全 1
                let a_len = s_total * m * k;
                let a: Vec<f16> = (0..a_len).map(|_| f16v(1.0)).collect();

                // B_orig: [K×N] 行主，全 1（run 内部会转置成 [N×K]）
                let b_orig_len = k * n;
                let b_orig: Vec<f16> = (0..b_orig_len).map(|_| f16v(1.0)).collect();

                // C: [S×M×N] 行主，初始化为 0
                let c_len = s_total * m * n;
                let mut c: Vec<f16> = (0..c_len).map(|_| f16v(0.0)).collect();

                // ---- 参数（M/N/K 与分块、微核）----
                let params = MatMulParams {
                    a_row: m,
                    b_row: n,
                    column: k,
                    a_row_step_macro: mb,
                    b_row_step_macro: nb,
                    column_step_macro: kc,
                    a_row_step_micro: mr,
                    b_row_step_micro: nr,
                };

                // ---- 构造算子（ptr2 仍传原始 B[K×N]，run 里会转置）----
                let op = MatMul::<f16>::new(
                    a.as_ptr(),      // A[S×M×K]
                    b_orig.as_ptr(), // B_orig[K×N]（原始布局！）
                    c.as_mut_ptr(),  // C[S×M×N]
                    false,           // output_to_kv: 本测试不用
                    params.a_row,
                    params.b_row,
                    params.column,
                    params.a_row_step_macro,
                    params.b_row_step_macro,
                    params.column_step_macro,
                    params.a_row_step_micro,
                    params.b_row_step_micro,
                );

                // ---- 执行（单线程，简化验证路径）----
                op.run(position_index, position_interval, m, 1, 0);

                // ---- 验证 ----
                let a_seq_stride = m * k;
                let c_seq_stride = m * n;

                // helper：取某个序列的 C 视图切片
                let view_c_seq = |s: usize| -> &[f16] {
                    let off = s * c_seq_stride;
                    &c[off..off + c_seq_stride]
                };

                // s=0 未处理，应该全 0
                assert!(
                    view_c_seq(0).iter().all(|&x| to_f32(x) == 0.0),
                    "C[s=0] should remain zeros"
                );

                // s=1、s=2 被处理，应该全部等于 K
                let expected_block: Vec<f16> = (0..(m * n)).map(|_| f16v(k as f32)).collect();
                assert!(
                    all_close_f16(view_c_seq(1), &expected_block, 1e-3),
                    "C[s=1] != K"
                );
                assert!(
                    all_close_f16(view_c_seq(2), &expected_block, 1e-3),
                    "C[s=2] != K"
                );

                // s=3 未处理，应该全 0
                assert!(
                    view_c_seq(3).iter().all(|&x| to_f32(x) == 0.0),
                    "C[s=3] should remain zeros"
                );
            }
        }
    }
}
