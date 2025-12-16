// === runner/matmul_add.rs ===
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
use super::mul_trait::MatMulAddTrait;

#[derive(Clone)]
pub struct MatMulAdd<T> {
    pub ptr1: ConstPtr<T>,     // A[S×M×K]
    pub ptr2: ConstPtr<T>,     // 指向构造期转置后的 B_nt[N×K]
    pub ptr3: ConstPtr<T>,     // residual[S×M×N]
    pub output_ptr: MutPtr<T>, // C[S×M×N]

    /// 仅承载 step 形状（MB/NB/KC/MR/NR）
    pub params: MatMulParams,
    pub _marker: PhantomData<T>,

    // “最大维度” M/N/K（与 MatMul 保持一致）
    pub m_max: usize,
    pub n_max: usize,
    pub k_max: usize,

    // 构造期转置得到的 B_nt（N×K，行主；行距=K）
    b_nt_buf: Box<[T]>,

    // 线程私有 KC×NR 面板池（连续大块，按线程切片）
    b_panel_pool: Box<[T]>,
    b_panel_stride_elems: usize, // = kc * nr
    cpu_max_for_scratch: usize,
}

impl<T> MatMulAdd<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    /// 构造函数：一次性完成
    /// 1) B[K×N] → B_nt[N×K] 全量转置，并让 `ptr2` 指向 B_nt
    /// 2) 为每个线程预分配 KC×NR 面板（运行期不再分配）
    pub unsafe fn new(
        ptr1: *const T,          // A
        ptr2_b_kxn: *const T,    // 原始 B[K×N]
        ptr3_residual: *const T, // residual
        output_ptr: *mut T,      // C
        params: MatMulParams,    // 仅 step 形状：MB/NB/KC/MR/NR
        m_max: usize,
        n_max: usize,
        k_max: usize,
        cpu_max_for_scratch: usize, // 运行期线程上限（保证不再分配）
    ) -> Self {
        // === (1) B → B_nt 转置 ===
        // 原 B：K×N（行主，行距=N）
        // 目标：N×K（行主，行距=K）
        let mut b_nt_vec: Vec<T> = vec![T::default(); n_max * k_max];
        let b_nt_ptr = b_nt_vec.as_mut_ptr();
        for kk in 0..k_max {
            let b_row = ptr2_b_kxn.add(kk * n_max); // B[kk, :]
            for jj in 0..n_max {
                *b_nt_ptr.add(jj * k_max + kk) = *b_row.add(jj); // B_nt[jj, kk] = B[kk, jj]
            }
        }
        let b_nt_box = b_nt_vec.into_boxed_slice();
        let b_nt_base = b_nt_box.as_ptr();

        // === (2) 预分配线程面板池 ===
        let kc = params.column_step_macro.max(1);
        let nr = params.b_row_step_micro.max(1);
        let b_panel_stride_elems = kc * nr;
        let pool_len = cpu_max_for_scratch * b_panel_stride_elems;
        let b_panel_pool: Vec<T> = vec![T::default(); pool_len];

        Self {
            ptr1: ConstPtr { ptr: ptr1 },
            ptr2: ConstPtr { ptr: b_nt_base }, // 指向 B_nt
            ptr3: ConstPtr { ptr: ptr3_residual },
            output_ptr: MutPtr { ptr: output_ptr },
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

    /// 行距=ldc 的 3×32 tile 拷贝（f16 走 SIMD，其他类型走标量）
    #[inline(always)]
    unsafe fn tile_copy_3x32(&self, src: *const T, dst: *mut T, ldc: usize) {
        // f16 + AVX-512：调用你已有的内核
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        if std::mem::size_of::<T>() == std::mem::size_of::<f16>() {
            let src_f16 = src as *const f16;
            let dst_f16 = dst as *mut f16;
            super::super::super::kernel::x86_64::f16_512::moe_merge::tile_copy_3x32(
                src_f16, dst_f16, ldc,
            );
            return;
        }

        // 通用标量实现（保持风格简洁；不额外分配）
        for r in 0..3 {
            let srow = src.add(r * ldc);
            let drow = dst.add(r * ldc);
            for c in 0..32 {
                *drow.add(c) = *srow.add(c);
            }
        }
    }

    /// 执行：先把 residual 拷到 output，然后做 output += A×B
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
            let m = batch_size; // 本次 M
            let n = self.n_max; // N
            let k = self.k_max; // K

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
            let a_base = self.ptr1.ptr; // A[S×M×K]
            let r_base = self.ptr3.ptr; // residual[S×M×N]
            let c_base = self.output_ptr.ptr; // C[S×M×N]
            let lda = k; // A 每行跨度
            let ldc = n; // C 每行跨度

            // ===== 序列范围 =====
            let s_begin = position_index;
            let s_end = position_index + position_interval;
            let s_len = s_end - s_begin;
            let a_seq_stride = m * k;
            let rc_seq_stride = m * n;

            // ===== 使用构造期转置的 B_nt（N×K，行主；行距=K）=====
            let b_nt_ptr = self.ptr2.ptr; // 已是 B_nt
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
                out: *mut T, // 输出：KC×NR 行主（长度 = kc*nr）
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
                    let rem = t % (tiles_m * tiles_n);
                    let tm = rem / tiles_n;
                    let tn = rem % tiles_n;

                    let s = s_begin + s_rel;
                    let m0 = tm * mb;
                    let n0 = tn * nb;

                    let m_blk = (m - m0).min(mb);
                    let n_blk = (n - n0).min(nb);
                    debug_assert!(m_blk % mr == 0 && n_blk % nr == 0);

                    let a_base_s = a_base.add(s * a_seq_stride);
                    let r_base_s = r_base.add(s * rc_seq_stride);
                    let c_base_s = c_base.add(s * rc_seq_stride);

                    // —— 每个 (m_blk × n_blk) tile：先做 residual→output 的初始化（按 3×32 子 tile）
                    let mut nt = 0;
                    while nt < n_blk {
                        let mut mi = 0;
                        while mi < m_blk {
                            let r_tile = r_base_s.add((m0 + mi) * ldc + (n0 + nt));
                            let c_tile = c_base_s.add((m0 + mi) * ldc + (n0 + nt));
                            // residual 拷贝到 output（3×32）
                            self.tile_copy_3x32(r_tile, c_tile, ldc);
                            mi += mr;
                        }
                        nt += nr;
                    }

                    // —— 紧接着按 Kc 做 output += A×B 的归约累加
                    let mut k0 = 0;
                    while k0 < k {
                        let mut nt = 0;
                        while nt < n_blk {
                            pack_b_panel::<T>(b_nt_ptr, ldb_row, n0 + nt, k0, kc, nr, b_panel_ptr);

                            let mut mi = 0;
                            while mi < m_blk {
                                let a_tile = a_base_s.add((m0 + mi) * lda + k0);
                                let c_tile = c_base_s.add((m0 + mi) * ldc + (n0 + nt));
                                // 注意：compute() 必须是 “累加到 C ” 的微核
                                self.compute(
                                    a_tile,
                                    b_panel_ptr as *const T,
                                    std::ptr::null(),
                                    c_tile,
                                );
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

/* ------------------ compute：保持你的 trait 风格 ------------------ */

impl<T> MatMulAddTrait<T> for MatMulAdd<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    // generic：调用通用微核（要求其为 “累加到 C ” 的语义）
    default fn compute(
        &self,
        input_ptr1: *const T,
        input_ptr2: *const T,
        _input_ptr3: *const T, // 本算子用不到（残差拷贝在 run 里做了）
        output_ptr: *mut T,
    ) {
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
}

// —— f16 特化：走 AVX-512（若可用），否则 generic
impl MatMulAddTrait<f16> for MatMulAdd<f16> {
    fn compute(
        &self,
        input_ptr1: *const f16,
        input_ptr2: *const f16,
        _input_ptr3: *const f16,
        output_ptr: *mut f16,
    ) {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_matmul_add_runner_f16_3x64x32() {
        const S: usize = 1;
        const M: usize = 3;
        const K: usize = 64;
        const N: usize = 32;

        let thread_num = 4;

        let mut a = vec![0.0f16; S * M * K];
        let mut b = vec![0.0f16; K * N];
        let mut residual = vec![0.0f16; S * M * N];
        let mut c = vec![0.0f16; S * M * N];

        // Init A
        for s in 0..S {
            for i in 0..M {
                for k in 0..K {
                    let val = (s + i + k) as f32 * 0.01;
                    a[s * M * K + i * K + k] = val as f16;
                }
            }
        }

        // Init B
        for k in 0..K {
            for j in 0..N {
                let val = (k + j) as f32 * 0.02;
                b[k * N + j] = val as f16;
            }
        }

        // Init Residual
        for s in 0..S {
            for i in 0..M {
                for j in 0..N {
                    let val = (s + i + j) as f32 * 0.03;
                    residual[s * M * N + i * N + j] = val as f16;
                }
            }
        }

        let params = MatMulParams {
            a_row_step_macro: 3,
            b_row_step_macro: 32,
            column_step_macro: 64,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let matmul_add = unsafe {
            MatMulAdd::<f16>::new(
                a.as_ptr(),
                b.as_ptr(),
                residual.as_ptr(),
                c.as_mut_ptr(),
                params,
                M,
                N,
                K,
                thread_num,
            )
        };

        for i in 0..thread_num {
            matmul_add.run(0, S, M, thread_num, i);
        }

        // Verify
        for s in 0..S {
            for i in 0..M {
                for j in 0..N {
                    let mut sum = 0.0f32;
                    // A * B
                    for k in 0..K {
                        let a_val = a[s * M * K + i * K + k] as f32;
                        let b_val = b[k * N + j] as f32;
                        sum += a_val * b_val;
                    }
                    // + Residual
                    sum += residual[s * M * N + i * N + j] as f32;

                    let got = c[s * M * N + i * N + j] as f32;
                    assert_abs_diff_eq!(got, sum, epsilon = 1e-1);
                }
            }
        }
    }

    #[test]
    fn test_matmul_add_runner_f16_144x2048x2048() {
        const S: usize = 1;
        const M: usize = 144;
        const K: usize = 2048;
        const N: usize = 2048;

        let thread_num = 8;

        let mut a = vec![0.0f16; S * M * K];
        let mut b = vec![0.0f16; K * N];
        let mut residual = vec![0.0f16; S * M * N];
        let mut c = vec![0.0f16; S * M * N];

        // Init A
        for s in 0..S {
            for i in 0..M {
                for k in 0..K {
                    let val = ((s + i + k) % 7) as f32 * 0.01;
                    a[s * M * K + i * K + k] = val as f16;
                }
            }
        }

        // Init B
        for k in 0..K {
            for j in 0..N {
                let val = ((k + j) % 11) as f32 * 0.01;
                b[k * N + j] = val as f16;
            }
        }

        // Init Residual
        for s in 0..S {
            for i in 0..M {
                for j in 0..N {
                    let val = ((s + i + j) % 13) as f32 * 0.01;
                    residual[s * M * N + i * N + j] = val as f16;
                }
            }
        }

        let params = MatMulParams {
            a_row_step_macro: 24,
            b_row_step_macro: 128,
            column_step_macro: 64,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let matmul_add = unsafe {
            MatMulAdd::<f16>::new(
                a.as_ptr(),
                b.as_ptr(),
                residual.as_ptr(),
                c.as_mut_ptr(),
                params,
                M,
                N,
                K,
                thread_num,
            )
        };

        for i in 0..thread_num {
            matmul_add.run(0, S, M, thread_num, i);
        }

        // Verify
        for s in 0..S {
            for i in 0..M {
                for j in 0..N {
                    let mut sum = 0.0f32;
                    // A * B
                    for k in 0..K {
                        let a_val = a[s * M * K + i * K + k] as f32;
                        let b_val = b[k * N + j] as f32;
                        sum += a_val * b_val;
                    }
                    // + Residual
                    sum += residual[s * M * N + i * N + j] as f32;

                    let got = c[s * M * N + i * N + j] as f32;
                    assert_abs_diff_eq!(got, sum, epsilon = 5e-1);
                }
            }
        }
    }
}
