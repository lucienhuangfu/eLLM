// === operators/matmul/matmul_add.rs ===
#![allow(non_snake_case)]
#![allow(unused_variables)] // run 参数里保留 position_* 但不使用

use std::f16;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

use crate::common::{
    matmul_params::MatMulParams,
    send_sync_ptr::{ConstPtr, MutPtr},
};
use crate::kernel;
use crate::operators::assign::assign;
use crate::operators::traits::MatMulAddTrait;

#[derive(Clone)]
pub struct MatMulAdd<T> {
    pub ptr1: ConstPtr<T>,     // A[M×K]
    pub ptr2: ConstPtr<T>,     // ✅ B_nt[N×K] row-major, stride = K
    pub ptr3: ConstPtr<T>,     // residual[M×N]
    pub output_ptr: MutPtr<T>, // C[M×N]

    pub decode_only_flag: bool,

    /// 仅承载 step 形状（MB/NB/KC/MR/NR）
    pub params: MatMulParams,
    pub _marker: PhantomData<T>,

    // 最大维度（与 MatMul 保持一致）
    pub m_max: usize,
    pub n_max: usize,
    pub k_max: usize,

    // 线程私有 KC×NR 面板池（连续大块，按线程切片）
    b_panel_pool: Box<[T]>,
    b_panel_stride_elems: usize, // = kc * nr
    cpu_max_for_scratch: usize,
}

impl<T> MatMulAdd<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    /// 构造函数：
    /// - 不再转置 B
    /// - ptr2 直接指向 B_nt[N×K]
    /// - 只预分配每线程 KC×NR panel scratch
    pub unsafe fn new(
        ptr1: *const T,          // A[M×K]
        ptr2_b_nt_nxk: *const T, // ✅ B_nt[N×K] row-major
        ptr3_residual: *const T, // residual[M×N]
        output_ptr: *mut T,      // C[M×N]
        params: MatMulParams,    // step 形状：MB/NB/KC/MR/NR
        m_max: usize,
        n_max: usize,
        k_max: usize,
    ) -> Self {
        let kc = params.column_step_macro.max(1);
        let nr = params.b_row_step_micro.max(1);
        let b_panel_stride_elems = kc * nr;

        let cpu_max_for_scratch = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let pool_len = cpu_max_for_scratch * b_panel_stride_elems;
        let b_panel_pool: Vec<T> = vec![T::default(); pool_len];

        Self {
            ptr1: ConstPtr { ptr: ptr1 },
            ptr2: ConstPtr {
                ptr: ptr2_b_nt_nxk,
            },
            ptr3: ConstPtr { ptr: ptr3_residual },
            output_ptr: MutPtr { ptr: output_ptr },
            decode_only_flag: false,
            params,
            _marker: PhantomData,
            m_max,
            n_max,
            k_max,
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

    /// 执行：先把 residual 覆盖到 output，然后做 output += A×B
    ///
    /// run 声明保持不变（与 Operator 统一），但 position_index/interval 不使用
    pub fn run(&self, prefill_size: usize, _decode_size: usize, thread_num: usize, thread_id: usize) {
    unsafe {
        // ===== 维度 =====
        let m_run = prefill_size;     // 真实 M
        let n = self.n_max;         // N
        let k = self.k_max;         // K

        // ===== 分块参数 =====
        let mb = self.params.a_row_step_macro.max(1);
        let nb = self.params.b_row_step_macro.max(1);
        let kc = self.params.column_step_macro.max(1);
        let mr = self.params.a_row_step_micro.max(1);
        let nr = self.params.b_row_step_micro.max(1);

        // === 关键：M 向上 pad 到 MR 的倍数（空算）===
        let m_pad = ((m_run + mr - 1) / mr) * mr;
        debug_assert!(m_pad <= self.m_max);
        let m = m_pad;

        // 保持你的对齐假设
        debug_assert!(mb % mr == 0);
        debug_assert!(n % nr == 0);
        debug_assert!(k % kc == 0);
        debug_assert!(thread_num <= self.cpu_max_for_scratch);
        debug_assert!(thread_id < thread_num);

        // ===== 基址与 stride（元素计）=====
        let a_base = self.ptr1.ptr;        // A[M×K]
        let r_base = self.ptr3.ptr;        // residual[M×N]
        let c_base = self.output_ptr.ptr;  // C[M×N]
        let b_nt = self.ptr2.ptr;          // B_nt[N×K]

        let lda = k;
        let ldc = n;
        let ldb_row = k;

        let b_panel_ptr = self.thread_b_panel_ptr(thread_id);

        #[inline(always)]
        unsafe fn pack_b_panel_from_nt<T: Copy>(
            b_nt: *const T,
            ldb_row: usize,
            n0: usize,
            k0: usize,
            kc: usize,
            nr: usize,
            out: *mut T,
        ) {
            for p in 0..kc {
                let src_col = k0 + p;
                let dst_row = out.add(p * nr);
                for lane in 0..nr {
                    let j = n0 + lane;
                    *dst_row.add(lane) = *b_nt.add(j * ldb_row + src_col);
                }
            }
        }

        // ===== tiles（按 m_pad 切分）=====
        let tiles_m = (m + mb - 1) / mb;
        let tiles_n = (n + nb - 1) / nb;
        let tiles = tiles_m * tiles_n;

        if let Some((tb, te)) = assign(tiles, thread_num, thread_id) {
            for t in tb..te {
                let tm = t / tiles_n;
                let tn = t % tiles_n;

                let m0 = tm * mb;
                let n0 = tn * nb;

                let m_blk = (m - m0).min(mb); // 这里 m 是 pad 后
                let n_blk = (n - n0).min(nb);
                debug_assert!(m_blk % mr == 0 && n_blk % nr == 0);

                // === (1) C = residual（tile 覆盖写）===
                let mut nt = 0;
                while nt < n_blk {
                    let mut mi = 0;
                    while mi < m_blk {
                        let r_tile = r_base.add((m0 + mi) * ldc + (n0 + nt));
                        let c_tile = c_base.add((m0 + mi) * ldc + (n0 + nt));

                        for r in 0..mr {
                            let rs = r_tile.add(r * ldc);
                            let cs = c_tile.add(r * ldc);
                            for cc in 0..nr {
                                *cs.add(cc) = *rs.add(cc);
                            }
                        }

                        mi += mr;
                    }
                    nt += nr;
                }

                // === (2) C += A×B ===
                let mut k0 = 0;
                while k0 < k {
                    let mut nt = 0;
                    while nt < n_blk {
                        pack_b_panel_from_nt::<T>(b_nt, ldb_row, n0 + nt, k0, kc, nr, b_panel_ptr);

                        let mut mi = 0;
                        while mi < m_blk {
                            let a_tile = a_base.add((m0 + mi) * lda + k0);
                            let c_tile = c_base.add((m0 + mi) * ldc + (n0 + nt));

                            self.compute(a_tile, b_panel_ptr as *const T, std::ptr::null(), c_tile);

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
    default fn compute(
        &self,
        input_ptr1: *const T,
        input_ptr2: *const T,
        _input_ptr3: *const T,
        output_ptr: *mut T,
    ) {
        let call_param = MatMulParams {
            a_row_step_macro: self.k_max,                     // lda = K
            b_row_step_macro: self.n_max,                     // ldc = N
            column_step_macro: self.params.column_step_macro, // kc
            a_row_step_micro: self.params.a_row_step_micro,   // mr
            b_row_step_micro: self.params.b_row_step_micro,   // nr
        };
        kernel::scalar::matmul_block::matmul_block(
            input_ptr1,
            input_ptr2,
            output_ptr,
            &call_param,
        );
    }
}

// —— f16 特化：AVX-512 FP16 累加微核（你贴的那个就是 loadC+fmadd）
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
            a_row_step_micro: self.params.a_row_step_micro,   // mr
            b_row_step_micro: self.params.b_row_step_micro,   // nr
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
        kernel::scalar::matmul_block::matmul_block(
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
    fn test_matmul_add_runner_f16_nt_6x64x32() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            // 这只是 runner test；不强制要求 avx512 才能跑也行
        }

        const M: usize = 6;
        const K: usize = 64;
        const N: usize = 32;

        let thread_num = 4;

        let mut a = vec![0.0f16; M * K];
        let mut b_nt = vec![0.0f16; N * K];
        let mut residual = vec![0.0f16; M * N];
        let mut c = vec![0.0f16; M * N];

        for i in 0..M {
            for kk in 0..K {
                a[i * K + kk] = (0.01f32 * (i as f32) + 0.001f32 * (kk as f32)) as f16;
            }
        }
        for j in 0..N {
            for kk in 0..K {
                b_nt[j * K + kk] = (0.02f32 * (kk as f32) + 0.003f32 * (j as f32)) as f16;
            }
        }
        for i in 0..M {
            for j in 0..N {
                residual[i * N + j] = (0.05f32 * (i as f32) + 0.0007f32 * (j as f32)) as f16;
            }
        }

        let params = MatMulParams {
            a_row_step_macro: M,  // MB
            b_row_step_macro: N,  // NB
            column_step_macro: K, // KC
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let runner = unsafe {
            MatMulAdd::<f16>::new(
                a.as_ptr(),
                b_nt.as_ptr(), // ✅ NT
                residual.as_ptr(),
                c.as_mut_ptr(),
                params,
                M,
                N,
                K,
            )
        };

        for tid in 0..thread_num {
            runner.run(M, 0, thread_num, tid);
        }

        for i in 0..M {
            for j in 0..N {
                let mut sum = residual[i * N + j] as f32;
                for kk in 0..K {
                    sum += (a[i * K + kk] as f32) * (b_nt[j * K + kk] as f32);
                }
                let got = c[i * N + j] as f32;
                assert_abs_diff_eq!(got, sum, epsilon = 2e-1);
            }
        }
    }
    #[test]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
fn test_matmul_add_runner_f16_nt_batch7_pad_to9() {
    const M_RUN: usize = 7;
    const M_MAX: usize = 9; // ceil_div(7,3)*3
    const K: usize = 64;
    const N: usize = 32;

    let thread_num = 4;

    let mut a = vec![0.0f16; M_MAX * K];
    let mut b_nt = vec![0.0f16; N * K];
    let mut residual = vec![0.0f16; M_MAX * N];
    let mut c = vec![0.0f16; M_MAX * N];

    // A：前 7 行填值，pad 行保持 0
    for i in 0..M_RUN {
        for kk in 0..K {
            a[i * K + kk] = (0.01f32 * (i as f32) + 0.001f32 * (kk as f32)) as f16;
        }
    }

    // B_nt：N×K
    for j in 0..N {
        for kk in 0..K {
            b_nt[j * K + kk] = (0.02f32 * (kk as f32) + 0.003f32 * (j as f32)) as f16;
        }
    }

    // residual：前 7 行填值，pad 行也给 0（避免 pad 行影响计时以外的东西）
    for i in 0..M_RUN {
        for j in 0..N {
            residual[i * N + j] = (0.05f32 * (i as f32) + 0.0007f32 * (j as f32)) as f16;
        }
    }
    // c 初始随便（会被 residual 覆盖），但我们也给 0
    for x in &mut c {
        *x = 0.0f32 as f16;
    }

    let params = MatMulParams {
        a_row_step_macro: 6,   // MB（3 的倍数）
        b_row_step_macro: 32,  // NB
        column_step_macro: 64, // KC
        a_row_step_micro: 3,   // MR
        b_row_step_micro: 32,  // NR
    };

    let runner = unsafe {
        MatMulAdd::<f16>::new(
            a.as_ptr(),
            b_nt.as_ptr(),
            residual.as_ptr(),
            c.as_mut_ptr(),
            params,
            M_MAX, // m_max
            N,
            K,
        )
    };

    let used = thread_num.min(runner.cpu_max_for_scratch);
    for tid in 0..used {
        runner.run(M_RUN, 0, used, tid); // batch=7（非 3 倍数）
    }

    // 只验证前 7 行
    for i in 0..M_RUN {
        for j in 0..N {
            let mut sum = residual[i * N + j] as f32;
            for kk in 0..K {
                sum += (a[i * K + kk] as f32) * (b_nt[j * K + kk] as f32);
            }
            let got = c[i * N + j] as f32;
            assert_abs_diff_eq!(got, sum, epsilon = 2e-1);
        }
    }
}
}



