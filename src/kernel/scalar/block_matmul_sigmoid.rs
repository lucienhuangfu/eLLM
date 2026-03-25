use std::ops::{Add, Mul};

use crate::common::matmul_params::MatMulParams;
use crate::common::num_traits::Sigmoid;

pub fn matmul_sigmoid<T>(
    a_base: *const T,
    b_nt_ptr: *const T,
    c_base: *mut T,
    params: &MatMulParams,
    m_max: usize,
    n_max: usize,
    k_max: usize,
    bias_ptr: Option<*const T>,
    use_routing_bias: bool,
    num_experts: usize,
    m0: usize,
    n0: usize,
    m_blk: usize,
    n_blk: usize,
    b_panel_ptr: *mut T,
) where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default + Sigmoid,
{
    unsafe {
        let n = n_max;
        let k = k_max;

        let kc = params.column_step_macro.max(1);
        let mr = params.a_row_step_micro.max(1);
        let nr = params.b_row_step_micro.max(1);

        debug_assert!(m_blk % mr == 0);
        debug_assert!(n_blk % nr == 0);
        debug_assert!(k % kc == 0);
        debug_assert!(m0 + m_blk <= m_max);
        debug_assert!(n0 + n_blk <= n);

        let lda = k;
        let ldc = n;
        let ldb_row = k;

        let bias_slice = if use_routing_bias {
            bias_ptr.map(|ptr| std::slice::from_raw_parts(ptr, num_experts))
        } else {
            None
        };

        #[inline(always)]
        unsafe fn pack_b_panel<T: Copy>(
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

        for mi in 0..m_blk {
            let c_row = c_base.add((m0 + mi) * ldc + n0);
            for c in 0..n_blk {
                *c_row.add(c) = T::default();
            }
        }

        let mut k0 = 0;
        while k0 < k {
            let mut nt = 0;
            while nt < n_blk {
                pack_b_panel::<T>(b_nt_ptr, ldb_row, n0 + nt, k0, kc, nr, b_panel_ptr);

                let mut mi = 0;
                while mi < m_blk {
                    let a_tile = a_base.add((m0 + mi) * lda + k0);
                    let c_tile = c_base.add((m0 + mi) * ldc + (n0 + nt));

                    for i in 0..mr {
                        for j in 0..nr {
                            let mut acc = *c_tile.add(i * ldc + j);
                            for kk in 0..kc {
                                let a_value = *a_tile.add(i * lda + kk);
                                let b_value = *b_panel_ptr.add(kk * nr + j);
                                acc = acc + a_value * b_value;
                            }
                            *c_tile.add(i * ldc + j) = acc;
                        }
                    }
                    mi += mr;
                }

                nt += nr;
            }
            k0 += kc;
        }

        for mi in 0..m_blk {
            let c_row = c_base.add((m0 + mi) * ldc + n0);
            apply_sigmoid_bias_row(c_row, bias_slice, n0, n_blk);
        }
    }
}

unsafe fn apply_sigmoid_bias_row<T>(
    c_row: *mut T,
    bias_slice: Option<&[T]>,
    col_offset: usize,
    n_blk: usize,
) where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default + Sigmoid,
{
    for c in 0..n_blk {
        let mut value = *c_row.add(c);
        if let Some(bias) = bias_slice {
            value = value + bias[col_offset + c];
        }
        *c_row.add(c) = value.sigmoid();
    }
}
