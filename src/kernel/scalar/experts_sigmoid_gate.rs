use std::ops::{Add, Mul};

use crate::common::num_traits::Sigmoid;
use crate::operators::linear::MatMul;
use crate::operators::traits::MatMulTrait;

pub fn experts_sigmoid_gate<T>(
    matmul: &MatMul<T>,
    bias_ptr: Option<*const T>,
    use_routing_bias: bool,
    num_experts: usize,
    m0: usize,
    n0: usize,
    m_blk: usize,
    n_blk: usize,
    thread_id: usize,
) where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default + Sigmoid,
{
    unsafe {
        let n = matmul.n_max;
        let k = matmul.k_max;

        let kc = matmul.params.column_step_macro.max(1);
        let mr = matmul.params.a_row_step_micro.max(1);
        let nr = matmul.params.b_row_step_micro.max(1);

        debug_assert!(m_blk % mr == 0);
        debug_assert!(n_blk % nr == 0);
        debug_assert!(k % kc == 0);

        debug_assert!(m0 + m_blk <= matmul.m_max);
        debug_assert!(n0 + n_blk <= n);

        let a_base = matmul.ptr1.ptr;
        let c_base = matmul.output_ptr.ptr;
        let lda = k;
        let ldc = n;

        let b_nt_ptr = matmul.ptr2.ptr;
        let ldb_row = k;
        let b_panel_ptr = matmul.thread_b_panel_ptr(thread_id);

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
                    let src = b_nt.add(j * ldb_row + src_col);
                    *dst_row.add(lane) = *src;
                }
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

                    matmul.compute(a_tile, b_panel_ptr as *const T, c_tile);
                    apply_sigmoid_bias_tile(c_tile, bias_slice, nt, mr, nr, ldc);
                    mi += mr;
                }

                nt += nr;
            }
            k0 += kc;
        }
    }
}

unsafe fn apply_sigmoid_bias_tile<T>(
    c_tile: *mut T,
    bias_slice: Option<&[T]>,
    col_offset: usize,
    mr: usize,
    nr: usize,
    ldc: usize,
) where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default + Sigmoid,
{
    for r in 0..mr {
        let row_ptr = c_tile.add(r * ldc);
        for c in 0..nr {
            let expert_idx = col_offset + c;
            let mut value = *row_ptr.add(c);
            if let Some(bias) = bias_slice {
                value = value + bias[expert_idx];
            }
            *row_ptr.add(c) = value.sigmoid();
        }
    }
}
