use std::ops::{Add, Mul};

use crate::common::matmul_params::MatMulSigmoidParams;
use crate::common::num_traits::Sigmoid;

pub fn matmul_sigmoid<T>(
    a_ptr: *const T,
    b_nt_ptr: *const T,
    c_ptr: *mut T,
    params: &MatMulSigmoidParams,
    bias_ptr: Option<*const T>,
    use_routing_bias: bool,
    m0: usize,
    n0: usize,
    m_blk: usize,
    n_blk: usize,
    b_panel_ptr: *mut T,
    acc_ptr: *mut T,
) where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default + Sigmoid,
{
    unsafe {
        let n = params.n_max;
        let k = params.k_max;

        let k_step = params.kc();
        let m_step = params.mr();
        let n_step = params.nr();

        debug_assert!(m_blk % m_step == 0);
        debug_assert!(n_blk % n_step == 0);
        debug_assert!(k % k_step == 0);
        debug_assert!(m0 + m_blk <= params.m_max);
        debug_assert!(n0 + n_blk <= n);

        let lda = k;
        let ldc = n;
        let ldb_row = k;
        let acc_stride = n_blk;

        let bias_slice = if use_routing_bias {
            bias_ptr.map(|ptr| std::slice::from_raw_parts(ptr, params.n_max))
        } else {
            None
        };

        #[inline(always)]
        unsafe fn pack_b_panel<T: Copy>(
            b_nt: *const T,
            ldb_row: usize,
            n0: usize,
            k0: usize,
            k_step: usize,
            n_step: usize,
            out: *mut T,
        ) {
            for k_offset in 0..k_step {
                let src_col = k0 + k_offset;
                let dst_row = out.add(k_offset * n_step);
                for n_offset in 0..n_step {
                    let src_col_idx = n0 + n_offset;
                    *dst_row.add(n_offset) = *b_nt.add(src_col_idx * ldb_row + src_col);
                }
            }
        }

        #[inline(always)]
        unsafe fn zero_accumulator<T: Default + Copy>(acc_ptr: *mut T, len: usize) {
            for idx in 0..len {
                *acc_ptr.add(idx) = T::default();
            }
        }

        zero_accumulator(acc_ptr, m_blk * n_blk);

        let mut k0 = 0;
        while k0 < k {
            let mut n0_block = 0;
            while n0_block < n_blk {
                pack_b_panel::<T>(
                    b_nt_ptr,
                    ldb_row,
                    n0 + n0_block,
                    k0,
                    k_step,
                    n_step,
                    b_panel_ptr,
                );

                let mut m0_block = 0;
                while m0_block < m_blk {
                    let a_block_ptr = a_ptr.add((m0 + m0_block) * lda + k0);
                    let acc_block_ptr = acc_ptr.add(m0_block * acc_stride + n0_block);

                    for m_offset in 0..m_step {
                        let a_row_ptr = a_block_ptr.add(m_offset * lda);
                        let acc_row_ptr = acc_block_ptr.add(m_offset * acc_stride);
                        for n_offset in 0..n_step {
                            let mut sum = *acc_row_ptr.add(n_offset);
                            for k_offset in 0..k_step {
                                let a_value = *a_row_ptr.add(k_offset);
                                let b_value = *b_panel_ptr.add(k_offset * n_step + n_offset);
                                sum = sum + a_value * b_value;
                            }
                            *acc_row_ptr.add(n_offset) = sum;
                        }
                    }
                    m0_block += m_step;
                }

                n0_block += n_step;
            }
            k0 += k_step;
        }

        let bias_row = bias_slice.map(|bias| &bias[n0..n0 + n_blk]);

        for m_offset in 0..m_blk {
            let acc_row_ptr = acc_ptr.add(m_offset * acc_stride);
            let c_row_ptr = c_ptr.add((m0 + m_offset) * ldc + n0);
            if let Some(bias_row) = bias_row {
                write_sigmoid_row_with_bias(c_row_ptr, acc_row_ptr, bias_row, n_blk);
            } else {
                write_sigmoid_row_without_bias(c_row_ptr, acc_row_ptr, n_blk);
            }
        }
    }
}

unsafe fn write_sigmoid_row_with_bias<T>(
    c_row: *mut T,
    acc_row: *const T,
    bias_row: &[T],
    n_blk: usize,
) where
    T: Copy + Add<Output = T> + Sigmoid,
{
    for c in 0..n_blk {
        let value = *acc_row.add(c) + bias_row[c];
        *c_row.add(c) = value.sigmoid();
    }
}

unsafe fn write_sigmoid_row_without_bias<T>(c_row: *mut T, acc_row: *const T, n_blk: usize)
where
    T: Copy + Sigmoid,
{
    for c in 0..n_blk {
        *c_row.add(c) = (*acc_row.add(c)).sigmoid();
    }
}
