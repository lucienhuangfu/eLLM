use std::ops::{Add, Mul};

use crate::common::matmul_params::MatMulParams;
use crate::common::num_traits::Sigmoid;

pub fn matmul_sigmoid<T>(
    a_ptr: *const T,
    b_nt_ptr: *const T,
    c_ptr: *mut T,
    params: &MatMulParams,
    m_max: usize,
    n_max: usize,
    k_max: usize,
    bias_ptr: Option<*const T>,
    use_routing_bias: bool,
    m0: usize,
    n0: usize,
    m_blk: usize,
    n_blk: usize,
    b_panel_ptr: *mut T,
    acc_ptr: *mut T,
)
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default + Sigmoid,
{
    unsafe {
        let n = n_max;
        let k = k_max;

        let k_step = params.kc();
        let m_step = params.mr();
        let n_step = params.nr();

        debug_assert!(m_blk % m_step == 0);
        debug_assert!(n_blk % n_step == 0);
        debug_assert!(k % k_step == 0);
        debug_assert!(m0 + m_blk <= m_max);
        debug_assert!(n0 + n_blk <= n);

        let lda = k;
        let ldc = n;
        let ldb_row = k;
        let acc_stride = n_blk;

        let bias_row = match (use_routing_bias, bias_ptr) {
            (true, Some(ptr)) => Some(std::slice::from_raw_parts(ptr.add(n0), n_blk)),
            _ => None,
        };

        for idx in 0..(m_blk * n_blk) {
            *acc_ptr.add(idx) = T::default();
        }

        let mut k0 = 0;
        while k0 < k {
            let mut n0_block = 0;
            while n0_block < n_blk {
                let packed_n0 = n0 + n0_block;
                for n_offset in 0..n_step {
                    let src_row_ptr = b_nt_ptr.add((packed_n0 + n_offset) * ldb_row + k0);
                    let dst_row_ptr = b_panel_ptr.add(n_offset * k_step);
                    for k_offset in 0..k_step {
                        *dst_row_ptr.add(k_offset) = *src_row_ptr.add(k_offset);
                    }
                }

                let mut m0_block = 0;
                while m0_block < m_blk {
                    let a_block_ptr = a_ptr.add((m0 + m0_block) * lda + k0);
                    let acc_block_ptr = acc_ptr.add(m0_block * acc_stride + n0_block);

                    for m_offset in 0..m_step {
                        let a_row_ptr = a_block_ptr.add(m_offset * lda);
                        let acc_row_ptr = acc_block_ptr.add(m_offset * acc_stride);
                        for n_offset in 0..n_step {
                            let b_row_ptr = b_panel_ptr.add(n_offset * k_step);
                            let mut sum = *acc_row_ptr.add(n_offset);
                            for k_offset in 0..k_step {
                                sum = sum + *a_row_ptr.add(k_offset) * *b_row_ptr.add(k_offset);
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

        if let Some(bias_row) = bias_row {
            for m_offset in 0..m_blk {
                let acc_row_ptr = acc_ptr.add(m_offset * acc_stride);
                let c_row_ptr = c_ptr.add((m0 + m_offset) * ldc + n0);
                for c in 0..n_blk {
                    *c_row_ptr.add(c) = (*acc_row_ptr.add(c) + bias_row[c]).sigmoid();
                }
            }
        } else {
            for m_offset in 0..m_blk {
                let acc_row_ptr = acc_ptr.add(m_offset * acc_stride);
                let c_row_ptr = c_ptr.add((m0 + m_offset) * ldc + n0);
                for c in 0..n_blk {
                    *c_row_ptr.add(c) = (*acc_row_ptr.add(c)).sigmoid();
                }
            }
        }
    }
}
