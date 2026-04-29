use std::marker::PhantomData;
use std::ops::{Add, Mul};

use crate::common::matmul_params::MatMulParams;
use crate::common::num_traits::Sigmoid;
use crate::common::send_sync_ptr::{ConstPtr, MutPtr};
use crate::kernel;
use crate::operators::assign::assign;

#[derive(Clone)]
pub struct MatMulSigmoid<T> {
    pub ptr1: ConstPtr<T>,
    pub ptr2: ConstPtr<T>,
    pub output_ptr: MutPtr<T>,
    pub params: MatMulParams,
    pub m_max: usize,
    pub n_max: usize,
    pub k_max: usize,
    pub _marker: PhantomData<T>,
    b_panel_pool: Box<[T]>,
    b_panel_stride_elems: usize,
    acc_pool: Box<[T]>,
    acc_stride_elems: usize,
    bias_ptr: Option<ConstPtr<T>>,
    use_routing_bias: bool,
}

impl<T> MatMulSigmoid<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    pub unsafe fn new(
        input_ptr: *const T,
        gate_weight_ptr: *const T,
        bias_ptr: Option<*const T>,
        output_ptr: *mut T,
        params: MatMulParams,
        m_max: usize,
        n_max: usize,
        k_max: usize,
        use_routing_bias: bool,
    ) -> Self {
        let kc = params.kc();
        let nr = params.nr();
        let mb = params.mb();
        let nb = params.nb();
        let b_panel_stride_elems = kc * nr;
        let acc_stride_elems = mb * nb;

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let pool_len = threads * b_panel_stride_elems;
        let acc_pool_len = threads * acc_stride_elems;
        let b_panel_pool: Vec<T> = vec![T::default(); pool_len];
        let acc_pool: Vec<T> = vec![T::default(); acc_pool_len];

        Self {
            ptr1: ConstPtr { ptr: input_ptr },
            ptr2: ConstPtr {
                ptr: gate_weight_ptr,
            },
            output_ptr: MutPtr { ptr: output_ptr },
            params,
            m_max,
            n_max,
            k_max,
            _marker: PhantomData,
            b_panel_pool: b_panel_pool.into_boxed_slice(),
            b_panel_stride_elems,
            acc_pool: acc_pool.into_boxed_slice(),
            acc_stride_elems,
            bias_ptr: bias_ptr.map(|ptr| ConstPtr { ptr }),
            use_routing_bias,
        }
    }
}

impl<T> MatMulSigmoid<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default + Sigmoid,
{
    pub fn run(
        &self,
        prefill_size: usize,
        _decode_size: usize,
        thread_num: usize,
        thread_id: usize,
    ) {
        unsafe {
            let m_run = prefill_size;
            let n = self.n_max;
            let k = self.k_max;
            let mb = self.params.mb();
            let nb = self.params.nb();
            let mr = self.params.mr();

            let m_pad = ((m_run + mr - 1) / mr) * mr;
            debug_assert!(m_pad <= self.m_max);
            debug_assert!(mb % mr == 0);
            debug_assert!(n % self.params.nr() == 0);
            debug_assert!(k % self.params.kc() == 0);

            let max_threads = if self.b_panel_stride_elems == 0 {
                0
            } else {
                self.b_panel_pool.len() / self.b_panel_stride_elems
            };

            debug_assert!(thread_num >= 1);
            debug_assert!(thread_id < thread_num);
            debug_assert!(thread_num <= max_threads);

            let tiles_m = (m_pad + mb - 1) / mb;
            let tiles_n = (n + nb - 1) / nb;
            let tiles = tiles_m * tiles_n;

            let b_panel_ptr = self
                .b_panel_pool
                .as_ptr()
                .add(thread_id * self.b_panel_stride_elems) as *mut T;
            let acc_ptr = self
                .acc_pool
                .as_ptr()
                .add(thread_id * self.acc_stride_elems) as *mut T;
            let bias_ptr = self.bias_ptr.map(|ptr| ptr.ptr);

            if let Some((tb, te)) = assign(tiles, thread_num, thread_id) {
                for t in tb..te {
                    let tm = t / tiles_n;
                    let tn = t % tiles_n;
                    let m0 = tm * mb;
                    let n0 = tn * nb;
                    let m_blk = (m_pad - m0).min(mb);
                    let n_blk = (n - n0).min(nb);

                    debug_assert!(m_blk % mr == 0);
                    debug_assert!(n_blk % self.params.nr() == 0);

                    kernel::scalar::block_matmul_sigmoid::matmul_sigmoid(
                        self.ptr1.ptr,
                        self.ptr2.ptr,
                        self.output_ptr.ptr,
                        &self.params,
                        self.m_max,
                        self.n_max,
                        self.k_max,
                        bias_ptr,
                        self.use_routing_bias,
                        m0,
                        n0,
                        m_blk,
                        n_blk,
                        b_panel_ptr,
                        acc_ptr,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_matmul_sigmoid_runner_f32_nt_bias() {
        const M: usize = 3;
        const K: usize = 64;
        const N: usize = 32;

        let mut a = vec![0.0f32; M * K];
        let mut b_nt = vec![0.0f32; N * K];
        let mut bias = vec![0.0f32; N];
        let mut c = vec![0.0f32; M * N];

        for i in 0..M {
            for kk in 0..K {
                a[i * K + kk] = 0.01 * (i as f32) + 0.001 * (kk as f32);
            }
        }
        for j in 0..N {
            for kk in 0..K {
                b_nt[j * K + kk] = 0.02 * (kk as f32) + 0.003 * (j as f32);
            }
            bias[j] = 0.05 * (j as f32);
        }

        let params = MatMulParams {
            a_row_step_macro: 3,
            b_row_step_macro: 32,
            column_step_macro: 64,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let runner = unsafe {
            MatMulSigmoid::<f32>::new(
                a.as_ptr(),
                b_nt.as_ptr(),
                Some(bias.as_ptr()),
                c.as_mut_ptr(),
                params,
                M,
                N,
                K,
                true,
            )
        };

        runner.run(M, 0, 1, 0);

        for i in 0..M {
            for j in 0..N {
                let mut sum = bias[j];
                for kk in 0..K {
                    sum += a[i * K + kk] * b_nt[j * K + kk];
                }
                let expected = 1.0f32 / (1.0f32 + (-sum).exp());
                let got = c[i * N + j];
                assert_abs_diff_eq!(got, expected, epsilon = 1e-5);
            }
        }
    }
}
