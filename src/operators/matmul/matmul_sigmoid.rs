use std::f16;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

use crate::common::matmul_params::MatMulParams;
use crate::common::num_traits::Sigmoid;
use crate::common::send_sync_ptr::{ConstPtr, MutPtr};
use crate::kernel;
use crate::operators::assign::assign;
use crate::operators::traits::MatMulSigmoidTrait;

#[derive(Clone)]
pub struct MatMulSigmoid<T> {
    pub ptr1: ConstPtr<T>,
    pub ptr2: ConstPtr<T>,
    pub output_ptr: MutPtr<T>,
    pub params: MatMulParams,
    pub _marker: PhantomData<T>,
    pub m_max: usize,
    pub n_max: usize,
    pub k_max: usize,
    b_panel_pool: Box<[T]>,
    b_panel_stride_elems: usize,
    bias_ptr: Option<ConstPtr<T>>,
    use_routing_bias: bool,
    num_experts: usize,
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
        batch_size: usize,
        num_experts: usize,
        hidden_size: usize,
        _decode_only_flag: bool,
        use_routing_bias: bool,
    ) -> Self {
        let kc = params.column_step_macro.max(1);
        let nr = params.b_row_step_micro.max(1);
        let b_panel_stride_elems = kc * nr;

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let pool_len = threads * b_panel_stride_elems;
        let b_panel_pool: Vec<T> = vec![T::default(); pool_len];

        Self {
            ptr1: ConstPtr { ptr: input_ptr },
            ptr2: ConstPtr {
                ptr: gate_weight_ptr,
            },
            output_ptr: MutPtr { ptr: output_ptr },
            params,
            _marker: PhantomData,
            m_max: batch_size,
            n_max: num_experts,
            k_max: hidden_size,
            b_panel_pool: b_panel_pool.into_boxed_slice(),
            b_panel_stride_elems,
            bias_ptr: bias_ptr.map(|ptr| ConstPtr { ptr }),
            use_routing_bias,
            num_experts,
        }
    }
}

impl<T> MatMulSigmoid<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default + Sigmoid,
{
    #[inline(always)]
    pub fn panel_threads(&self) -> usize {
        if self.b_panel_stride_elems == 0 {
            0
        } else {
            self.b_panel_pool.len() / self.b_panel_stride_elems
        }
    }

    #[inline(always)]
    pub fn thread_b_panel_ptr(&self, thread_id: usize) -> *mut T {
        unsafe {
            self.b_panel_pool
                .as_ptr()
                .add(thread_id * self.b_panel_stride_elems) as *mut T
        }
    }

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

            let mb = self.params.a_row_step_macro.max(1);
            let nb = self.params.b_row_step_macro.max(1);
            let kc = self.params.column_step_macro.max(1);
            let mr = self.params.a_row_step_micro.max(1);
            let nr = self.params.b_row_step_micro.max(1);

            let m_pad = ((m_run + mr - 1) / mr) * mr;
            debug_assert!(m_pad <= self.m_max);
            debug_assert!(mb % mr == 0);
            debug_assert!(n % nr == 0);
            debug_assert!(k % kc == 0);

            let max_threads = self.panel_threads();
            debug_assert!(thread_num >= 1);
            debug_assert!(thread_id < thread_num);
            debug_assert!(thread_num <= max_threads);

            let tiles_m = (m_pad + mb - 1) / mb;
            let tiles_n = (n + nb - 1) / nb;
            let tiles = tiles_m * tiles_n;

            if let Some((tb, te)) = assign(tiles, thread_num, thread_id) {
                for t in tb..te {
                    let tm = t / tiles_n;
                    let tn = t % tiles_n;

                    let m0 = tm * mb;
                    let n0 = tn * nb;
                    let m_blk = (m_pad - m0).min(mb);
                    let n_blk = (n - n0).min(nb);

                    debug_assert!(m_blk % mr == 0 && n_blk % nr == 0);

                    <Self as MatMulSigmoidTrait<T>>::compute(
                        self, m0, n0, m_blk, n_blk, thread_id,
                    );
                }
            }
        }
    }
}

impl<T> MatMulSigmoidTrait<T> for MatMulSigmoid<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default + Sigmoid,
{
    default fn compute(&self, m0: usize, n0: usize, m_blk: usize, n_blk: usize, thread_id: usize) {
        let b_panel_ptr = self.thread_b_panel_ptr(thread_id);
        kernel::scalar::block_matmul_sigmoid::matmul_sigmoid(
            self.ptr1.ptr,
            self.ptr2.ptr,
            self.output_ptr.ptr,
            &self.params,
            self.m_max,
            self.n_max,
            self.k_max,
            self.bias_ptr.map(|ptr| ptr.ptr),
            self.use_routing_bias,
            self.num_experts,
            m0,
            n0,
            m_blk,
            n_blk,
            b_panel_ptr,
        );
    }
}

impl MatMulSigmoidTrait<f16> for MatMulSigmoid<f16> {
    fn compute(&self, m0: usize, n0: usize, m_blk: usize, n_blk: usize, thread_id: usize) {
        let b_panel_ptr = self.thread_b_panel_ptr(thread_id);
        kernel::scalar::block_matmul_sigmoid::matmul_sigmoid(
            self.ptr1.ptr,
            self.ptr2.ptr,
            self.output_ptr.ptr,
            &self.params,
            self.m_max,
            self.n_max,
            self.k_max,
            self.bias_ptr.map(|ptr| ptr.ptr),
            self.use_routing_bias,
            self.num_experts,
            m0,
            n0,
            m_blk,
            n_blk,
            b_panel_ptr,
        );
    }
}

impl MatMulSigmoidTrait<f32> for MatMulSigmoid<f32> {
    fn compute(&self, m0: usize, n0: usize, m_blk: usize, n_blk: usize, thread_id: usize) {
        let b_panel_ptr = self.thread_b_panel_ptr(thread_id);
        kernel::scalar::block_matmul_sigmoid::matmul_sigmoid(
            self.ptr1.ptr,
            self.ptr2.ptr,
            self.output_ptr.ptr,
            &self.params,
            self.m_max,
            self.n_max,
            self.k_max,
            self.bias_ptr.map(|ptr| ptr.ptr),
            self.use_routing_bias,
            self.num_experts,
            m0,
            n0,
            m_blk,
            n_blk,
            b_panel_ptr,
        );
    }
}
