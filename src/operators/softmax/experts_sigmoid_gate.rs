use std::f16;
use std::ops::{Add, Mul};

use crate::common::matmul_params::MatMulParams;
use crate::common::num_traits::Sigmoid;
use crate::common::send_sync_ptr::ConstPtr;
use crate::operators::assign::assign;
use crate::operators::linear::MatMul;
use crate::kernel;
use crate::operators::traits::ExpertsSigmoidGateTrait;

#[derive(Clone)]
pub struct ExpertsSigmoidGate<T> {
    matmul: MatMul<T>,
    bias_ptr: Option<ConstPtr<T>>,
    use_routing_bias: bool,
    num_experts: usize,
}

impl<T> ExpertsSigmoidGate<T>
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
        decode_only_flag: bool,
        use_routing_bias: bool,
    ) -> Self {
        let matmul = MatMul::new(
            input_ptr,
            gate_weight_ptr,
            output_ptr,
            false,
            params,
            batch_size,
            num_experts,
            hidden_size,
            decode_only_flag,
        );

        Self {
            matmul,
            bias_ptr: bias_ptr.map(|ptr| ConstPtr { ptr }),
            use_routing_bias,
            num_experts,
        }
    }
}

impl<T> ExpertsSigmoidGate<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default + Sigmoid,
{
    pub fn run(&self, prefill_size: usize, _decode_size: usize, thread_num: usize, thread_id: usize) {
        unsafe {
            let m_run = prefill_size;
            let n = self.matmul.n_max;
            let k = self.matmul.k_max;

            let mb = self.matmul.params.a_row_step_macro.max(1);
            let nb = self.matmul.params.b_row_step_macro.max(1);
            let kc = self.matmul.params.column_step_macro.max(1);
            let mr = self.matmul.params.a_row_step_micro.max(1);
            let nr = self.matmul.params.b_row_step_micro.max(1);

            let m_pad = ((m_run + mr - 1) / mr) * mr;
            debug_assert!(m_pad <= self.matmul.m_max);
            debug_assert!(mb % mr == 0);
            debug_assert!(n % nr == 0);
            debug_assert!(k % kc == 0);

            let max_threads = self.matmul.panel_threads();
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

                    self.compute(m0, n0, m_blk, n_blk, thread_id);
                }
            }
        }
    }
}

impl<T> ExpertsSigmoidGateTrait<T> for ExpertsSigmoidGate<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default + Sigmoid,
{
    default fn compute(&self, m0: usize, n0: usize, m_blk: usize, n_blk: usize, thread_id: usize) {
        kernel::scalar::experts_sigmoid_gate::experts_sigmoid_gate(
            &self.matmul,
            self.bias_ptr.map(|ptr| ptr.ptr),
            self.use_routing_bias,
            self.num_experts,
            m0,
            n0,
            m_blk,
            n_blk,
            thread_id,
        );
    }
}

impl ExpertsSigmoidGateTrait<f16> for ExpertsSigmoidGate<f16> {
    fn compute(&self, m0: usize, n0: usize, m_blk: usize, n_blk: usize, thread_id: usize) {
        kernel::scalar::experts_sigmoid_gate::experts_sigmoid_gate(
            &self.matmul,
            self.bias_ptr.map(|ptr| ptr.ptr),
            self.use_routing_bias,
            self.num_experts,
            m0,
            n0,
            m_blk,
            n_blk,
            thread_id,
        );
    }
}

impl ExpertsSigmoidGateTrait<f32> for ExpertsSigmoidGate<f32> {
    fn compute(&self, m0: usize, n0: usize, m_blk: usize, n_blk: usize, thread_id: usize) {
        kernel::scalar::experts_sigmoid_gate::experts_sigmoid_gate(
            &self.matmul,
            self.bias_ptr.map(|ptr| ptr.ptr),
            self.use_routing_bias,
            self.num_experts,
            m0,
            n0,
            m_blk,
            n_blk,
            thread_id,
        );
    }
}
