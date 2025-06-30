use std::cell::RefCell;
use std::rc::Rc;
use std::ops::{Add, Sub, Div, Mul, AddAssign, Neg};
use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{neg_infinity::NegInfinity, exp::Exp};
use crate::kernel::generic::sigmoid::Sigmoid;

use super::super::memory::cache::Cache;
use crate::compiler::operator::Operator;
use super::super::ptensor::linear::Linear;
use super::super::ptensor::tensor::Tensor;
use crate::compiler::zip_map::complex_zip::ComplexZipMap;
use crate::compiler::mul::attention_mul::AttentionMul;
// use crate::compiler::map::softmax_map::SoftmaxMap;
// use crate::compiler::mul::vec_mul::VecMul;
// use crate::compiler::mul::col_mul::ColMul;

#[derive(Clone)]
pub struct SelfAttention<T> {

    sequence_length: usize,
    batch_size: usize,
    num_attention_heads: usize,
    num_kv_heads: usize,
    attention_head_size: usize,
    all_head_size: usize,
    hidden_size: usize,
    inverse_sqrt_head: T,
    query: Linear<T>,
    key: Linear<T>,
    value: Linear<T>,
    wo: Linear<T>,
    // freqs_cis: Tensor,
    cpu_num: usize,
    scope_name: String,
    cache: Rc<RefCell<Cache<T>>>,
    operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
}

impl<T> SelfAttention<T> 
where T: Copy 
    + Default 
    + Sub<Output = T>
    + Neg<Output = T>
    + Exp
    + NegInfinity
    + Sigmoid<T>
    + Sqrt
{

    pub fn new(
        hidden_size: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        sequence_length: usize,
        batch_size: usize,
        inverse_sqrt_head: T,
        cpu_num: usize,
        parent_scope_name: &str,
        cache: Rc<RefCell<Cache<T>>>,
        operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    ) -> Self {
        let attention_head_size: usize = hidden_size / num_attention_heads;
        let all_head_size: usize = num_attention_heads * attention_head_size;
        let kv_head_size = num_kv_heads * attention_head_size;
        // let inverse_sqrt_head = f32::sqrt(attention_head_size as f32).recip();
        let scope_name = format!("{}.self_attn", parent_scope_name);
        SelfAttention {
            num_attention_heads: num_attention_heads,
            num_kv_heads: num_kv_heads,
            attention_head_size: attention_head_size,
            all_head_size: all_head_size,
            sequence_length: sequence_length,
            query: Linear::new(
                hidden_size,
                all_head_size,
                1,
                format!("{}.q_proj", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            key: Linear::new(
                hidden_size,
                kv_head_size,
                1,
                format!("{}.k_proj", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            value: Linear::new(
                hidden_size,
                kv_head_size,
                sequence_length,
                format!("{}.v_proj", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            wo: Linear::new(
                hidden_size,
                hidden_size,
                1,
                format!("{}.o_proj", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            batch_size: batch_size,
            hidden_size: hidden_size,
            inverse_sqrt_head: inverse_sqrt_head,
            cpu_num: cpu_num,
            cache: cache,
            operator_queue: operator_queue,
            scope_name: scope_name,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<T>,
        position_embedding: &Tensor<T>,
        // tensor_name: String,
    ) -> Tensor<T> {
        unsafe {

            let value_tensor = self
            .value
            .forward(hidden_states, format!("{}.value_tensor", self.scope_name));
           
            // [batch_size, hidden_size]
            let query_tensor = self
                .query
                .forward(hidden_states, format!("{}.query_tensor", self.scope_name));

            // [batch_size, group_hidden_size]
            let key_tensor = self
                .key
                .forward(hidden_states, format!("{}.key_tensor", self.scope_name));

            let view_query = query_tensor.view(vec![
                self.batch_size,
                self.num_attention_heads,
                self.attention_head_size,
            ]);

            let view_key = key_tensor.view(vec![
                self.batch_size,
                self.num_kv_heads,
                self.attention_head_size,
            ]);

            
            // 这两个zipmap 可以合并
            // [sequence, batch_size, head_num, head_size] <- [batch_size, head_num, head_size]  [sequence, 1, 1, head_size]
            // output 为一个
            let query_position_tensor = view_query.zip_mapv(
                position_embedding,
                Operator::ComplexZip(ComplexZipMap::new(self.attention_head_size, 
                    self.num_attention_heads, 
                    self.batch_size, 
                    self.cpu_num)),
                
                true,
                format!("{}.query_position_tensor", self.scope_name),
            );
            
            let key_position_tensor = view_key.zip_mapv(
                position_embedding,
                Operator::ComplexZip(ComplexZipMap::new(
                    self.attention_head_size,
                    self.num_kv_heads,
                    self.batch_size,
                    self.cpu_num,
                )),
                
                false,
                format!("{}.key_position_tensor", self.scope_name),
            );

            
            //[batch_size, head_num, sequence_num,  head_size] < - [sequence_num, batch_size, head_num, head_size]
            let mut view_key_position_tensor = key_position_tensor.permute(vec![1, 2, 0, 3]);
            
            
            let mut view_value_tensor = value_tensor.view(vec![
                self.sequence_length,
                self.batch_size,
                self.num_kv_heads,
                self.attention_head_size
            ]);
            let mut view_value_tensor2 = view_value_tensor.permute(vec![1, 2, 0, 3]);

          
            // [batch_size, head_num, head_size] <- [batch_size, head_num, head_size] [batch_size, head_num, sequence_num, head_size] [batch_size, head_num, sequence_num, head_size] 
            let context_tensor = query_position_tensor.attention(
                &view_key_position_tensor,  
                &view_value_tensor2, 
                Operator::AttentionMul(AttentionMul::new(self.attention_head_size, self.num_attention_heads, view_key_position_tensor.strides[2], self.inverse_sqrt_head, self.cpu_num)),
                format!("{}.context_tensor", self.scope_name));

            let mut view_context_tensor = context_tensor.view(vec![
                self.batch_size,
                self.hidden_size
            ]);
            



            // [batch_size, hidden_size]
            let output_tensor = self.wo.forward(&view_context_tensor, format!("{}.output_tensor", self.scope_name));
            // let output_tensor = self.wo.forward(&hidden_states, format!("{}.output_tensor", self.scope_name));
            output_tensor
          
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn test_self_attention() {
        let hidden_size = 128;
        let num_attention_heads = 64;
        let num_kv_heads = 8;
        let sequence_length = 10;
        let batch_size = 32;
        let inverse_sqrt_head = 1.0 / (hidden_size as f32).sqrt();
        let attention_head_size: usize = hidden_size / num_attention_heads;

        let cache = Rc::new(RefCell::new(Cache::new()));
        let operator_queue = Rc::new(RefCell::new(Vec::new()));

        let self_attention = SelfAttention::new(
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            sequence_length,
            batch_size,
            inverse_sqrt_head,
            num_cpus::get(),
            "model.layer.1.self_attn",
            cache.clone(),
            operator_queue.clone(),
        );

        let hidden_states = Tensor::zeros(
            vec![batch_size, hidden_size],
            String::from("model.layer.1.hidden_tensor"),
            cache.clone(),
            operator_queue.clone(),
        );

        let position_embedding = Tensor::zeros(
            vec![sequence_length, 1, 1, attention_head_size],
            String::from("model.position_embedding.weight"),
            cache.clone(),
            operator_queue.clone(),
        );

        let output = self_attention.forward(&hidden_states, &position_embedding);

        // Add assertions to validate the output
        assert_eq!(output.shape, vec![batch_size, hidden_size]);

        // Execute the operator queue
        let thread_num: usize = num_cpus::get();
        for operator in output.operator_queue.borrow().iter() {
            for i in 0..thread_num {
                operator.run(1, 1, i);
            }
        }

        // Add more assertions as needed
    }



    #[test]
    fn test_self_attention_f16() {
        let hidden_size = 8192;
        let num_attention_heads = 64;
        let num_kv_heads = 8;
        let sequence_length = 10;
        let batch_size = 32;

        let position_index = 8;
        let current_batch_size = 1;

        let inverse_sqrt_head = 1.0 / (hidden_size as f16).sqrt();
        let attention_head_size: usize = hidden_size / num_attention_heads;

        let cache = Rc::new(RefCell::new(Cache::new()));
        let operator_queue = Rc::new(RefCell::new(Vec::new()));

        let self_attention = SelfAttention::new(
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            sequence_length,
            batch_size,
            inverse_sqrt_head,
            num_cpus::get(),
            "model.layer.1.self_attn",
            cache.clone(),
            operator_queue.clone(),
        );

        let hidden_states = Tensor::zeros(
            vec![batch_size, hidden_size],
            String::from("model.layer.1.hidden_tensor"),
            cache.clone(),
            operator_queue.clone(),
        );

        let position_embedding = Tensor::zeros(
            vec![sequence_length, 1, 1, attention_head_size],
            String::from("model.position_embedding.weight"),
            cache.clone(),
            operator_queue.clone(),
        );

        let output = self_attention.forward(&hidden_states, &position_embedding);

        // Add assertions to validate the output
        // assert_eq!(output.shape, vec![batch_size, hidden_size]);

        // Execute the operator queue
        let thread_num: usize = num_cpus::get();
        for operator in output.operator_queue.borrow().iter() {
            for i in 0..thread_num {
                operator.run(current_batch_size, position_index, i);
            }
        }
        // Add more assertions as needed
    }
}
