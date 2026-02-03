use std::cell::RefCell;
use std::ops::{AddAssign, Neg, Sub};
use std::rc::Rc;

use crate::kernel::generic::from_f32::FromF32;
use crate::kernel::generic::sigmoid::Sigmoid;
use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{exp::Exp, neg_infinity::NegInfinity};

use super::super::memory::cache::Cache;
// use crate::compiler::mul::attention_add::AttentionAdd;
use super::super::init::matmul_params::MatMulParams;
use crate::compiler::operator::Operator;
use crate::init::record::TokenList;
// use super::super::ptensor::linear::Linear;
use super::super::ptensor::tensor::Tensor;

use super::config::Config;

// #[derive(Clone)]
pub struct Attention<T>
where
    T: Copy + PartialOrd,
{
    // sequence_length: usize,
    // batch_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_key_value_groups: usize,
    head_dim: usize,
    scaling: T,
    attention_dropout: T,
    is_causal: bool,
    layer_idx: usize,
    q_weight: Tensor<T>,
    k_weight: Tensor<T>,
    v_weight: Tensor<T>,
    o_weight: Tensor<T>,
    scope_name: String,
    cache: Rc<RefCell<Cache<T>>>,
    operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
}

impl<T> Attention<T>
where
    T: Copy
        + PartialOrd
        + Default
        + Sub<Output = T>
        + Neg<Output = T>
        + Exp
        + NegInfinity
        + Sigmoid<T>
        + Sqrt
        + FromF32
        + AddAssign,
{
    pub fn new(
        config: &Config,
        layer_idx: usize,
        parent_scope_name: &str,
        cache: Rc<RefCell<Cache<T>>>,
        operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    ) -> Self {
        let head_dim: usize = config.head_dim;
        let num_key_value_groups = config.num_attention_heads / config.num_key_value_heads;
        // T::sqrt(T::from_usize(config.attention_head_size )),
        let scaling = T::from_f32(1.0 / (head_dim as f32).sqrt());

        let scope_name = format!("{}.self_attn", parent_scope_name);
        Self {
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            num_key_value_groups: num_key_value_groups,
            head_dim: head_dim,
            scaling: scaling,
            attention_dropout: T::default(),
            is_causal: false,
            layer_idx: layer_idx,
            q_weight: Tensor::zeros(
                vec![ config.num_attention_heads * head_dim, config.hidden_size],
                format!("{}.q_proj.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            k_weight: Tensor::zeros(
                vec![config.num_key_value_heads * head_dim, config.hidden_size],
                format!("{}.k_proj.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            v_weight: Tensor::zeros(
                vec![config.num_key_value_heads * head_dim, config.hidden_size ],
                format!("{}.v_proj.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),

            o_weight: Tensor::zeros(
                vec![config.hidden_size, config.num_attention_heads * head_dim],
                format!("{}.o_proj.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            cache: cache,
            operator_queue: operator_queue,
            scope_name: scope_name,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<T>,
        residual: &Tensor<T>,
        position_embedding: &Tensor<T>,
        token_list_ptr: *const TokenList,
        decode_only_flag: bool,
        // tensor_name: String,
    ) -> Tensor<T> {
        unsafe {
            //println!("hidden_states shape: {:?}", hidden_states.shape);


            // mul_rms_complex 合并operators
            // [sequence_chunk_size, batch_size, hidden_size]
            // [sequence_chunk_size, batch_size, kv_hidden_size]
            let (query_states, key_states, value_states) = hidden_states.matmul3(
                &self.q_weight,
                &self.k_weight,
                &self.v_weight,
                position_embedding,
                self.head_dim,
                MatMulParams {
                    a_row_step_macro: 3,
                    b_row_step_macro: 64,
                    column_step_macro: 64,
                    a_row_step_micro: 3,
                    b_row_step_micro: 32,
                },
                self.scope_name.clone(),
            );
            
            let view_query_states = query_states.view(vec![
                query_states.shape[0],
                query_states.shape[1],
                self.num_attention_heads,
                self.head_dim,
            ]);

            let view_key_states = key_states.view(vec![
                key_states.shape[0],
                key_states.shape[1],
                self.num_key_value_heads,
                self.head_dim,
            ]);

            //[batch_size, head_num, sequence_num,  head_size] < - [sequence_num, batch_size, head_num, head_size]
            let mut view_key_position_tensor = view_key_states.permute(vec![1, 2, 0, 3]);

            let mut view_value_states = value_states.view(vec![
                value_states.shape[0],
                value_states.shape[1],
                self.num_key_value_heads,
                self.head_dim,
            ]);

            let mut view_value_states2 = view_value_states.permute(vec![1, 2, 0, 3]);

            // [position_window_size, batch_size, head_num, head_size] <- [position_window_size, batch_size, head_num, head_size] [batch_size, head_num, sequence_num, head_size] [batch_size, head_num, sequence_num, head_size]
            let attn_output = view_query_states.attention(
                &view_key_position_tensor,
                &view_value_states2,
                self.scaling,
                format!("{}.attn_output", self.scope_name),
            );

            println!("attn_output shape: {:?}", attn_output.shape);
            let mut view_context_tensor = attn_output.view(vec![
                attn_output.shape[0],
                attn_output.shape[1],
                attn_output.shape[2] * attn_output.shape[3]
            ]);

            // [sequence_chunk_size, batch_size, hidden_size]
            // matmul + add
            let output_tensor = view_context_tensor.matmul_add(
                &self.o_weight,
                &residual,
                MatMulParams {
                    a_row_step_macro: 3,
                    b_row_step_macro: 256,
                    column_step_macro: 16,
                    a_row_step_micro: 3,
                    b_row_step_micro: 32,
                },
                self.scope_name.clone(),
            );
            output_tensor
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::init::record::{TokenList, TokenRecord};

    #[test]
    fn test_self_attention() {
        let sequence_chunk_size = 1;
        let batch_size = 3;
        // let hidden_size = 128;
        // let num_attention_heads = 64;
        // let num_kv_heads = 8;
        // let sequence_length = 10;

        let config =
            Config::load_from_file(r"models/Qwen3-Coder-30B-A3B-Instruct/config.json").unwrap();

        // let inverse_sqrt_head = 1.0 / (config.hidden_size as f32).sqrt();
        let attention_head_size: usize = config.head_dim;
        // config.hidden_size / config.num_attention_heads;

        let cache = Rc::new(RefCell::new(Cache::new(std::collections::HashMap::<
            String,
            Vec<f32>,
        >::new())));
        let operator_queue = Rc::new(RefCell::new(Vec::new()));

        let self_attention = Attention::new(
            // hidden_size,
            // num_attention_heads,
            // num_kv_heads,
            // sequence_length,
            // batch_size,
            // inverse_sqrt_head,
            // num_cpus::get(),
            &config,
            1,
            "model.layers.1.self_attn",
            cache.clone(),
            operator_queue.clone(),
        );

        let hidden_states = Tensor::zeros(
            vec![sequence_chunk_size, batch_size, config.hidden_size],
            String::from("model.layers.1.hidden_tensor"),
            cache.clone(),
            operator_queue.clone(),
        );

        let residual_tensor = Tensor::zeros(
            vec![sequence_chunk_size, batch_size, config.hidden_size],
            String::from("model.layers.1.residual_tensor"),
            cache.clone(),
            operator_queue.clone(),
        );

        let position_embedding = Tensor::zeros(
            vec![config.max_position_embeddings, 1, 1, attention_head_size],
            String::from("model.position_embedding.weight"),
            cache.clone(),
            operator_queue.clone(),
        );

        let token_list = TokenList {
            token_records: vec![
                TokenRecord {
                    batch_index: 0,
                    position_index: 0,
                };
                batch_size
            ]
            .into_boxed_slice(),
            current_token_size: batch_size,
            lift_records: vec![].into_boxed_slice(),
            current_lift_size: 0,
        };

        let output = self_attention.forward(
            &hidden_states,
            &residual_tensor,
            &position_embedding,
            &token_list,
            false,
        );

        // Add assertions to validate the output
        debug_assert_eq!(
            output.shape,
            vec![
                sequence_chunk_size,
                batch_size,
                config.hidden_size
            ]
        );

        // Execute the operator queue
        let thread_num: usize = num_cpus::get();
        for operator in output.operator_queue.borrow().iter() {
            for i in 0..thread_num {
                operator.run(0, 1, batch_size, thread_num, i);
            }
        }

        // Add more assertions as needed
    }

    #[test]
    fn test_self_attention_f16() {
        let sequence_chunk_size = 1;
        let batch_size = 3;

        let config =
            Config::load_from_file(r"models/Qwen3-Coder-30B-A3B-Instruct/config.json").unwrap();

        let attention_head_size: usize = config.head_dim;


        let cache = Rc::new(RefCell::new(Cache::new(std::collections::HashMap::<
            String,
            Vec<f16>,
        >::new())));
        let operator_queue = Rc::new(RefCell::new(Vec::new()));

        let self_attention = Attention::new(
            &config,
            1,
            "model.layers.1.self_attn",
            cache.clone(),
            operator_queue.clone(),
        );

        let hidden_states = Tensor::zeros(
            vec![sequence_chunk_size, batch_size, config.hidden_size],
            String::from("model.layers.1.hidden_tensor"),
            cache.clone(),
            operator_queue.clone(),
        );

        let residual_tensor = Tensor::zeros(
            vec![
                sequence_chunk_size,
                batch_size,
                config.hidden_size,
            ],
            String::from("model.layers.1.residual_tensor"),
            cache.clone(),
            operator_queue.clone(),
        );

        let position_embedding = Tensor::zeros(
            vec![config.max_position_embeddings, 1, 1, attention_head_size],
            String::from("model.position_embedding.weight"),
            cache.clone(),
            operator_queue.clone(),
        );

        let output = self_attention.forward(&hidden_states, &residual_tensor, &position_embedding);

        
        // Add assertions to validate the output
        debug_assert_eq!(
            output.shape,
            vec![
                sequence_chunk_size,
                batch_size,
                config.hidden_size
            ]
        );
         

        // Execute the operator queue
        let thread_num: usize = num_cpus::get();
        for operator in output.operator_queue.borrow().iter() {
            for i in 0..thread_num {
                operator.run(0, 1, batch_size, thread_num, i);
            }
        }
    }
}
