use std::ops::{AddAssign, Neg, Sub};
use std::rc::Rc;

use crate::common::num_traits::FromNumber;
use crate::common::num_traits::Sigmoid;
use crate::common::num_traits::Sqrt;
use crate::common::num_traits::{exp::Exp, neg_infinity::NegInfinity};

use super::super::common::matmul_params::MatMulParams;
use super::super::runtime::tensor::{Tensor, TensorCtx};

use super::config::Config;
use super::names::AttentionTensorNames;

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
    ctx: Rc<TensorCtx<T>>,
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
        + Sigmoid
        + Sqrt
        + FromNumber
        + AddAssign,
{
    pub fn new(
        config: &Config,
        names: AttentionTensorNames,
        ctx: Rc<TensorCtx<T>>,
    ) -> Self {
        let head_dim: usize = config.head_dim;
        let num_key_value_groups = config.num_attention_heads / config.num_key_value_heads;
        let scaling = T::from_f32(1.0 / (head_dim as f32).sqrt());

        Self {
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            num_key_value_groups: num_key_value_groups,
            head_dim: head_dim,
            scaling: scaling,
            attention_dropout: T::default(),
            is_causal: false,
            layer_idx: 0,
            q_weight: ctx.zeros(
                vec![config.num_attention_heads * head_dim, config.hidden_size],
                names.q_proj,
            ),
            k_weight: ctx.zeros(
                vec![config.num_key_value_heads * head_dim, config.hidden_size],
                names.k_proj,
            ),
            v_weight: ctx.zeros(
                vec![config.num_key_value_heads * head_dim, config.hidden_size],
                names.v_proj,
            ),

            o_weight: ctx.zeros(
                vec![config.hidden_size, config.num_attention_heads * head_dim],
                names.o_proj,
            ),
            ctx: ctx,
            scope_name: names.scope,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<T>,
        residual: &Tensor<T>,
        position_embedding: &Tensor<T>,
        decode_only_flag: bool,
        // tensor_name: String,
    ) -> Tensor<T> {
        {
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
            let view_key_position_tensor = view_key_states.permute(vec![1, 2, 0, 3]);

            let view_value_states = value_states.view(vec![
                value_states.shape[0],
                value_states.shape[1],
                self.num_key_value_heads,
                self.head_dim,
            ]);

            let view_value_states2 = view_value_states.permute(vec![1, 2, 0, 3]);

            let thread_num = num_cpus::get().max(1);

            // [position_window_size, batch_size, head_num, head_size] <- [position_window_size, batch_size, head_num, head_size] [batch_size, head_num, sequence_num, head_size] [batch_size, head_num, sequence_num, head_size]
            let attn_output = view_query_states.attention(
                &view_key_position_tensor,
                &view_value_states2,
                self.scaling,
                decode_only_flag,
                thread_num,
                format!("{}.attn_output", self.scope_name),
            );

            println!("attn_output shape: {:?}", attn_output.shape);
            let view_context_tensor = attn_output.view(vec![
                attn_output.shape[0],
                attn_output.shape[1],
                attn_output.shape[2] * attn_output.shape[3],
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
    use crate::mem_mgr::cache::Cache;
    use std::cell::RefCell;

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
        let ctx = Rc::new(TensorCtx::new(cache, operator_queue));

        let self_attention = Attention::new(
            &config,
            crate::moe::names::AttentionTensorNames {
                scope: String::from("model.layers.1.self_attn"),
                q_proj: String::from("model.layers.1.self_attn.q_proj.weight"),
                k_proj: String::from("model.layers.1.self_attn.k_proj.weight"),
                v_proj: String::from("model.layers.1.self_attn.v_proj.weight"),
                o_proj: String::from("model.layers.1.self_attn.o_proj.weight"),
            },
            ctx.clone(),
        );

        let hidden_states = ctx.zeros(
            vec![sequence_chunk_size, batch_size, config.hidden_size],
            String::from("model.layers.1.hidden_tensor"),
        );

        let residual_tensor = ctx.zeros(
            vec![sequence_chunk_size, batch_size, config.hidden_size],
            String::from("model.layers.1.residual_tensor"),
        );

        let position_embedding = ctx.zeros(
            vec![config.max_position_embeddings, 1, 1, attention_head_size],
            String::from("model.position_embedding.weight"),
        );

        let output =
            self_attention.forward(&hidden_states, &residual_tensor, &position_embedding, false);

        // Add assertions to validate the output
        debug_assert_eq!(
            output.shape,
            vec![sequence_chunk_size, batch_size, config.hidden_size]
        );

        // Execute the operator queue
        let thread_num: usize = num_cpus::get();
        for operator in output.operator_queue.borrow().iter() {
            for i in 0..thread_num {
                operator.run(batch_size, 0, thread_num, i, &[], &[], &mut Vec::new());
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
        let ctx = Rc::new(TensorCtx::new(cache, operator_queue));

        let self_attention = Attention::new(
            &config,
            AttentionTensorNames {
                scope: String::from("model.layers.1.self_attn"),
                q_proj: String::from("model.layers.1.self_attn.q_proj.weight"),
                k_proj: String::from("model.layers.1.self_attn.k_proj.weight"),
                v_proj: String::from("model.layers.1.self_attn.v_proj.weight"),
                o_proj: String::from("model.layers.1.self_attn.o_proj.weight"),
            },
            ctx.clone(),
        );

        let hidden_states = ctx.zeros(
            vec![sequence_chunk_size, batch_size, config.hidden_size],
            String::from("model.layers.1.hidden_tensor"),
        );

        let residual_tensor = ctx.zeros(
            vec![sequence_chunk_size, batch_size, config.hidden_size],
            String::from("model.layers.1.residual_tensor"),
        );

        let position_embedding = ctx.zeros(
            vec![config.max_position_embeddings, 1, 1, attention_head_size],
            String::from("model.position_embedding.weight"),
        );

        let output =
            self_attention.forward(&hidden_states, &residual_tensor, &position_embedding, false);

        // Add assertions to validate the output
        debug_assert_eq!(
            output.shape,
            vec![sequence_chunk_size, batch_size, config.hidden_size]
        );

        // Execute the operator queue
        let thread_num: usize = num_cpus::get();
        for operator in output.operator_queue.borrow().iter() {
            for i in 0..thread_num {
                operator.run(batch_size, 0, thread_num, i, &[], &[], &mut Vec::new());
            }
        }
    }
}
