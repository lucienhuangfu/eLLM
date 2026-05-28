use std::ops::{AddAssign, Neg, Sub};

use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::num_traits::{Exp, FromNumber, NegInfinity, Sigmoid, Sqrt};

use crate::common::matmul_params::MatMulParams;
use crate::tensor::{GlobalOperatorQueue, Tensor};

use super::config::Config;
use super::names::AttentionTensorNames;

// #[derive(Clone)]
pub struct Attention<T>
where
    T: Copy + PartialOrd,
{
    chunk_size: usize,
    sequence_length: usize,
    batch_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    scaling: T,
    q_weight: Tensor<T>,
    k_weight: Tensor<T>,
    v_weight: Tensor<T>,
    o_weight: Tensor<T>,
    scope_name: String,
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
        + AddAssign
        + GlobalMemPool
        + GlobalOperatorQueue,
{
    pub fn new(
        config: &Config,
        chunk_size: usize,
        batch_size: usize,
        names: AttentionTensorNames,
    ) -> Self {
        let head_dim: usize = config.head_dim;
        let scaling = T::from_f32(1.0 / (head_dim as f32).sqrt());

        Self {
            chunk_size,
            sequence_length: config.max_position_embeddings,
            batch_size: batch_size,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: head_dim,
            scaling: scaling,
            q_weight: Tensor::zeros(
                vec![config.num_attention_heads * head_dim, config.hidden_size],
                names.q_proj,
            ),
            k_weight: Tensor::zeros(
                vec![config.num_key_value_heads * head_dim, config.hidden_size],
                names.k_proj,
            ),
            v_weight: Tensor::zeros(
                vec![config.num_key_value_heads * head_dim, config.hidden_size],
                names.v_proj,
            ),

            o_weight: Tensor::zeros(
                vec![config.hidden_size, config.num_attention_heads * head_dim],
                names.o_proj,
            ),
            scope_name: names.scope,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<T>,
        residual: &Tensor<T>,
        position_embedding: &Tensor<T>,
        decode_only_flag: bool,
        _tensor_name: String,
    ) -> Tensor<T> {
        {
            //println!("hidden_states shape: {:?}", hidden_states.shape);

            // mul_rms_complex 合并operators
            // q [chunk_size, hidden_size]
            // k [batch_size, head_num, sequence_length, head_dim]
            // v [batch_size, head_num, sequence_length, head_dim]
            let (query_states, key_states, value_states) = hidden_states.matmul3(
                &self.q_weight,
                &self.k_weight,
                &self.v_weight,
                position_embedding,
                self.sequence_length,
                self.batch_size,
                self.num_key_value_heads,
                self.num_attention_heads / self.num_key_value_heads,
                self.head_dim,
                MatMulParams {
                    a_row_step_macro: 3,
                    b_row_step_macro: 128,
                    column_step_macro: 64,
                    a_row_step_micro: 3,
                    b_row_step_micro: 32,
                },
                self.scope_name.clone(),
            );

            if decode_only_flag {
                hidden_states.lift_vector();
            }

            // q [chunk_size, head_num, group_num, head_dim] <- [chunk_size, hidden_size]
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

            // [chunk_size, head_num, head_size] <- [chunk_size, head_num, group_num, head_dim] [batch_size, head_num, sequence_length, head_dim] [batch_size, head_num, sequence_length, head_dim]
            let attn_output = view_query_states.attention(
                &view_key_position_tensor,
                &view_value_states2,
                self.sequence_length,
                self.batch_size,
                self.scaling,
                decode_only_flag,
                thread_num,
                format!("{}.attn_output", self.scope_name),
            );

            let output_sequence_length = attn_output.shape[0];
            let output_batch_size = attn_output.shape[1];
            let output_hidden_size = attn_output.shape[2] * attn_output.shape[3];
            let output_rows = output_sequence_length * output_batch_size;

            let view_context_tensor = attn_output.view(vec![output_rows, output_hidden_size]);
            if decode_only_flag {
                view_context_tensor.lift_vector();
            }
            let residual_hidden_size = *residual
                .shape
                .last()
                .expect("residual tensor must have at least one dimension");
            let view_residual_tensor = residual.view(vec![output_rows, residual_hidden_size]);

            // [sequence_length, batch_size, hidden_size]
            // matmul + add
            let output_tensor_2d = view_context_tensor.matmul_add(
                &self.o_weight,
                &view_residual_tensor,
                MatMulParams {
                    a_row_step_macro: 3,
                    b_row_step_macro: 256,
                    column_step_macro: 16,
                    a_row_step_micro: 3,
                    b_row_step_micro: 32,
                },
                decode_only_flag,
                self.scope_name.clone(),
            );
            output_tensor_2d.view(vec![
                output_sequence_length,
                output_batch_size,
                self.o_weight.shape[0],
            ])
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::mem_mgr::mem_pool::GlobalMemPool;
    use std::collections::HashMap;

    #[test]
    fn test_self_attention() {
        let sequence_length = 1;
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

        f32::init_global(HashMap::new());
        f32::init_operator_queue();

        let self_attention = Attention::<f32>::new(
            &config,
            sequence_length,
            batch_size,
            crate::transformer::names::AttentionTensorNames {
                scope: String::from("model.layers.1.self_attn"),
                q_proj: String::from("model.layers.1.self_attn.q_proj.weight"),
                k_proj: String::from("model.layers.1.self_attn.k_proj.weight"),
                v_proj: String::from("model.layers.1.self_attn.v_proj.weight"),
                o_proj: String::from("model.layers.1.self_attn.o_proj.weight"),
            },
        );

        let hidden_states = Tensor::zeros(
            vec![sequence_length, batch_size, config.hidden_size],
            String::from("model.layers.1.hidden_tensor"),
        );

        let residual_tensor = Tensor::zeros(
            vec![sequence_length, batch_size, config.hidden_size],
            String::from("model.layers.1.residual_tensor"),
        );

        let position_embedding = Tensor::zeros(
            vec![config.max_position_embeddings, 1, 1, attention_head_size],
            String::from("model.position_embedding.weight"),
        );

        let output = self_attention.forward(
            &hidden_states,
            &residual_tensor,
            &position_embedding,
            false,
            String::from("test_output"),
        );

        // Add assertions to validate the output
        debug_assert_eq!(
            output.shape,
            vec![sequence_length, batch_size, config.hidden_size]
        );

        // Execute the operator queue
        let thread_num: usize = num_cpus::get();
        f32::with_operator_queue(|queue| {
            for operator in queue.iter() {
                for i in 0..thread_num {
                    operator.run(batch_size, 0, thread_num, i, &[], &[], &mut Vec::new());
                }
            }
        });

        // Add more assertions as needed
    }

    #[test]
    fn test_self_attention_f16() {
        let sequence_length = 1;
        let batch_size = 3;

        let config =
            Config::load_from_file(r"models/Qwen3-Coder-30B-A3B-Instruct/config.json").unwrap();

        let attention_head_size: usize = config.head_dim;

        f16::init_global(HashMap::new());
        f16::init_operator_queue();

        let self_attention = Attention::<f16>::new(
            &config,
            sequence_length,
            batch_size,
            AttentionTensorNames {
                scope: String::from("model.layers.1.self_attn"),
                q_proj: String::from("model.layers.1.self_attn.q_proj.weight"),
                k_proj: String::from("model.layers.1.self_attn.k_proj.weight"),
                v_proj: String::from("model.layers.1.self_attn.v_proj.weight"),
                o_proj: String::from("model.layers.1.self_attn.o_proj.weight"),
            },
        );

        let hidden_states = Tensor::zeros(
            vec![sequence_length, batch_size, config.hidden_size],
            String::from("model.layers.1.hidden_tensor"),
        );

        let residual_tensor = Tensor::zeros(
            vec![sequence_length, batch_size, config.hidden_size],
            String::from("model.layers.1.residual_tensor"),
        );

        let position_embedding = Tensor::zeros(
            vec![config.max_position_embeddings, 1, 1, attention_head_size],
            String::from("model.position_embedding.weight"),
        );

        let output = self_attention.forward(
            &hidden_states,
            &residual_tensor,
            &position_embedding,
            false,
            String::from("test_output"),
        );

        // Add assertions to validate the output
        debug_assert_eq!(
            output.shape,
            vec![sequence_length, batch_size, config.hidden_size]
        );

        // Execute the operator queue
        let thread_num: usize = num_cpus::get();
        f16::with_operator_queue(|queue| {
            for operator in queue.iter() {
                for i in 0..thread_num {
                    operator.run(batch_size, 0, thread_num, i, &[], &[], &mut Vec::new());
                }
            }
        });
    }
}
