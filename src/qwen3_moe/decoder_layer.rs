use std::cell::RefCell;
use std::rc::Rc;
// use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use crate::kernel::generic::from_f32::FromF32;
use crate::kernel::generic::sigmoid::Sigmoid;
use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{exp::Exp, neg_infinity::NegInfinity};

use super::super::compiler::operator::Operator;
use super::super::memory::cache::Cache;
use super::super::ptensor::tensor::Tensor;
use super::attention::Attention;
use super::config::Config;
use super::sparse_moe_block::SparseMoeBlock;
use crate::init::record::TokenRecord;
// use super::moe_layer::MoeLayer;
// use crate::qwen3_moe::mlp;
// use super::feedforward::FeedForward;

#[derive(Clone)]
pub struct DecoderLayer<T> {
    sequence_length: usize,
    // sequence_chunk_size: usize,
    batch_size: usize,
    hidden_size: usize,
    head_dim: usize,
    rms_norm_eps: T,
    layer_idx: usize,
    word_embedding: Rc<Tensor<T>>,
    position_embedding: Rc<Tensor<T>>,
    self_attention: Attention<T>,
    // moe_layer: MoeLayer<T>,
    sparse_moe_block: SparseMoeBlock<T>,
    scope_name: String,
    cache: Rc<RefCell<Cache<T>>>,
    operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
}

impl<T> DecoderLayer<T>
where
    T: Copy
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
        sequence_length: usize,
        // sequence_chunk_size: usize,
        batch_size: usize,
        word_embedding: Rc<Tensor<T>>,
        position_embedding: Rc<Tensor<T>>,
        parent_scope_name: &str,
        cache: Rc<RefCell<Cache<T>>>,
        operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    ) -> Self {
        let scope_name = format!("{}.layers.{}", parent_scope_name, layer_idx);

        // let mlp_scope_name = format!("{}.mlp", scope_name);
        /*
        let moe_layer =
            if config.num_experts > 0 && (layer_idx + 1) % config.decoder_sparse_step == 0 {
                // sparse moe block
                MoeLayer::new_sparse_moe(
                    config.hidden_size,
                    config.intermediate_size,
                    config.num_experts,
                    config.num_experts_per_tok,
                    config.norm_topk_prob,
                    &scope_name,
                    cache.clone(),
                    operator_queue.clone(),
                )
            } else {
                // MLP Layer
                MoeLayer::new_mlp(
                    config.hidden_size,
                    config.intermediate_size,
                    &scope_name,
                    cache.clone(),
                    operator_queue.clone(),
                )
            };*/

        Self {
            sequence_length: config.max_position_embeddings,
            // sequence_chunk_size: sequence_chunk_size,
            batch_size: batch_size,
            hidden_size: config.hidden_size,
            head_dim: config.head_dim,
            rms_norm_eps: T::from_f32(config.rms_norm_eps),
            layer_idx: layer_idx,
            self_attention: Attention::<T>::new(
                config,
                layer_idx,
                &scope_name,
                cache.clone(),
                operator_queue.clone(),
            ),
            sparse_moe_block: SparseMoeBlock::new(
                config.hidden_size,
                config.moe_intermediate_size,
                config.num_experts,
                config.num_experts_per_tok,
                config.norm_topk_prob,
                &scope_name,
                cache.clone(),
                operator_queue.clone(),
            ),
            word_embedding: word_embedding,
            position_embedding: position_embedding,
            cache: cache,
            operator_queue: operator_queue,
            scope_name: scope_name,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<T>,
        token_ptr: *const TokenRecord,
        // input_sequences: *mut usize,
        // sequences: *mut usize,
        decode_only_flag: bool,
        tensor_name: String,
    ) -> Tensor<T> {
        // # Attention 层
        let (hidden_states_owned, norm_hidden) = if self.layer_idx != 0 {
            let norm_hidden = hidden_states.rms(
                self.rms_norm_eps,
                false,
                format!("{}.norm_hidden", self.scope_name),
            );
            (hidden_states.clone(), norm_hidden)
        } else {
            Tensor::lookup_rms(
                token_ptr,
                &*self.word_embedding,
                // self.sequence_chunk_size,
                self.batch_size,
                self.rms_norm_eps,
                self.scope_name.clone(),
                self.cache.clone(),
                self.operator_queue.clone(),
            )
        };
        let hidden_states = &hidden_states_owned;

        //  attention + add
        let attention_hidden_states = self.self_attention.forward(
            &norm_hidden,
            hidden_states,
            &*self.position_embedding,
            decode_only_flag,
            // format!("{}.attention_hidden1", self.scope_name),
        );

        let norm_hidden_states = attention_hidden_states.rms(
            // self.layernorm_weight.data,
            self.rms_norm_eps,
            decode_only_flag,
            format!("{}.norm_hidden2", self.scope_name),
        );

        norm_hidden_states.data;
        let output_hidden_states = self.sparse_moe_block.forward(
            &norm_hidden_states,
            &attention_hidden_states,
            decode_only_flag,
            format!("{}.attention_hidden3", self.scope_name),
            // num_cpus::get(),
        );

        output_hidden_states
        // hidden_states.clone()
        // attention_hidden_states
        // norm_hidden_states
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::init::record::TokenRecord;
    use std::slice;

    #[test]
    fn test_decoder_layer_f32() {
        let batch_size = 6;

        let config =
            Config::load_from_file(r"models/Qwen3-Coder-30B-A3B-Instruct/config.json").unwrap();

        let hidden_size = config.hidden_size;
        let max_position_embeddings = config.max_position_embeddings;
        let head_dim = config.head_dim;

        let cache = Rc::new(RefCell::new(Cache::new(std::collections::HashMap::new())));
        let operator_queue = Rc::new(RefCell::new(Vec::new()));

        let vocab_size = config.vocab_size;
        let word_embedding = Rc::new(Tensor::zeros(
            vec![vocab_size, hidden_size],
            String::from("model.embed_tokens.weight"),
            cache.clone(),
            operator_queue.clone(),
        ));
        let position_embedding = Rc::new(Tensor::zeros(
            vec![max_position_embeddings, 1, 1, head_dim],
            String::from("model.rotary_emb.weight"),
            cache.clone(),
            operator_queue.clone(),
        ));

        let layer = DecoderLayer::<f32>::new(
            &config,
            1,
            max_position_embeddings,
            // sequence_chunk_size,
            batch_size,
            word_embedding.clone(),
            position_embedding.clone(),
            "model",
            cache.clone(),
            operator_queue.clone(),
        );

        let shape = vec![batch_size, hidden_size];
        let input = Tensor::from_cache(
            shape.clone(),
            String::from("model.layers.1.input_tensor"),
            cache.clone(),
            operator_queue.clone(),
        );

        for i in 0..input.shape.iter().product() {
            unsafe {
                input.data.add(i).write(1.0);
            }
        }

        let sequences: Vec<TokenRecord> = (0..batch_size)
            .map(|i| TokenRecord {
                token_id: 0,
                batch_index: i,
                position_index: 0,
            })
            .collect();

        let output_tensor = layer.forward(
            &input,
            sequences.as_ptr(),
            false,
            String::from("model.layers.1.output_tensor"),
        );

        // Validate output shape
        debug_assert_eq!(output_tensor.shape, vec![batch_size, hidden_size]);

        // Execute the operator queue
        let thread_num: usize = num_cpus::get();
        for (index, operator) in output_tensor.operator_queue.borrow().iter().enumerate() {
            println!("operator {} in queue", index);
            for i in 0..thread_num {
                operator.run(batch_size, 0, thread_num, i);
            }
        }

        assert_eq!(output_tensor.shape, vec![batch_size, hidden_size]);
    }

    /*
    #[test]
    fn test_decoder_layer_f16() {
        let position_window_size = 4;
        let batch_size = 6;

        let config = Config::load_from_file(r"models/Qwen2.5-0.5B-Instruct/config.json").unwrap();

        let sequence_chunk_size = position_window_size;
        let hidden_size = config.hidden_size;
        let max_position_embeddings = config.max_position_embeddings;
        let head_dim = config.head_dim;

        let cache: Rc<RefCell<Cache<f16>>> =
            Rc::new(RefCell::new(Cache::new(std::collections::HashMap::new())));
        let operator_queue = Rc::new(RefCell::new(Vec::new()));

        let vocab_size = config.vocab_size;
        let word_embedding = Rc::new(Tensor::zeros(
            vec![vocab_size, hidden_size],
            String::from("model.embed_tokens.weight"),
            cache.clone(),
            operator_queue.clone(),
        ));
        let position_embedding = Rc::new(Tensor::zeros(
            vec![max_position_embeddings, 1, 1, head_dim],
            String::from("model.rotary_emb.weight"),
            cache.clone(),
            operator_queue.clone(),
        ));

        let layer = DecoderLayer::<f16>::new(
            &config,
            0,
            max_position_embeddings,
            sequence_chunk_size,
            batch_size,
            word_embedding.clone(),
            position_embedding.clone(),
            "model",
            cache.clone(),
            operator_queue.clone(),
        );

        let shape = vec![position_window_size, batch_size, hidden_size];
        let input = Tensor::from_cache(
            shape.clone(),
            String::from("model.layers.0.input_tensor"),
            cache.clone(),
            operator_queue.clone(),
        );

        for i in 0..input.shape.iter().product() {
            unsafe {
                input.data.add(i).write(f16::from_f32(1.0));
            }
        }

        let mut sequences = vec![0; sequence_chunk_size * batch_size];
        let output_tensor = layer.forward(
            &input,
            sequences.as_mut_ptr(),
            String::from("model.layers.0.output_tensor"),
        );

        // Validate output shape
        debug_assert_eq!(
            output_tensor.shape,
            vec![position_window_size, batch_size, hidden_size]
        );

        // Execute the operator queue
        let thread_num: usize = num_cpus::get();
        for (index, operator) in output_tensor.operator_queue.borrow().iter().enumerate() {
            println!("operator {} in queue", index);
            for i in 0..thread_num {
                operator.run(1, 1, batch_size, thread_num, i);
            }
        }

        assert_eq!(
            output_tensor.shape,
            vec![position_window_size, batch_size, hidden_size]
        );
    } */
}
