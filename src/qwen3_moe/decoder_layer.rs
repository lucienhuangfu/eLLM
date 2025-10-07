use std::cell::RefCell;
use std::rc::Rc;
// use serde::{Deserialize, Serialize};
use std::ops::{Add, Sub, Div, Mul, AddAssign, Neg };

use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{neg_infinity::NegInfinity, exp::Exp};
use crate::kernel::generic::sigmoid::Sigmoid;
use crate::kernel::generic::from_f32::FromF32;

use super::config::Config;

use super::super::compiler::operator::Operator;
use super::super::memory::cache::Cache;
use super::super::ptensor::tensor::Tensor;
// use super::feedforward::FeedForward;
use super::attention::Attention;
use super::moe_layer::MoeLayer;


#[derive(Clone)]
pub struct DecoderLayer<'a, T> {
    sequence_length: usize,
    sequence_chunk_size: usize,
    batch_size: usize,
    hidden_size: usize,
    head_dim: usize,
    rms_norm_eps: T,
    layer_idx: usize,
    layernorm_weight: Tensor<T>,
    word_embedding: &'a Tensor<T>,
    position_embedding: &'a Tensor<T>,
    self_attention: Attention<T>,
    moe_layer: MoeLayer<T>,

    scope_name: String,
    cache: Rc<RefCell<Cache<T>>>,
    operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
}

impl<'a, T> DecoderLayer<'a, T> 
where T: Copy 
    + Default 
    + Sub<Output = T>
    + Neg<Output = T>
    + Exp
    + NegInfinity
    + Sigmoid<T>
    + Sqrt
    + FromF32

{
    pub fn new(
        config: &Config,
        layer_idx: usize,
        sequence_length: usize,
        sequence_chunk_size: usize,
        batch_size: usize,
        word_embedding: &'a Tensor<T>,
        position_embedding: &'a Tensor<T>,
        parent_scope_name: &str,
        cache: Rc<RefCell<Cache<T>>>,
        operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    ) -> Self {
        let scope_name = format!("{}.layers.{}", parent_scope_name, layer_idx);

        let moe_layer = if config.num_experts > 0 && (layer_idx + 1) % config.decoder_sparse_step == 0 {
            // sparse moe block
            MoeLayer::new_sparse_moe(
                sequence_chunk_size,
                config.hidden_size,
                config.num_experts,
                config.num_experts_per_tok,
                config.norm_topk_prob,
                config.intermediate_size,
                config.head_dim,
                &scope_name,
                cache.clone(),
                operator_queue.clone(),
            )
        } else {
            // MLP Layer
            MoeLayer::new_mlp(
                sequence_chunk_size,
                config.head_dim,
                config.hidden_size,
                config.intermediate_size,
                &scope_name,
                cache.clone(),
                operator_queue.clone(),
            )
        };

        Self {
            sequence_length: config.max_position_embeddings,
            sequence_chunk_size: sequence_chunk_size,
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
            moe_layer: moe_layer,
            // weight: allocate_f16(config.hidden_size),
            layernorm_weight: Tensor::zeros(
                vec![1, config.head_dim],
                format!("{}.layernorm.weight", scope_name),
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
        // sequences: *mut usize,
        tensor_name: String,
    ) -> Tensor<T> {
        // # Attention 层
        let (hidden_states, norm_hidden) = if self.index != 0 {
            let norm_hidden = hidden_states.rms(
                    self.input_layernorm.data,
                    self.rms_norm_eps,
                format!("{}.norm_hidden", self.scope_name),
            );
            (hidden_states, norm_hidden)
        } else {
            hidden_states.lookup_rms(
                self.word_embedding.data,
                    self.input_layernorm.data,
                    self.rms_norm_eps,
                format!("{}.norm_hidden", self.scope_name),
            )
        };

        //  attention + add
        let attention_hidden_states = self.self_attention.forward(
            &norm_hidden,
            hidden_states,
            self.position_embedding,
            // format!("{}.attention_hidden1", self.scope_name),
            // self.cpu_num,
        );
        let norm_hidden_states = attention_hidden_states.rms(
            self.layernorm_weight.data,
            self.rms_norm_eps,
            format!("{}.norm_hidden2", self.scope_name)
        );

        let output_hidden_states = self.moe_layer.forward(
            &norm_hidden_states,
            &attention_hidden_states,
            format!("{}.attention_hidden3", self.scope_name),
            // num_cpus::get(),
        );

        /* 
        let view_attention_hidden2 = attention_hidden2.view(vec![attention_hidden2.shape[0],
            attention_hidden2.shape[1]/self.head_dim, 
            self.head_dim]);

        let view_attention_hidden3 = attention_hidden3.view(vec![attention_hidden3.shape[0],
            attention_hidden3.shape[1]/self.head_dim, 
            self.head_dim]);


        // [batch_size, head_num, head_size]
        let out = view_attention_hidden2.add(
            &view_attention_hidden3,
            format!("{}.output", self.scope_name),
        );
        
        out.view(attention_hidden2.shape.clone())
        */
        output_hidden_states

    }
}

#[cfg(test)]
mod test {

    use super::*;
    use std::slice;
    use approx::assert_relative_eq;

    #[test]
    fn test_layer() {
        let cpu_num = num_cpus::get();
        let mut config: Config = Config::new();
        config.load_model_config(r"models/Llama-2-7b-hf/config.json");
        config.load_compile_config(r"models/Llama-2-7b-hf.json");

        let batch_size = config.batch_size;
        // let sequence_length = config;
        let hidden_size = config.hidden_size;
        let num_attention_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads;
        let max_position_embeddings = config.max_position_embeddings;
        let attention_head_size = config.attention_head_size;
        let multiple_of = config.multiple_of;
        let rms_norm_eps = config.rms_norm_eps;

        let cache = Rc::new(RefCell::new(Cache::new(std::collections::HashMap::new())));
        let operator_queue = Rc::new(RefCell::new(Vec::new()));

        let vocab_size = 4096;
        let word_embedding = Tensor::zeros(vec![4096, hidden_size], String::from("model.word_embedding.weight"), cache.clone(), operator_queue.clone());
        let position_embedding = Tensor::zeros(vec![max_position_embeddings, 1, 1, attention_head_size], String::from("model.position_embedding.weight"), cache.clone(), operator_queue.clone());

        let layer = TransformerBlock::<f32>::new(&config, 
            &word_embedding, 
            &position_embedding, 
            0, 
            cpu_num, 
            "model", 
            cache.clone(), 
            operator_queue.clone());

        let shape = vec![batch_size, hidden_size];
        let input = Tensor::from_cache(shape.clone(), String::from("model.layers.0.input_tensor"), cache.clone(), operator_queue.clone());
        for i in 0..input.shape.iter().product() {
            unsafe {
                input.data.add(i).write(1.0);
            }
        }

        let mut sequences = vec![0; config.max_position_embeddings];
        let output_tensor = layer.forward(&input, sequences.as_mut_ptr() , String::from("model.layers.0.output_tensor"));

        let thread_num: usize = num_cpus::get();
        for operator in output_tensor.operator_queue.borrow().iter() {
            for i in 0..thread_num {
                operator.run(1, 64, i);
            }
        }

        /*
        let output_shape = vec![batch_size, hidden_size];
        let size = output_shape.iter().product();
        let mut result = vec![0.0; size];
        for i in 0..hidden_size {
            result[i] = hidden_size as f32;
        }

        let output_slice = unsafe { std::slice::from_raw_parts(output_tensor.data, size) };
        assert_relative_eq!(output_slice, &result[..], max_relative = 1e-6);
         */
    }

     
    #[test]
    fn test_layer_f16() {
        let cpu_num = num_cpus::get();
        let mut config: Config = Config::new();
        config.load_model_config(r"models/Llama-2-7b-hf/config.json");
        config.load_compile_config(r"models/Llama-2-7b-hf.json");

        let batch_size = config.batch_size;
        // let sequence_length = config;
        let hidden_size = config.hidden_size;
        let num_attention_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads;
        let max_position_embeddings = config.max_position_embeddings;
        let attention_head_size = config.attention_head_size;
        let multiple_of = config.multiple_of;
        let rms_norm_eps = config.rms_norm_eps as f16;

        let cache: Rc<RefCell<Cache<f16>>> = Rc::new(RefCell::new(Cache::new(std::collections::HashMap::new())));
        let operator_queue = Rc::new(RefCell::new(Vec::new()));

        let vocab_size = 4096;
        let word_embedding = Tensor::zeros(vec![4096, hidden_size], String::from("model.word_embedding.weight"), cache.clone(), operator_queue.clone());
        let position_embedding = Tensor::zeros(vec![max_position_embeddings, 1, 1, attention_head_size], String::from("model.position_embedding.weight"), cache.clone(), operator_queue.clone());

        let layer = TransformerBlock::<f16>::new(&config, 
            &word_embedding, 
            &position_embedding, 
            0, 
            cpu_num, 
            "model", 
            cache.clone(), 
            operator_queue.clone());

        let shape = vec![batch_size, hidden_size];
        let input = Tensor::from_cache(shape.clone(), String::from("model.layers.0.input_tensor"), cache.clone(), operator_queue.clone());
        for i in 0..input.shape.iter().product() {
            unsafe {
                input.data.add(i).write(1.0);
            }
        }

        let mut sequences = vec![0; config.max_position_embeddings];
        let output_tensor = layer.forward(&input, sequences.as_mut_ptr() , String::from("model.layers.0.output_tensor"));

        let thread_num: usize = num_cpus::get();
        for operator in output_tensor.operator_queue.borrow().iter() {
            for i in 0..thread_num {
                operator.run(1, 128, i);
            }
        }

        /*
        let output_shape = vec![batch_size, hidden_size];
        let size = output_shape.iter().product();
        let mut result = vec![0.0; size];
        for i in 0..hidden_size {
            result[i] = hidden_size as f32;
        }

        let output_slice = unsafe { std::slice::from_raw_parts(output_tensor.data, size) };
        assert_relative_eq!(output_slice, &result[..], max_relative = 1e-6);
         */
    } 

}