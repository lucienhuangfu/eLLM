use std::cell::RefCell;
use std::rc::Rc;
// use serde::{Deserialize, Serialize};
use std::ops::{Add, Sub, Div, Mul, AddAssign, Neg };

use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{neg_infinity::NegInfinity, exp::Exp};
use crate::kernel::generic::sigmoid::Sigmoid;
use crate::kernel::generic::from_f32::FromF32;

use crate::init::config::Config;
use super::super::compiler::map::rms_map::RMSMap;
use super::super::compiler::map::lookup_rms_map::LookupRMSMap;
use super::super::compiler::zip_map::add_zip::AddZipMap;
use super::super::compiler::zip_map::add_rms_zip::AddRMSZipMap;
use super::super::compiler::operator::Operator;
use super::super::memory::cache::Cache;
use super::super::ptensor::tensor::Tensor;
use super::feedforward::FeedForward;
use super::attention::Attention;


#[derive(Clone)]
pub struct TransformerBlock<'a, T> {
    self_attention: Attention<T>,
    feedforward: FeedForward<T>,
    scope_name: String,
    input_layernorm: Tensor<T>,
    post_attention_layernorm: Tensor<T>,
    rms_norm_eps: T,
    cpu_num: usize,
    word_embedding: &'a Tensor<T>,
    position_embedding: &'a Tensor<T>,
    hidden_size: usize,
    head_size: usize,
    max_batch_size: usize,
    index: usize,
    cache: Rc<RefCell<Cache<T>>>,
    operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
}

impl<'a, T> TransformerBlock<'a, T> 
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
        word_embedding: &'a Tensor<T>,
        position_embedding: &'a Tensor<T>,
        index: usize,
        cpu_num: usize,
        parent_scope_name: &str,
        cache: Rc<RefCell<Cache<T>>>,
        operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    ) -> Self {
        let scope_name = format!("{}.layers.{}", parent_scope_name, index);
        TransformerBlock {
            self_attention: Attention::<T>::new(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.max_position_embeddings,
                config.batch_size,
                T::sqrt(T::from_usize(config.attention_head_size )),
                cpu_num,
                &scope_name,
                cache.clone(),
                operator_queue.clone(),
            ),
            feedforward: FeedForward::<T>::new(
                config.hidden_size,
                4 * config.hidden_size,
                config.attention_head_size,
                config.multiple_of,
                &scope_name,
                cache.clone(),
                operator_queue.clone(),
            ),
            // weight: allocate_f16(config.hidden_size),
            input_layernorm: Tensor::zeros(
                vec![1, config.hidden_size],
                format!("{}.input_layernorm.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            post_attention_layernorm: Tensor::zeros(
                vec![1, config.hidden_size],
                format!("{}.post_attention_layernorm.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            rms_norm_eps: T::from_f32(config.rms_norm_eps),
            cpu_num: cpu_num,
            word_embedding: word_embedding,
            position_embedding: position_embedding,
            hidden_size: config.hidden_size,
            head_size: config.attention_head_size,
            max_batch_size: config.batch_size,
            cache: cache,
            operator_queue: operator_queue,
            scope_name: scope_name,
            index: index,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<T>,
        sequences: *mut usize,
        tensor_name: String,
    ) -> Tensor<T> {
        // # Attention 层
        let norm_hidden = if self.index != 0 {
            hidden_states.mapv(
                Operator::RMSMap(RMSMap::new(
                    hidden_states.shape[1],
                    self.input_layernorm.data,
                    self.rms_norm_eps,
                    self.cpu_num,
                )),
                format!("{}.norm_hidden", self.scope_name),
            )
        } else {
            hidden_states.mapv(
                Operator::LookupRMSMap(LookupRMSMap::new(
                    hidden_states.shape[1],
                    self.input_layernorm.data,
                    self.rms_norm_eps,
                    self.cpu_num,
                    self.word_embedding.data,
                    sequences,
                    self.hidden_size,
                    self.max_batch_size,
                )),
                format!("{}.norm_hidden", self.scope_name),
            )
        };

        let attention_hidden1 = self.self_attention.forward(
            &norm_hidden,
            self.position_embedding,
            // format!("{}.attention_hidden1", self.scope_name),
            // self.cpu_num,
        );
        let attention_hidden2 = attention_hidden1.zip_mapv(
            hidden_states,
            Operator::AddRMSZipMap(AddRMSZipMap::new(
                attention_hidden1.shape[1],
                self.post_attention_layernorm.data,
                self.rms_norm_eps,
                self.cpu_num,
            )),
            false,
            format!("{}.norm_hidden2", self.scope_name),
        );
        let attention_hidden3 = self.feedforward.forward(
            &attention_hidden2,
            format!("{}.attention_hidden3", self.scope_name),
            self.cpu_num,
        );

        let view_attention_hidden2 = attention_hidden2.view(vec![attention_hidden2.shape[0],
            attention_hidden2.shape[1]/self.head_size , 
            self.head_size]);

        let view_attention_hidden3 = attention_hidden3.view(vec![attention_hidden3.shape[0],
            attention_hidden3.shape[1]/self.head_size , 
            self.head_size]);

        // [batch_size, head_num, head_size]
        let out = view_attention_hidden2.zip_mapv(
            &view_attention_hidden3,
            Operator::AddZipMap(AddZipMap::<T>::new(self.head_size, view_attention_hidden2.shape[1], self.cpu_num)),
            false,
            // bug
            format!("{}.output", self.scope_name),
        );
        out.view(attention_hidden2.shape.clone())
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

        let cache = Rc::new(RefCell::new(Cache::new()));
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
        let input = Tensor::from_cache(shape.clone(), String::from("model.layer.0.input_tensor"), cache.clone(), operator_queue.clone());
        for i in 0..input.shape.iter().product() {
            unsafe {
                input.data.add(i).write(1.0);
            }
        }

        let mut sequences = vec![0; config.max_position_embeddings];
        let output_tensor = layer.forward(&input, sequences.as_mut_ptr() , String::from("model.layer.0.output_tensor"));

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

        let cache: Rc<RefCell<Cache<f16>>> = Rc::new(RefCell::new(Cache::new()));
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
        let input = Tensor::from_cache(shape.clone(), String::from("model.layer.0.input_tensor"), cache.clone(), operator_queue.clone());
        for i in 0..input.shape.iter().product() {
            unsafe {
                input.data.add(i).write(1.0);
            }
        }

        let mut sequences = vec![0; config.max_position_embeddings];
        let output_tensor = layer.forward(&input, sequences.as_mut_ptr() , String::from("model.layer.0.output_tensor"));

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