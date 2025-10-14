use core_affinity;
use nom::sequence;
use std::cell::RefCell;
use std::cell::SyncUnsafeCell;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::rc::Rc;
use std::sync::Barrier;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Instant;

// use serde::{Deserialize, Serialize};
// use hurdles::Barrier;
// use super::barrier::Barrier;
// use serde::{Deserialize, Serialize};

use super::config::Config;
use crate::kernel::generic::from_f32::FromF32;
use crate::kernel::generic::sigmoid::Sigmoid;
use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{exp::Exp, neg_infinity::NegInfinity};

use super::super::compiler::map::rms_map::RMSMap;
use super::super::compiler::operator::Operator;
use super::super::init::matmul_params::matmulParams;
use super::super::memory::cache::Cache;
use super::super::memory::model_loader::SafeTensorsLoader;
use super::super::ptensor::linear::Linear;
use super::super::ptensor::tensor::Tensor;
use super::decoder_layer::DecoderLayer;

// use super::rope::precompute_freqs_cis;

#[derive(Clone)]
pub struct Model<T> {
    // config: Config,

    // sequences: Vec<usize>,
    word_embedding: Rc<Tensor<T>>,
    position_embedding: Rc<Tensor<T>>,
    lm_head_weight: Tensor<T>,
    pub layers: Vec<DecoderLayer<T>>,
    rms_norm_eps: T,
    pub sequence_chunk_size: usize,
    pub batch_size: usize,
    pub hidden_size: usize,
    pub topk_size: usize,
    scope_name: String,
    pub cache: Rc<RefCell<Cache<T>>>,
    pub operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
}

impl<T> Model<T>
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
        + Send
        + Sync,
{
    pub fn new(
        config: &Config,
        
        sequence_length: usize,
        sequence_chunk_size: usize,
        batch_size: usize,
    ) -> Self {
        let scope_name = String::from("model");

        // let torch_file = String::from("D:/llama-3-chinese-8b-instruct-v3");
        // let loader = SafeTensorsLoader::new(&torch_file).unwrap();
        // let tensors = loader.load_all_weights_f16().unwrap();
        let parameter_tensors = std::collections::HashMap::new();

        let cache = Rc::new(RefCell::new(Cache::new(parameter_tensors)));

        let operator_queue: Rc<RefCell<Vec<Operator<T>>>> = Rc::new(RefCell::new(Vec::new()));

        // Create default tensors
        let word_embedding = Rc::new(Tensor::zeros(
            vec![config.vocab_size, config.hidden_size],
            String::from("model.embed_tokens.weight"),
            cache.clone(),
            operator_queue.clone(),
        ));

        let position_embedding = Rc::new(Tensor::zeros(
            vec![config.max_position_embeddings, 1, 1, config.head_dim],
            String::from("model.position_embedding.weight"),
            cache.clone(),
            operator_queue.clone(),
        ));

        let mut layers: Vec<DecoderLayer<T>> = Vec::new();
        for i in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::<T>::new(
                &config,
                i,
                sequence_length,
                sequence_chunk_size,
                batch_size,
                word_embedding.clone(),
                position_embedding.clone(),
                &scope_name.clone(),
                cache.clone(),
                operator_queue.clone(),
            ));
        }

        Self {
            // sequences: vec![0; (config.max_position_embeddings + 1) * batch_size],
            word_embedding: word_embedding.clone(),
            position_embedding: position_embedding.clone(),
            lm_head_weight: Tensor::zeros(
                vec![config.hidden_size, config.vocab_size],
                String::from("lm_head.weight"),
                cache.clone(),
                operator_queue.clone(),
            ),            
            layers: layers,
            batch_size: batch_size,
            hidden_size: config.hidden_size,
            sequence_chunk_size: sequence_chunk_size,
            topk_size: config.num_experts_per_tok,
            rms_norm_eps: T::from_f32(config.rms_norm_eps),
            scope_name: scope_name,
            cache: cache,
            operator_queue: operator_queue,
        }
    }

    pub fn forward(&mut self, sequences: *mut usize) -> (*const usize, Tensor<T>) {
        // -> Tensor<T> {
        // let sequences = vec![0; (self.config.max_position_embeddings + 1) * self.config.batch_size].into_boxed_slice();

        let mut hidden_state = Tensor::<T>::zeros(
            vec![self.batch_size, self.hidden_size],
            format!("{}.norm.weight", self.scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        for (i, layer_module) in self.layers.iter().enumerate() {
            hidden_state = layer_module.forward(
                &hidden_state,
                sequences,
                format!("{}.hidden_states.{}", self.scope_name, i),
            );
            // all_hidden_states.push(hidden_states);
        }

        let norm_state = hidden_state.rms(
            self.rms_norm_eps,
            format!("{}.norm_hidden.output", self.scope_name),
        );

        let (indices_ptr, values_tensor, sum_tensor) = norm_state.matmul_topk(
            &self.lm_head_weight,
            matmulParams {
                a_row_step_macro: 16,
                b_row_step_macro: 16,
                column_step_macro: 16,
                a_row_step_micro: 8,
                b_row_step_micro: 8,
            },
            8,
            format!("{}.lm_head.output", self.scope_name),
        );

        let (topk_indice, topk_value) = values_tensor.topk_softmax(
            indices_ptr,
            &sum_tensor,
            self.topk_size,
            format!("{}.softmax.output", self.scope_name),
        );

        (topk_indice, topk_value)
    }
}

// unsafe impl<T: Copy + Default + Send + Sync> Send for Transformer<T> {}
// unsafe impl<T: Copy + Default + Send + Sync> Sync for Transformer<T> {}

#[cfg(test)]
mod test {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::thread;

    /*
    use super::*;
    use crate::init::config::Config;
    use crate::llama::model_loader::SafeTensorsLoader;
    use crate::memory::allocator::allocate_init;
    use crate::memory::cache::Cache;
    use crate::ptensor::tensor::Tensor;

    #[test]
    fn test_model_forward() {
        // let cpu_num =  thread::available_parallelism().unwrap().get();
        let mut config: Config = Config::new();
        config.load_model_config(r"models/Llama-2-7b-hf/config.json");
        config.load_compile_config(r"models/Llama-2-7b-hf.json");

        let mut model = Transformer::<f32>::new(
            config.clone(),
            // word_embedding,
            // position_embedding,
            // norm_weight,
            // cpu_num,
            // cache.clone(),
            // operator_queue.clone(),
        );

        // let mut sequences: Vec<usize> = vec![0; (config.max_position_embeddings + 1)*config.batch_size];
        // let mut sequences = allocate_init::<usize>((config.max_position_embeddings + 1)*config.batch_size, 0);
        let output_tensor = unsafe { model.build() };
        /*
        let thread_num: usize = num_cpus::get();
        for operator in output_tensor.operator_queue.borrow().iter() {
            for i in 0..thread_num {
                operator.run(1, 0, i);
            }
        }
         */
        // Add assertions to verify the output_tensor
        // For example:
        // assert_eq!(output_tensor.shape, vec![config.batch_size, config.hidden_size]);
    } */

    /*
       #[test]
       fn test_model_forward_f16() {
           let cpu_num = thread::available_parallelism().unwrap().get();
           let mut config: Config = Config::new();
           config.load_model_config(r"models/Llama-2-7b-hf/config.json");
           config.load_compile_config(r"models/Llama-2-7b-hf.json");

           let model_dir = "models/Llama-2-7b-hf"; // Define model_dir
           let loader = SafeTensorsLoader::new("D:/llama-3-chinese-8b-instruct-v3").unwrap();

           // 分别加载配置和权重
           // let config = loader.load_config()?;
           let weights = loader.load_all_weights_f16().unwrap();

           let cache: Rc<RefCell<Cache<f16>>> = Rc::new(RefCell::new(Cache::new(weights)));
           let operator_queue = Rc::new(RefCell::new(Vec::new()));

           let word_embedding = Tensor::zeros(vec![config.vocab_size, config.hidden_size], String::from("model.embed_tokens.weight"), cache.clone(), operator_queue.clone());
           let position_embedding = Tensor::zeros(vec![config.max_position_embeddings, 1, 1, config.attention_head_size], String::from("model.position_embedding.output"), cache.clone(), operator_queue.clone());
           let norm_weight = Tensor::zeros(vec![1, config.hidden_size], String::from("model.norm.weight"), cache.clone(), operator_queue.clone());

           let model = Transformer::<f16>::new(
               config.clone(),
               // word_embedding,
               // position_embedding,
               // norm_weight,
               // cpu_num,
               // cache.clone(),
               // operator_queue.clone(),
           );

           let sequence_length = 128;
           // let mut sequences: Vec<usize> = vec![0; (config.max_position_embeddings + 1)*config.batch_size];
           let mut sequences = allocate_init::<usize>((sequence_length + 1)*config.batch_size, 0);

           let output_tensor = unsafe {
               model.build(sequences)
           };
    */
    /*
    let thread_num: usize = cpu_num;

    for p in 0..sequence_length {
        for operator in output_tensor.operator_queue.borrow().iter() {
            for i in 0..thread_num {
                operator.run(1, p, i);
            }
            println!("{}", p);
        }
    }
    */

    // Add assertions to verify the output_tensor
    // For example:
    // assert_eq!(output_tensor.shape, vec![config.batch_size, config.hidden_size]);
    // }
}
