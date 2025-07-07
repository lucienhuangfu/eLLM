use std::cell::RefCell;
use std::rc::Rc;
use serde::{Deserialize, Serialize};
use std::ops::{Add, Sub, Div, Mul, AddAssign, Neg };

use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{neg_infinity::NegInfinity, exp::Exp};
use crate::kernel::generic::sigmoid::Sigmoid;
use crate::kernel::generic::from_f32::FromF32;

use super::super::compiler::operator::Operator;
use super::super::compiler::map::rms_map::RMSMap;


use crate::init::config::Config;
use super::super::memory::cache::Cache;
use super::super::ptensor::linear::Linear;
use super::super::ptensor::tensor::Tensor;
use super::transformer_block::TransformerBlock;
// use super::rope::precompute_freqs_cis;

#[derive(Clone)]
pub struct Model<T> {
    config: Config,
    // pub layer: Vec<Layer<T>>,
    norm_weight: Tensor<T>,
    word_embedding: Tensor<T>,
    rms_norm_eps: T,
    output_linear: Linear<T>,
    scope_name: String,
    position_embedding: Tensor<T>,
    cpu_num: usize,
    pub cache: Rc<RefCell<Cache<T>>>,
    pub operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
   
}

impl<T> Model<T> 
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
        config: Config,
        word_embedding: Tensor<T>,
        position_embedding: Tensor<T>,
        norm_weight: Tensor<T>,
        cpu_num: usize,
        cache: Rc<RefCell<Cache<T>>>,
        operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    ) -> Self {
        let scope_name = String::from("model");
        let module_vec: Vec<TransformerBlock<T>> = Vec::new();
        Model {
            // layer: module_vec,
            output_linear: Linear::<T>::new(
                config.hidden_size,
                config.vocab_size,
                1,
                format!("lm_head"),
                cache.clone(),
                operator_queue.clone()
            ),

            position_embedding: position_embedding,
            word_embedding: word_embedding,
            norm_weight: norm_weight,
            rms_norm_eps: T::from_f32(config.rms_norm_eps),
            config: config,
            cpu_num: cpu_num,
            scope_name: scope_name,
            cache: cache,
            operator_queue: operator_queue,
        }
    }

    pub fn forward(&self, sequences: *mut usize) -> Tensor<T> {
        let mut layer_vec: Vec<TransformerBlock<T>> = Vec::new();

        for i in 0..self.config.num_hidden_layers {
            layer_vec.push(TransformerBlock::<T>::new(
                &self.config,
                &self.word_embedding,
                &self.position_embedding,
                i,
                self.cpu_num,
                &self.scope_name,
                self.cache.clone(),
                self.operator_queue.clone(),
            ));
        }

        let mut hidden_state =
            Tensor::<T>::zeros(vec![self.config.batch_size, self.config.hidden_size], format!("{}.norm.weight", self.scope_name),self.cache.clone(), self.operator_queue.clone());
        for (i, layer_module) in layer_vec.iter().enumerate() {
            hidden_state = layer_module.forward(
                &hidden_state,
                sequences,
                format!("{}.hidden_states.{}", self.scope_name, i),
            );
            // all_hidden_states.push(hidden_states);
        }
        
    
        // hidden_states.last().unwrap().to_owned();

   
        let norm_output = hidden_state.mapv(
            Operator::RMSMap(RMSMap::new(self.config.hidden_size, self.norm_weight.data, self.rms_norm_eps, self.cpu_num)),
            format!("{}.norm_hidden.output", self.scope_name),
        );    
    
        let logits = self
            .output_linear
            .forward(&norm_output, format!("{}.lm_head.output", self.scope_name));
        /* 
        unsafe {
            logits.reduce(
                sequences.add(self.config.batch_size),
                self.config.max_position_embeddings,
                Operator::ArgmaxReduce(ArgmaxReduce::new(
                    self.config.hidden_size,
                    self.config.batch_size,
                    self.cpu_num,
                )),
                format!("{}.argmax.weight", self.scope_name),
            );
        }*/

        hidden_state
    }
}


#[cfg(test)]
mod test {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::thread;

    use super::*;
    use crate::init::config::Config;
    use crate::ptensor::tensor::Tensor;
    use crate::memory::cache::Cache;
    use crate::memory::allocator::allocate_init;

    #[test]
    fn test_model_forward() {
        let cpu_num =  thread::available_parallelism().unwrap().get();
        let mut config: Config = Config::new();
        config.load_model_config(r"models/Llama-2-7b-hf/config.json");
        config.load_compile_config(r"models/Llama-2-7b-hf.json");

        let cache = Rc::new(RefCell::new(Cache::new()));
        let operator_queue = Rc::new(RefCell::new(Vec::new()));

        let word_embedding = Tensor::zeros(vec![config.vocab_size, config.hidden_size], String::from("model.embed_tokens.weight"), cache.clone(), operator_queue.clone());
        let position_embedding = Tensor::zeros(vec![config.max_position_embeddings, 1, 1, config.attention_head_size], String::from("model.position_embedding.output"), cache.clone(), operator_queue.clone());
        let norm_weight = Tensor::zeros(vec![1, config.hidden_size], String::from("model.norm.weight"), cache.clone(), operator_queue.clone());

        let model = Model::<f32>::new(
            config.clone(),
            word_embedding,
            position_embedding,
            norm_weight,
            cpu_num,
            cache.clone(),
            operator_queue.clone(),
        );

        // let mut sequences: Vec<usize> = vec![0; (config.max_position_embeddings + 1)*config.batch_size];
        let mut sequences = allocate_init::<usize>((config.max_position_embeddings + 1)*config.batch_size, 0);
        let output_tensor = unsafe {
            model.forward(sequences) 
        };
        let thread_num: usize = num_cpus::get();
        for operator in output_tensor.operator_queue.borrow().iter() {
            for i in 0..thread_num {
                operator.run(1, 0, i);
            }
        }

        // Add assertions to verify the output_tensor
        // For example:
        // assert_eq!(output_tensor.shape, vec![config.batch_size, config.hidden_size]);
    }

    #[test]
    fn test_model_forward_f16() {
        let cpu_num = thread::available_parallelism().unwrap().get();
        let mut config: Config = Config::new();
        config.load_model_config(r"models/Llama-2-7b-hf/config.json");
        config.load_compile_config(r"models/Llama-2-7b-hf.json");

        let cache: Rc<RefCell<Cache<f16>>> = Rc::new(RefCell::new(Cache::new()));
        let operator_queue = Rc::new(RefCell::new(Vec::new()));

        let word_embedding = Tensor::zeros(vec![config.vocab_size, config.hidden_size], String::from("model.embed_tokens.weight"), cache.clone(), operator_queue.clone());
        let position_embedding = Tensor::zeros(vec![config.max_position_embeddings, 1, 1, config.attention_head_size], String::from("model.position_embedding.output"), cache.clone(), operator_queue.clone());
        let norm_weight = Tensor::zeros(vec![1, config.hidden_size], String::from("model.norm.weight"), cache.clone(), operator_queue.clone());

        let model = Model::<f16>::new(
            config.clone(),
            word_embedding,
            position_embedding,
            norm_weight,
            cpu_num,
            cache.clone(),
            operator_queue.clone(),
        );

        let sequence_length = 128;
        // let mut sequences: Vec<usize> = vec![0; (config.max_position_embeddings + 1)*config.batch_size];
        let mut sequences = allocate_init::<usize>((sequence_length + 1)*config.batch_size, 0);

        let output_tensor = unsafe {
            model.forward(sequences) 
        };
        let thread_num: usize = cpu_num;

        for p in 0..sequence_length {
            for operator in output_tensor.operator_queue.borrow().iter() {
                for i in 0..thread_num {
                    operator.run(1, p, i);
                }
                println!("{}", p);
            }
        }



        // Add assertions to verify the output_tensor
        // For example:
        // assert_eq!(output_tensor.shape, vec![config.batch_size, config.hidden_size]);
    }

}
