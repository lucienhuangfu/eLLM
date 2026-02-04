use core_affinity;
use std::cell::RefCell;
use std::cell::SyncUnsafeCell;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::ptr::null;
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

// use super::super::compiler::map::rms_map::RMSMap;
use super::super::compiler::operator::Operator;
use super::super::init::matmul_params::MatMulParams;
use super::super::memory::cache::Cache;
// use super::super::memory::model_loader::SafeTensorsLoader;
// use super::super::ptensor::linear::Linear;
use super::super::ptensor::tensor::Tensor;
use super::decoder_layer::DecoderLayer;
use crate::init::record::{BatchList, TokenList, TokenRecord};

// use super::rope::precompute_freqs_cis;

// #[derive(Clone)]
pub struct Model<T>
where
    T: Copy + PartialOrd,
{
    // config: Config,
    // sequences: Vec<usize>,
    word_embedding: Rc<Tensor<T>>,
    position_embedding: Rc<Tensor<T>>,
    lm_head_weight: Tensor<T>,
    pub layers: Vec<DecoderLayer<T>>,
    rms_norm_eps: T,
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
        + PartialOrd
        + Default
        + Sub<Output = T>
        + Neg<Output = T>
        + Exp
        + NegInfinity
        + Sigmoid<T>
        + Sqrt
        + FromF32
        + AddAssign
        + Send
        + Sync,
{
    pub fn new(
        config: &Config,
        sequence_length: usize,
        // sequence_chunk_size: usize,
        batch_size: usize,
        topk_size: usize,
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
                // sequence_chunk_size,
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
                vec![config.vocab_size, config.hidden_size],
                String::from("lm_head.weight"),
                cache.clone(),
                operator_queue.clone(),
            ),
            layers: layers,
            batch_size: batch_size,
            hidden_size: config.hidden_size,
            // sequence_chunk_size: sequence_chunk_size,
            topk_size: topk_size,
            rms_norm_eps: T::from_f32(config.rms_norm_eps),
            scope_name: scope_name,
            cache: cache,
            operator_queue: operator_queue,
        }
    }

    pub fn forward(
        &mut self,
        input_sequences: *mut usize,
        token_list_ptr: *const TokenList,
        batch_list_ptr: *mut BatchList,
        eos_id: usize,
    ) -> (*const usize, Tensor<T>) {
        // -> Tensor<T> {
        // let sequences = vec![0; (self.config.max_position_embeddings + 1) * self.config.batch_size].into_boxed_slice();

        let mut hidden_state = Tensor::<T>::zeros(
            vec![self.batch_size, self.hidden_size],
            format!("{}.hidden_state.output", self.scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        for (i, layer_module) in self.layers.iter().enumerate() {
            let decode_only_flag = if i == (self.layers.len() - 1) {
                true
            } else {
                false
            };

            hidden_state = layer_module.forward(
                &hidden_state,
                input_sequences,
                token_list_ptr,
                decode_only_flag,
                format!("{}.hidden_states.{}.output", self.scope_name, i),
            );
            // all_hidden_states.push(hidden_states);
        }

        let norm_state = hidden_state.rms(
            self.rms_norm_eps,
            true,
            format!("{}.norm_hidden", self.scope_name),
        );
   
        let (indices_ptr, values_tensor) = norm_state.matmul_local_topk(
            &self.lm_head_weight,
           MatMulParams {
                    a_row_step_macro: 3,
                    b_row_step_macro: 64,
                    column_step_macro: 64,
                    a_row_step_micro: 3,
                    b_row_step_micro: 32,
                },
            self.topk_size,
            format!("{}.lm_head", self.scope_name),
        );

        let (topk_indice, topk_value) = values_tensor.topk_softmax(
            indices_ptr,

            unsafe { input_sequences.add(self.batch_size) },
            self.topk_size,
            eos_id,
            format!("{}.softmax", self.scope_name),
        );

        (topk_indice, topk_value)
        // (null(), values_tensor)
    }
}

// unsafe impl<T: Copy + Default + Send + Sync> Send for Transformer<T> {}
// unsafe impl<T: Copy + Default + Send + Sync> Sync for Transformer<T> {}

#[cfg(test)]
mod test {

    use super::*;
    // use crate::init::config::Config;
    // use crate::llama::model_loader::SafeTensorsLoader;
    use crate::init::record::{
        BatchList, BatchRecord, Phase, PrefillEndRecord, TokenList, TokenRecord,
    };
    use crate::memory::allocator::allocate_init;
    use crate::memory::cache::Cache;
    use crate::ptensor::tensor::Tensor;

    #[test]
    fn test_model_forward() {
        // let cpu_num =  thread::available_parallelism().unwrap().get();
        let sequence_length = 128;
        let sequence_chunk_size = 1;
        let batch_size = 3;
        let topk_size = 8;

        let config =
            Config::load_from_file(r"models/Qwen3-Coder-30B-A3B-Instruct/config.json").unwrap();

        let mut model = Model::<f32>::new(
            &config,
            sequence_length,
            batch_size,
            topk_size, // word_embedding,
                       // position_embedding,
                       // norm_weight,
                       // cpu_num,
                       // cache.clone(),
                       // operator_queue.clone(),
        );

 // let mut sequences: Vec<usize> = vec![0; (config.max_position_embeddings + 1)*config.batch_size];
        let mut sequences =
            allocate_init::<usize>((config.max_position_embeddings + 1) * batch_size, 0);

        let token_records: Vec<TokenRecord> = (0..batch_size)
            .map(|i| TokenRecord {
                // token_id: 0,
                batch_index: i,
                position_index: 0,
            })
            .collect();

        let lift_records: Vec<PrefillEndRecord> = Vec::new();

        let token_list = TokenList {
            token_records: token_records.into_boxed_slice(),
            current_token_size: batch_size,
            lift_records: lift_records.into_boxed_slice(),
            current_lift_size: 0,
        };

        let batch_records: Vec<BatchRecord> = (0..batch_size)
            .map(|i| BatchRecord {
                sequence_index: i,
                kv_index: i,
                phase: Phase::Decode,
                prompt_length: i,
                notify: std::sync::Arc::new(tokio::sync::Notify::new()),
            })
            .collect();

        let mut batch_list = BatchList {
            records: batch_records.into_boxed_slice(),
            current_size: batch_size,
        };

        let eos_id = 151643;

        let (output_indices, output_tensor) = unsafe {
            model.forward(
                sequences,
                &token_list as *const TokenList,
                &mut batch_list as *mut BatchList,
                eos_id,
            )
        };

        let thread_num: usize = num_cpus::get();
        for operator in model.operator_queue.borrow().iter() {
            for i in 0..thread_num {
                operator.run(batch_size, 0, thread_num, i);
            }
        }

        // Add assertions to verify the output_tensor
        // For example:
        // assert_eq!(output_tensor.shape, vec![config.batch_size, config.hidden_size]);
    }

    #[test]
    fn test_model_forward_f16() {
        let sequence_length = 128;
        let sequence_chunk_size = 1;
        let batch_size = 3;
        let topk_size = 8;

        let config =
            Config::load_from_file(r"models/Qwen3-Coder-30B-A3B-Instruct/config.json").unwrap();

        let mut model = Model::<f16>::new(
            &config,
            sequence_length,
            // sequence_chunk_size,
            batch_size,
            topk_size,
        );

        let mut sequences =
            allocate_init::<usize>((config.max_position_embeddings + 1) * batch_size, 0);
                let token_records: Vec<TokenRecord> = (0..batch_size)
            .map(|i| TokenRecord {
                // token_id: 0,
                batch_index: i,
                position_index: 0,
            })
            .collect();

        let lift_records: Vec<PrefillEndRecord> = Vec::new();

        let token_list = TokenList {
            token_records: token_records.into_boxed_slice(),
            current_token_size: batch_size,
            lift_records: lift_records.into_boxed_slice(),
            current_lift_size: 0,
        };

        let batch_records: Vec<BatchRecord> = (0..batch_size)
            .map(|i| BatchRecord {
                sequence_index: i,
                kv_index: i,
                phase: Phase::Decode,
                prompt_length: i,
                notify: std::sync::Arc::new(tokio::sync::Notify::new()),
            })
            .collect();

        let mut batch_list = BatchList {
            records: batch_records.into_boxed_slice(),
            current_size: batch_size,
        };

        let eos_id = 151643;

        let (output_indices, output_tensor) = unsafe {
            model.forward(
                sequences,
                &token_list as *const TokenList,
                &mut batch_list as *mut BatchList,
                eos_id,
            )
        };


        let thread_num = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        for operator in model.operator_queue.borrow().iter() {
            for i in 0..thread_num {
                operator.run(batch_size, 0, thread_num, i);
            }
        }
    }
}
