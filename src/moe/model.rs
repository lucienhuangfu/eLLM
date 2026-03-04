use core_affinity;
use std::cell::RefCell;
// use std::cell::SyncUnsafeCell;
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
use crate::common::num_traits::FromNumber;
use crate::common::num_traits::Sigmoid;
use crate::common::num_traits::Sqrt;
use crate::common::num_traits::{exp::Exp, neg_infinity::NegInfinity};

// use super::super::ops::map::rms_map::RMSMap;
use super::super::runtime::operator::Operator;
use super::super::common::matmul_params::MatMulParams;
use super::super::mem_mgr::cache::Cache;
// use super::super::mem_mgr::model_loader::SafeTensorsLoader;
// use super::super::ptensor::linear::Linear;
use super::super::runtime::tensor::{Tensor, TensorCtx};
use super::decoder_layer::DecoderLayer;
// use crate::runtime::inference::state::TokenRecord;

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
    pub ctx: Rc<TensorCtx<T>>,
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
        + Sigmoid
        + Sqrt
        + FromNumber
        + AddAssign
        + Send
        + Sync,
{
    pub fn new(
        config: &Config,
        position_vec: Vec<T>,
        sequence_length: usize,
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
        let ctx = Rc::new(TensorCtx::new(cache, operator_queue));

        // Create default tensors
        let word_embedding = Rc::new(ctx.zeros(
            vec![config.vocab_size, config.hidden_size],
            String::from("model.embed_tokens.weight"),
        ));

        let position_embedding = Rc::new(ctx.tensor_from_vec(
            vec![config.max_position_embeddings, 1, 1, config.head_dim],
            position_vec,
            String::from("model.position_embedding.weight"),
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
                ctx.clone(),
            ));
        }

        Self {
            // sequences: vec![0; (config.max_position_embeddings + 1) * batch_size],
            word_embedding: word_embedding.clone(),
            position_embedding: position_embedding.clone(),
            lm_head_weight: ctx.zeros(
                vec![config.vocab_size, config.hidden_size],
                String::from("lm_head.weight"),
            ),
            layers: layers,
            batch_size: batch_size,
            hidden_size: config.hidden_size,
            // sequence_chunk_size: sequence_chunk_size,
            topk_size: topk_size,
            rms_norm_eps: T::from_f32(config.rms_norm_eps),
            scope_name: scope_name,
            ctx: ctx,
        }
    }

    pub fn forward(
        &mut self,
        input_sequences: *mut usize,
        eos_id: usize,
    ) -> (*const usize, Tensor<T>) {
        // -> Tensor<T> {
        // let sequences = vec![0; (self.config.max_position_embeddings + 1) * self.config.batch_size].into_boxed_slice();

        let mut hidden_state = self.ctx.zeros(
            vec![self.batch_size, self.hidden_size],
            format!("{}.hidden_state.output", self.scope_name),
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
            input_sequences,
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
    // use crate::common::config::Config;
    // use crate::llama::model_loader::SafeTensorsLoader;
    use crate::runtime::inference::state::{Phase, SequenceState};
    use crate::moe::rope::precompute_freqs_cis_t;
    use crate::mem_mgr::allocator::allocate_init;
    use crate::runtime::tensor::Tensor;

    #[test]
    fn test_model_forward() {
        // let cpu_num =  thread::available_parallelism().unwrap().get();
        let sequence_length = 128;
        let sequence_chunk_size = 1;
        let batch_size = 3;
        let topk_size = 8;

        let config =
            Config::load_from_file(r"models/Qwen3-Coder-30B-A3B-Instruct/config.json").unwrap();

        let position_vec = precompute_freqs_cis_t::<f32>(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta as f32,
        );
        let mut model = Model::<f32>::new(
            &config,
            position_vec,
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

        let batch_records: Vec<SequenceState> = (0..batch_size)
            .map(|i| SequenceState {
                sequence_index: i,
                kv_index: i,
                phase: Phase::Decode,
                // prompt_length: i,
                notify: std::sync::Arc::new(tokio::sync::Notify::new()),
            })
            .collect();

        let mut batch_list = batch_records;
        let eos_id = 151643;

        let (output_indices, output_tensor) = unsafe { model.forward(sequences, eos_id) };

        let thread_num: usize = num_cpus::get();
        for operator in model.ctx.operator_queue.borrow().iter() {
            for i in 0..thread_num {
                operator.run(batch_size, 0, thread_num, i, &[], &[], &mut Vec::new());
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

        let position_vec = precompute_freqs_cis_t::<f16>(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta as f32,
        );
        let mut model = Model::<f16>::new(
            &config,
            position_vec,
            sequence_length,
            // sequence_chunk_size,
            batch_size,
            topk_size,
        );

        let mut sequences =
            allocate_init::<usize>((config.max_position_embeddings + 1) * batch_size, 0);
        let batch_records: Vec<SequenceState> = (0..batch_size)
            .map(|i| SequenceState {
                sequence_index: i,
                kv_index: i,
                phase: Phase::Decode,
                // prompt_length: i,
                notify: std::sync::Arc::new(tokio::sync::Notify::new()),
            })
            .collect();

        let mut batch_list = batch_records;
        let eos_id = 151643;

        let (output_indices, output_tensor) = unsafe { model.forward(sequences, eos_id) };

        let thread_num = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        for operator in model.ctx.operator_queue.borrow().iter() {
            for i in 0..thread_num {
                operator.run(batch_size, 0, thread_num, i, &[], &[], &mut Vec::new());
            }
        }
    }
}








