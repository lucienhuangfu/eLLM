use std::ops::{AddAssign, Neg, Sub};
use std::rc::Rc;

// use serde::{Deserialize, Serialize};
// use hurdles::Barrier;
// use super::barrier::Barrier;
// use serde::{Deserialize, Serialize};

use super::config::Config;
use super::names::model_tensor_names;
use crate::num_traits::FromNumber;
use crate::num_traits::NegInfinity;
use crate::num_traits::{Exp, Sigmoid, Sqrt};

// use super::super::operators::map::rms_map::RMSMap;
use crate::kernel::common::matmul_params::MatMulParams;
use crate::mem_mgr::mem_pool::GlobalMemPool;
// use super::super::mem_mgr::model_loader::SafeTensorsLoader;
// use super::super::ptensor::linear::Linear;
use super::decoder_layer::DecoderLayer;
use crate::tensor::{GlobalOperatorQueue, Tensor};
// use crate::runtime::inference::state::TokenRecord;

#[cfg(test)]
use super::rope::RotaryEmbedding;

// #[derive(Clone)]
pub struct Model<T>
where
    T: Copy + PartialOrd,
{
    lm_head_weight: Tensor<T>,
    norm_weight: Tensor<T>,
    pub layers: Vec<DecoderLayer<T>>,
    rms_norm_eps: T,
    pub chunk_size: usize,
    pub sequence_length: usize,
    pub batch_size: usize,
    pub hidden_size: usize,
    pub topk_size: usize,
    pub top_p: T,
    pub min_p: T,
    pub do_sample: bool,
    pub eos_id: usize,
    pub eos_ids: Vec<usize>,
    scope_name: String,
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
        + Sync
        + GlobalMemPool
        + GlobalOperatorQueue,
{
    /// Backward-compatible constructor (greedy, no sampling).
    pub fn new(
        config: &Config,
        position_vec: Vec<T>,
        chunk_size: usize,
        sequence_length: usize,
        batch_size: usize,
        topk_size: usize,
        eos_ids: Vec<usize>,
    ) -> Self {
        Self::with_sampling(
            config,
            position_vec,
            chunk_size,
            sequence_length,
            batch_size,
            topk_size,
            T::from_f32(1.0),
            T::default(),
            false,
            eos_ids,
        )
    }

    /// Full constructor with sampling parameters.
    pub fn with_sampling(
        config: &Config,
        position_vec: Vec<T>,
        chunk_size: usize,
        sequence_length: usize,
        batch_size: usize,
        topk_size: usize,
        top_p: T,
        min_p: T,
        do_sample: bool,
        eos_ids: Vec<usize>,
    ) -> Self {
        let model_names = model_tensor_names(config);
        let scope_name = model_names.scope.clone();

        T::init_operator_queue();

        // Create default tensors
        let word_embedding = Rc::new(Tensor::zeros(
            vec![config.vocab_size, config.hidden_size],
            model_names.token_embedding.clone(),
        ));

        let position_embedding = Rc::new(Tensor::from_vec(
            vec![config.max_position_embeddings, 1, 1, config.head_dim],
            position_vec,
            model_names.position_embedding.clone(),
        ));

        let mut layers: Vec<DecoderLayer<T>> = Vec::new();
        for i in 0..config.layers.len() {
            layers.push(DecoderLayer::<T>::new(
                &config,
                i,
                chunk_size,
                sequence_length,
                batch_size,
                word_embedding.clone(),
                position_embedding.clone(),
                &scope_name.clone(),
            ));
        }

        let eos_id = eos_ids.first().copied().unwrap_or(0);

        Self {
            lm_head_weight: Tensor::zeros(
                vec![config.vocab_size, config.hidden_size],
                model_names.lm_head.clone(),
            ),
            norm_weight: Tensor::zeros(vec![config.hidden_size], model_names.norm_weight.clone()),
            layers: layers,
            chunk_size: chunk_size,
            sequence_length: sequence_length,
            batch_size: batch_size,
            hidden_size: config.hidden_size,
            topk_size: topk_size,
            top_p,
            min_p,
            do_sample,
            eos_id: eos_id,
            eos_ids,
            rms_norm_eps: T::from_f32(config.rms_norm_eps),
            scope_name: scope_name,
        }
    }

    pub fn forward(
        &mut self,
        input_sequences: *mut usize,
        batch_temperature: *mut T,
    ) -> (*const usize, Tensor<T>) {
        // -> Tensor<T> {
        // let sequences = vec![0; (self.config.max_position_embeddings + 1) * self.config.batch_size].into_boxed_slice();

        let mut hidden_state = Tensor::zeros(
            vec![self.chunk_size, self.hidden_size],
            format!("{}.hidden_state.output", self.scope_name),
        );

        let trace_alignment = std::env::var_os("ELLM_ALIGN_TRACE").is_some();

        for (i, layer_module) in self.layers.iter().enumerate() {
            if trace_alignment {
                eprintln!("building layer {i}");
            }
            let decode_only_flag = i == (self.layers.len() - 1);

            hidden_state = layer_module.forward(
                &hidden_state,
                input_sequences,
                decode_only_flag,
                format!("{}.hidden_states.{}.output", self.scope_name, i),
            );
            // all_hidden_states.push(hidden_states);
        }

        if trace_alignment {
            eprintln!("building final norm");
        }
        let norm_state = hidden_state.rms(
            &self.norm_weight,
            self.rms_norm_eps,
            false,
            format!("{}.norm_hidden", self.scope_name),
        );

        // Lift: copy last prefill token's norm to batch position, so MatMulTopK
        // processes the correct token during both prefill and decode.
        norm_state.lift_vector();

        if trace_alignment {
            eprintln!("building lm_head/topk");
        }
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
            batch_temperature,
            self.sequence_length,
            self.topk_size,
            self.top_p,
            self.min_p,
            self.do_sample,
            self.eos_ids.clone(),
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
    use crate::mem_mgr::allocator::AlignedBox;
    use crate::runtime::sequence_slice::SequenceSlice;
    use crate::runtime::{Phase, SequenceState};
    use std::collections::HashMap;

    fn build_batch_list(batch_size: usize) -> Vec<SequenceState> {
        (0..batch_size)
            .map(|i| SequenceState {
                filling_length: 0,
                sequence_index: i,
                kv_index: i,
                phase: Phase::Decode,
                notify: std::sync::Arc::new(tokio::sync::Notify::new()),
            })
            .collect()
    }

    fn build_decode_list(batch_size: usize) -> Vec<SequenceSlice> {
        (0..batch_size)
            .map(|batch_index| SequenceSlice {
                batch_index,
                sequence_index: batch_index,
                token_start_index: batch_index,
                length: 1,
                last_token_flag: true,
            })
            .collect()
    }

    fn build_prefill_list(batch_size: usize) -> Vec<Vec<SequenceSlice>> {
        vec![{
            (0..batch_size)
                .map(|batch_index| SequenceSlice {
                    batch_index,
                    sequence_index: batch_index,
                    token_start_index: batch_index,
                    length: 1,
                    last_token_flag: false,
                })
                .collect()
        }]
    }

    #[test]
    #[ignore = "model-scale integration test; run manually on a large machine"]
    fn test_model_forward() {
        // let cpu_num =  thread::available_parallelism().unwrap().get();
        let sequence_length = 128;
        let batch_size = 3;
        let topk_size = 8;
        let thread_num = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let config =
            Config::load_from_file(r"models/Qwen3-Coder-30B-A3B-Instruct/config.json").unwrap();

        let position_vec = RotaryEmbedding::new(
            config.head_dim,
            config.rotary_dim,
            config.max_position_embeddings,
            config.rope_theta as f32,
            config.rope_scaling.clone(),
        )
        .forward::<f32>();
        let eos_id = 151643;
        f32::init_global(HashMap::new());
        let mut batch_temperature = vec![1.0f32; batch_size];
        let mut model = Model::<f32>::new(
            &config,
            position_vec,
            sequence_length, // chunk_size
            sequence_length, // sequence_length
            batch_size,
            topk_size,
            vec![eos_id],
        );

        // let mut sequences: Vec<usize> = vec![0; (config.max_position_embeddings + 1)*config.batch_size];
        let mut sequences_box =
            AlignedBox::allocate_init((config.max_position_embeddings) * batch_size, 0);

        let mut batch_list = build_batch_list(batch_size);
        let prefill_list = build_prefill_list(batch_size);
        let decode_list = build_decode_list(batch_size);

        let (_output_indices, _output_tensor) =
            model.forward(sequences_box.as_mut_ptr(), batch_temperature.as_mut_ptr());

        f32::with_operator_queue(|queue| {
            for thread_id in 0..thread_num {
                for operator in queue.iter() {
                    operator.run(
                        batch_size,
                        1,
                        thread_num,
                        thread_id,
                        &prefill_list,
                        &decode_list,
                        &mut batch_list,
                    );
                }
            }
        });

        assert_eq!(batch_list.len(), batch_size);
        assert!(batch_list
            .iter()
            .all(|record| matches!(record.phase, Phase::Decode)));

        // Add assertions to verify the output_tensor
        // For example:
        // assert_eq!(output_tensor.shape, vec![config.batch_size, config.hidden_size]);
    }

    #[test]
    #[ignore = "model-scale integration test; run manually on a large machine"]
    fn test_model_forward_f16() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip test_model_forward_f16: avx512fp16 not detected");
            return;
        }

        let sequence_length = 128;
        let batch_size = 3;
        let topk_size = 8;
        let thread_num = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let config =
            Config::load_from_file(r"models/Qwen3-Coder-30B-A3B-Instruct/config.json").unwrap();

        let position_vec = RotaryEmbedding::new(
            config.head_dim,
            config.rotary_dim,
            config.max_position_embeddings,
            config.rope_theta as f32,
            config.rope_scaling.clone(),
        )
        .forward::<f16>();
        let eos_id = 151643;
        f16::init_global(HashMap::new());
        let mut batch_temperature = vec![1.0f16; batch_size];
        let mut model = Model::<f16>::new(
            &config,
            position_vec,
            sequence_length, // chunk_size
            sequence_length, // sequence_length
            batch_size,
            topk_size,
            vec![eos_id],
        );

        let mut sequences_box =
            AlignedBox::allocate_init((config.max_position_embeddings) * batch_size, 0);
        let mut batch_list = build_batch_list(batch_size);
        let prefill_list = build_prefill_list(batch_size);
        let decode_list = build_decode_list(batch_size);

        let (_output_indices, _output_tensor) =
            model.forward(sequences_box.as_mut_ptr(), batch_temperature.as_mut_ptr());

        f16::with_operator_queue(|queue| {
            for thread_id in 0..thread_num {
                for operator in queue.iter() {
                    operator.run(
                        batch_size,
                        1,
                        thread_num,
                        thread_id,
                        &prefill_list,
                        &decode_list,
                        &mut batch_list,
                    );
                }
            }
        });

        assert_eq!(batch_list.len(), batch_size);
        assert!(batch_list
            .iter()
            .all(|record| matches!(record.phase, Phase::Decode)));
    }

    #[test]
    fn test_qwen3_06b_creation() {
        let sequence_length = 128;
        let batch_size = 1;
        let topk_size = 8;

        let config_path = r"models/Qwen3-0.6B/config.json";
        if !std::path::Path::new(config_path).exists() {
            eprintln!("skip test_qwen3_06b_creation: {config_path} not found");
            return;
        }

        let config = Config::load_from_file(config_path).unwrap();

        let position_vec = RotaryEmbedding::new(
            config.head_dim,
            config.rotary_dim,
            config.max_position_embeddings,
            config.rope_theta as f32,
            config.rope_scaling.clone(),
        )
        .forward::<f32>();

        let eos_id = config.eos_token_id;
        // Initialize global mem pool with dummy parameters instead of empty
        let mut params = HashMap::new();
        // Just put a dummy small vec for one key so parameters isn't empty
        params.insert("dummy.key".to_string(), vec![0.0f32; 10]);
        f32::init_global(params);

        let model = Model::<f32>::new(
            &config,
            position_vec,
            sequence_length, // chunk_size
            sequence_length, // sequence_length
            batch_size,
            topk_size,
            vec![eos_id],
        );

        assert_eq!(model.layers.len(), config.num_hidden_layers);
        assert_eq!(model.hidden_size, config.hidden_size);
        assert_eq!(model.batch_size, batch_size);
        assert_eq!(model.topk_size, topk_size);
        assert_eq!(model.eos_id, eos_id);
    }

    #[test]
    fn test_qwen3_06b_creation_f16() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip test_qwen3_06b_creation_f16: avx512fp16 not detected");
            return;
        }
        let sequence_length = 128;
        let batch_size = 1;
        let topk_size = 8;

        let config_path = r"models/Qwen3-0.6B/config.json";
        if !std::path::Path::new(config_path).exists() {
            eprintln!("skip test_qwen3_06b_creation_f16: {config_path} not found");
            return;
        }

        let config = Config::load_from_file(config_path).unwrap();

        let position_vec = RotaryEmbedding::new(
            config.head_dim,
            config.rotary_dim,
            config.max_position_embeddings,
            config.rope_theta as f32,
            config.rope_scaling.clone(),
        )
        .forward::<f16>();

        let eos_id = config.eos_token_id;
        // Initialize global mem pool with empty parameters
        f16::init_global(HashMap::new());

        let model = Model::<f16>::new(
            &config,
            position_vec,
            sequence_length, // chunk_size
            sequence_length, // sequence_length
            batch_size,
            topk_size,
            vec![eos_id],
        );

        assert_eq!(model.layers.len(), config.num_hidden_layers);
        assert_eq!(model.hidden_size, config.hidden_size);
        assert_eq!(model.batch_size, batch_size);
        assert_eq!(model.topk_size, topk_size);
        assert_eq!(model.eos_id, eos_id);
    }
}
