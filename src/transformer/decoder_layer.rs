use std::ops::{AddAssign, Neg, Sub};
use std::rc::Rc;

use crate::common::num_traits::FromNumber;
use crate::common::num_traits::Sigmoid;
use crate::common::num_traits::Sqrt;
use crate::common::num_traits::{exp::Exp, neg_infinity::NegInfinity};

use super::super::runtime::tensor::{Tensor, TensorCtx};
use super::attention::Attention;
use super::config::{AttentionKind, Config, FfnKind};
use super::mlp::MLP;
use super::names::{layer_tensor_names, FfnTensorNames};
use super::sparse_moe_block::SparseMoeBlock;

pub enum AttentionBlock<T>
where
    T: Copy + PartialOrd,
{
    Full(Attention<T>),
    SlidingWindow(Attention<T>),
}

pub enum FfnBlock<T>
where
    T: Copy + PartialOrd,
{
    Dense(MLP<T>),
    SparseMoe(SparseMoeBlock<T>),
}

// #[derive(Clone)]
pub struct DecoderLayer<T>
where
    T: Copy + PartialOrd,
{
    sequence_length: usize,
    // sequence_chunk_size: usize,
    batch_size: usize,
    hidden_size: usize,
    head_dim: usize,
    rms_norm_eps: T,
    layer_idx: usize,
    word_embedding: Rc<Tensor<T>>,
    position_embedding: Rc<Tensor<T>>,
    self_attention: AttentionBlock<T>,
    ffn_block: FfnBlock<T>,
    scope_name: String,
    ctx: Rc<TensorCtx<T>>,
}

impl<T> DecoderLayer<T>
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
        layer_idx: usize,
        // sequences: *mut usize,
        _sequence_length: usize,
        batch_size: usize,
        word_embedding: Rc<Tensor<T>>,
        position_embedding: Rc<Tensor<T>>,
        _parent_scope_name: &str,
        ctx: Rc<TensorCtx<T>>,
    ) -> Self {
        let names = layer_tensor_names(config, layer_idx);
        let self_attention = match config.layers[layer_idx].attention {
            AttentionKind::Full => AttentionBlock::Full(Attention::<T>::new(
                config,
                names.attention.clone(),
                ctx.clone(),
            )),
            AttentionKind::SlidingWindow => AttentionBlock::SlidingWindow(Attention::<T>::new(
                config,
                names.attention.clone(),
                ctx.clone(),
            )),
            AttentionKind::Linear => panic!("linear attention is not implemented for layer {}", layer_idx),
        };

        let ffn_block = match (&config.layers[layer_idx].ffn, names.ffn) {
            (FfnKind::Dense { intermediate_size }, FfnTensorNames::Dense(ffn_names)) => {
                FfnBlock::Dense(MLP::new(
                    config.hidden_size,
                    *intermediate_size,
                    ffn_names,
                    ctx.clone(),
                ))
            }
            (
                FfnKind::SparseMoe {
                    intermediate_size,
                    num_experts,
                    num_experts_per_tok,
                    norm_topk_prob,
                },
                FfnTensorNames::SparseMoe(ffn_names),
            ) => FfnBlock::SparseMoe(SparseMoeBlock::new(
                config.hidden_size,
                *intermediate_size,
                *num_experts,
                *num_experts_per_tok,
                *norm_topk_prob,
                ffn_names,
                ctx.clone(),
            )),
            _ => unreachable!("ffn plan and names must match"),
        };

        Self {
            sequence_length: config.max_position_embeddings,
            batch_size: batch_size,
            hidden_size: config.hidden_size,
            head_dim: config.head_dim,
            rms_norm_eps: T::from_f32(config.rms_norm_eps),
            layer_idx: layer_idx,
            self_attention,
            ffn_block,
            word_embedding: word_embedding,
            position_embedding: position_embedding,
            ctx: ctx,
            scope_name: names.scope,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<T>,
        input_sequences: *mut usize,
        decode_only_flag: bool,
        _tensor_name: String,
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
            self.ctx.lookup_rms(
                input_sequences,
                &*self.word_embedding,
                self.batch_size,
                self.rms_norm_eps,
                self.scope_name.clone(),
            )
        };
        let hidden_states = &hidden_states_owned;

        let attention_hidden_states = match &self.self_attention {
            AttentionBlock::Full(attention) | AttentionBlock::SlidingWindow(attention) => {
                attention.forward(
                    &norm_hidden,
                    hidden_states,
                    &*self.position_embedding,
                    decode_only_flag,
                )
            }
        };

        let norm_hidden_states = attention_hidden_states.rms(
            // self.layernorm_weight.data,
            self.rms_norm_eps,
            decode_only_flag,
            format!("{}.norm_hidden2", self.scope_name),
        );

        let output_hidden_states = match &self.ffn_block {
            FfnBlock::Dense(mlp) => mlp.forward(
                &norm_hidden_states,
                &attention_hidden_states,
                format!("{}.attention_hidden3", self.scope_name),
            ),
            FfnBlock::SparseMoe(sparse_moe_block) => sparse_moe_block.forward(
                &norm_hidden_states,
                &attention_hidden_states,
                decode_only_flag,
                format!("{}.attention_hidden3", self.scope_name),
            ),
        };

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
        );*/
        output_hidden_states
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::mem_mgr::cache::Cache;
    use std::cell::RefCell;
    // use std::slice;

    #[test]
    fn test_decoder_layer_f32() {
        let sequence_chunk_size = 1;
        let batch_size = 6;

        let config =
            Config::load_from_file(r"models/Qwen3-Coder-30B-A3B-Instruct/config.json").unwrap();

        let hidden_size = config.hidden_size;
        let max_position_embeddings = config.max_position_embeddings;
        let head_dim = config.head_dim;

        let cache = Rc::new(RefCell::new(Cache::new(std::collections::HashMap::new())));
        let operator_queue = Rc::new(RefCell::new(Vec::new()));
        let ctx = Rc::new(TensorCtx::new(cache, operator_queue));

        let vocab_size = config.vocab_size;
        let word_embedding = Rc::new(ctx.zeros(
            vec![vocab_size, hidden_size],
            String::from("model.embed_tokens.weight"),
        ));
        let position_embedding = Rc::new(ctx.zeros(
            vec![max_position_embeddings, 1, 1, head_dim],
            String::from("model.position_embedding.weight"),
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
            ctx.clone(),
        );

        let shape = vec![batch_size, hidden_size];
        let input = ctx.tensor(
            shape.clone(),
            String::from("model.layers.1.input_tensor"),
        );

        for i in 0..input.shape.iter().product() {
            unsafe {
                input.data.add(i).write(1.0);
            }
        }

        let mut sequences = vec![0; sequence_chunk_size * batch_size];
        let output_tensor = layer.forward(
            &input,
            sequences.as_mut_ptr(),
            false,
            String::from("model.layers.1.output_tensor"),
        );

        // Validate output shape
        debug_assert_eq!(output_tensor.shape, vec![batch_size, hidden_size]);

        // Execute the operator queue
        let thread_num = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        for (index, operator) in output_tensor.operator_queue.borrow().iter().enumerate() {
            println!("operator {} in queue", index);
            for i in 0..thread_num {
                operator.run(batch_size, 0, thread_num, i, &[], &[], &mut Vec::new());
            }
        }

        assert_eq!(output_tensor.shape, vec![batch_size, hidden_size]);
    }

    #[test]
    fn test_decoder_layer_f16() {
        let position_window_size = 1;
        let batch_size = 3;

        let config =
            Config::load_from_file(r"models/Qwen3-Coder-30B-A3B-Instruct/config.json").unwrap();

        let sequence_chunk_size = position_window_size;
        let hidden_size = config.hidden_size;
        let max_position_embeddings = config.max_position_embeddings;
        let head_dim = config.head_dim;

        let cache: Rc<RefCell<Cache<f16>>> =
            Rc::new(RefCell::new(Cache::new(std::collections::HashMap::new())));
        let operator_queue = Rc::new(RefCell::new(Vec::new()));
        let ctx = Rc::new(TensorCtx::new(cache, operator_queue));

        let vocab_size = config.vocab_size;
        let word_embedding = Rc::new(ctx.zeros(
            vec![vocab_size, hidden_size],
            String::from("model.embed_tokens.weight"),
        ));
        let position_embedding = Rc::new(ctx.zeros(
            vec![max_position_embeddings, 1, 1, head_dim],
            String::from("model.position_embedding.weight"),
        ));

        let layer = DecoderLayer::<f16>::new(
            &config,
            0,
            max_position_embeddings,
            batch_size,
            word_embedding.clone(),
            position_embedding.clone(),
            "model",
            ctx.clone(),
        );

        let shape = vec![position_window_size, batch_size, hidden_size];
        let input = ctx.tensor(
            shape.clone(),
            String::from("model.layers.0.input_tensor"),
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
            false,
            String::from("model.layers.0.output_tensor"),
        );

        // Validate output shape
        debug_assert_eq!(
            output_tensor.shape,
            vec![position_window_size, batch_size, hidden_size]
        );

        // Execute the operator queue
        let thread_num = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        for (index, operator) in output_tensor.operator_queue.borrow().iter().enumerate() {
            println!("operator {} in queue", index);
            for i in 0..thread_num {
                operator.run(
                    sequence_chunk_size * batch_size,
                    sequence_chunk_size,
                    thread_num,
                    i,
                    &[],
                    &[],
                    &mut Vec::new(),
                );
            }
        }

        assert_eq!(
            output_tensor.shape,
            vec![position_window_size, batch_size, hidden_size]
        );
    }
}





