#![feature(f16)]

use ellm::mem_mgr::allocator::AlignedBox;
use ellm::mem_mgr::mem_pool::GlobalMemPool;
use ellm::operators::operator::Operator;
use ellm::runtime::chat_template::ChatTemplate;
use ellm::runtime::model_loader::SafeTensorsLoader;
use ellm::runtime::sequence_slice::SequenceSlice;
use ellm::runtime::tokenizer_loader::load_tiktoken;
use ellm::runtime::{Config, GenerationConfig, Phase, SequenceState};
use ellm::tensor::GlobalOperatorQueue;
use ellm::transformer::model::Model;
use ellm::transformer::rope::RotaryEmbedding;
use serde_json::json;
use std::f16;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/Qwen3-0.6B".to_string());
    let max_tokens: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "10".to_string())
        .parse()
        .unwrap_or(10);

    let config = Config::load_from_file(format!("{model_dir}/config.json"))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let generation_config =
        GenerationConfig::load_from_file(format!("{model_dir}/generation_config.json")).ok();

    let tokenizer_path = format!("{model_dir}/tokenizer.json");
    let tokenizer_config_path = format!("{model_dir}/tokenizer_config.json");
    let chat_template_path = format!("{model_dir}/chat_template.jinja");

    let messages = [("user", "你好，请用一句话介绍 Rust。")];
    let chat_template = ChatTemplate::from_model_files(&chat_template_path, &tokenizer_config_path)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let prompt = chat_template
        .apply_chat_template(&messages, true)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let tokenizer =
        load_tiktoken(&tokenizer_path, &tokenizer_config_path).map_err(|e| anyhow::anyhow!(e))?;
    let input_ids = tokenizer.encode_with_special_tokens(&prompt);

    let sequence_length = input_ids.len() + max_tokens + 1;
    let batch_size = 1;
    let chunk_size = input_ids.len() + max_tokens;
    let top_k = 1;

    eprintln!(
        "Prompt tokens: {} | max output: {} | seq_len: {}",
        input_ids.len(),
        max_tokens,
        sequence_length
    );

    eprintln!("loading f16 weights");
    let params = SafeTensorsLoader::new(&model_dir)?.load_all_weights_f16()?;
    f16::init_global_strict(params);

    eprintln!("building rotary embeddings");
    let position_vec = RotaryEmbedding::new(
        config.head_dim,
        config.rotary_dim,
        config.max_position_embeddings,
        config.rope_theta as f32,
        config.rope_scaling.clone(),
    )
    .forward::<f16>();

    let eos_ids = generation_config
        .as_ref()
        .and_then(|g| g.eos_token_id_list.clone())
        .filter(|ids| !ids.is_empty())
        .unwrap_or(config.eos_token_ids.clone());

    eprintln!("building model graph");
    let mut model = Model::<f16>::new(
        &config,
        position_vec,
        chunk_size,
        sequence_length,
        batch_size,
        top_k,
        eos_ids.clone(),
    );

    let sequences = AlignedBox::allocate_init(sequence_length * batch_size, 0usize);
    for (index, token_id) in input_ids.iter().enumerate() {
        unsafe {
            *sequences.as_mut_ptr().add(index) = *token_id as usize;
        }
    }

    let mut batch_temperature = vec![1.0f16; batch_size];
    let (_output_indices, _output_tensor) =
        model.forward(sequences.as_mut_ptr(), batch_temperature.as_mut_ptr());
    let operator_queue = f16::take_operator_queue();

    let phase2_start = operator_queue
        .iter()
        .position(|op| matches!(op, Operator::MatMulTopK(_)))
        .unwrap_or(operator_queue.len());

    let prefill_len = input_ids.len();
    let prefill_slice = SequenceSlice {
        batch_index: 0,
        sequence_index: 0,
        token_start_index: 0,
        length: prefill_len,
        last_token_flag: true,
    };
    let prefill_list = vec![vec![prefill_slice.clone()]];
    let decode_list = vec![prefill_slice.clone()];

    let mut batch_list = vec![SequenceState {
        sequence_index: 0,
        kv_index: 0,
        filling_length: prefill_len,
        phase: Phase::Prefill,
        notify: Arc::new(tokio::sync::Notify::new()),
    }];

    let mut generated_tokens: Vec<usize> = vec![];

    // ============ Prefill ============
    for (index, operator) in operator_queue.iter().enumerate().take(phase2_start) {
        operator.run(
            prefill_len,
            1,
            1,
            0,
            &prefill_list,
            &decode_list,
            &mut batch_list,
        );
    }

    f16::with_global(|pool| {
        let norm_size = prefill_len * 1024;
        let norm_ptr = pool.get("model.norm_hidden.output", &vec![norm_size]);
        unsafe {
            std::ptr::copy(norm_ptr.add((prefill_len - 1) * 1024), norm_ptr, 1024);
        }
    });

    for (index, operator) in operator_queue.iter().enumerate().skip(phase2_start) {
        operator.run(0, 1, 1, 0, &prefill_list, &decode_list, &mut batch_list);
    }

    let token = unsafe { *sequences.as_mut_ptr().add(prefill_len) };
    generated_tokens.push(token);
    eprintln!(
        "Token 1: {token} ({})",
        tokenizer.decode(vec![token as u32]).unwrap_or_default()
    );

    // ============ Decode ============
    for step in 1..max_tokens {
        if matches!(batch_list[0].phase, Phase::Eos) {
            break;
        }

        let decode_slice = SequenceSlice {
            batch_index: 0,
            sequence_index: batch_list[0].sequence_index,
            token_start_index: 0,
            length: 1,
            last_token_flag: true,
        };
        let empty_prefill: Vec<Vec<SequenceSlice>> = vec![];

        for (index, operator) in operator_queue.iter().enumerate().take(phase2_start) {
            operator.run(
                0,
                1,
                1,
                0,
                &empty_prefill,
                &[decode_slice.clone()],
                &mut batch_list,
            );
        }

        for (index, operator) in operator_queue.iter().enumerate().skip(phase2_start) {
            operator.run(
                0,
                1,
                1,
                0,
                &empty_prefill,
                &[decode_slice.clone()],
                &mut batch_list,
            );
        }

        let token = unsafe { *sequences.as_mut_ptr().add(batch_list[0].sequence_index) };
        generated_tokens.push(token);

        let decoded = tokenizer.decode(vec![token as u32]).unwrap_or_default();
        eprintln!("Token {}: {token} ({decoded})", step + 1);

        if eos_ids.contains(&token) {
            break;
        }
    }

    let all_text = tokenizer
        .decode(
            generated_tokens
                .iter()
                .map(|t| *t as u32)
                .collect::<Vec<_>>(),
        )
        .unwrap_or_default();
    eprintln!("Generated: {all_text}");

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "model_dir": model_dir,
            "prompt": prompt,
            "input_ids": input_ids,
            "generated_token_ids": generated_tokens,
            "generated_text": all_text,
        }))?
    );

    Ok(())
}
