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

const DEFAULT_PROMPTS: &[&str] = &[
    "你好，请用一句话介绍 Rust。",
    "What is the capital of France?",
    "Write a short poem about programming.",
    "解释一下什么是机器学习。",
    "How does a CPU cache work?",
    "用 Python 写一个快速排序算法。",
    "Tell me a joke about computers.",
    "什么是深度学习中的注意力机制？",
    "Explain the difference between stack and heap.",
    "Write a haiku about debugging.",
];

fn main() -> anyhow::Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/Qwen3-0.6B".to_string());
    let max_tokens: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "8".to_string())
        .parse()
        .unwrap_or(8);
    let batch_size: usize = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "2".to_string())
        .parse()
        .unwrap_or(2);

    let config = Config::load_from_file(format!("{model_dir}/config.json"))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let generation_config =
        GenerationConfig::load_from_file(format!("{model_dir}/generation_config.json")).ok();

    let tokenizer_path = format!("{model_dir}/tokenizer.json");
    let tokenizer_config_path = format!("{model_dir}/tokenizer_config.json");
    let chat_template_path = format!("{model_dir}/chat_template.jinja");

    let chat_template = ChatTemplate::from_model_files(&chat_template_path, &tokenizer_config_path)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let tokenizer =
        load_tiktoken(&tokenizer_path, &tokenizer_config_path).map_err(|e| anyhow::anyhow!(e))?;

    // Read hidden_size from config for correct norm tensor sizing
    let hidden_size = config.hidden_size;

    // Select prompts based on batch_size
    let prompts: Vec<&str> = DEFAULT_PROMPTS
        .iter()
        .take(batch_size)
        .copied()
        .collect();

    let mut all_input_ids: Vec<Vec<u32>> = vec![];
    let mut all_prompts: Vec<String> = vec![];
    for prompt in &prompts {
        let messages = [("user", *prompt)];
        let rendered = chat_template
            .apply_chat_template(&messages, true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let ids = tokenizer.encode_with_special_tokens(&rendered);
        eprintln!("Prompt '{}': {} tokens", prompt, ids.len());
        all_input_ids.push(ids);
        all_prompts.push(rendered);
    }

    // Use the max input length for everything
    let max_input_len = all_input_ids.iter().map(|ids| ids.len()).max().unwrap();
    let total_input: usize = all_input_ids.iter().map(|ids| ids.len()).sum();
    let sequence_length = max_input_len + max_tokens + 8;
    let chunk_size = total_input + batch_size * max_tokens;
    let top_k = 1;

    eprintln!(
        "Batch size: {batch_size}, max input len: {max_input_len}, seq_len: {sequence_length}, hidden_size: {hidden_size}"
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
        top_k,
        eos_ids,
    );

    // Fill sequences array
    let sequences = AlignedBox::allocate_init(sequence_length * batch_size, 0usize);
    for (batch_idx, ids) in all_input_ids.iter().enumerate() {
        for (pos, token_id) in ids.iter().enumerate() {
            unsafe {
                *sequences
                    .as_mut_ptr()
                    .add(batch_idx * sequence_length + pos) = *token_id as usize;
            }
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

    // Build prefill/decode lists for each batch entry.
    // Uses cumulative token offsets so batch tokens are contiguous in chunk space.
    let mut prefill_slices_per_thread: Vec<Vec<SequenceSlice>> = vec![vec![]];
    let mut decode_slices: Vec<SequenceSlice> = vec![];
    let mut batch_list = Vec::new();
    let mut token_offset = 0usize;

    for batch_idx in 0..batch_size {
        let input_len = all_input_ids[batch_idx].len();
        let slice = SequenceSlice {
            batch_index: batch_idx,
            sequence_index: 0,
            token_start_index: token_offset,
            length: input_len,
            last_token_flag: true,
        };
        prefill_slices_per_thread[0].push(slice.clone());
        decode_slices.push(slice);
        batch_list.push(SequenceState {
            sequence_index: 0,
            kv_index: 0,
            filling_length: input_len,
            phase: Phase::Prefill,
            notify: Arc::new(tokio::sync::Notify::new()),
        });
        token_offset += input_len;
    }

    let prefill_list = prefill_slices_per_thread;
    let decode_list = decode_slices;
    let total_prefill_tokens: usize = batch_list.iter().map(|s| s.filling_length).sum();
    let mut generated_tokens: Vec<Vec<usize>> = vec![vec![]; batch_size];

    let thread_num = 1; // Single-threaded for alignment

    // ============ Prefill Phase 1 ============
    for (_index, operator) in operator_queue.iter().enumerate().take(phase2_start) {
        operator.run(
            total_prefill_tokens,
            batch_size,
            thread_num,
            0,
            &prefill_list,
            &decode_list,
            &mut batch_list,
        );
    }

    // Manual lift: copy last token's norm to decode position (batch_idx) for each batch.
    // Uses hidden_size from config instead of hardcoded value.
    f16::with_global(|pool| {
        let norm_total = chunk_size * hidden_size;
        let norm_ptr = pool.get("model.norm_hidden.output", &vec![norm_total]);
        let mut offset = 0usize;
        for batch_idx in 0..batch_size {
            let input_len = all_input_ids[batch_idx].len();
            let src_offset = (offset + input_len - 1) * hidden_size;
            let dst_offset = batch_idx * hidden_size;
            unsafe {
                std::ptr::copy(
                    norm_ptr.add(src_offset),
                    norm_ptr.add(dst_offset),
                    hidden_size,
                );
            }
            offset += input_len;
        }
    });

    // Phase 2
    for (_index, operator) in operator_queue.iter().enumerate().skip(phase2_start) {
        operator.run(
            0,
            batch_size,
            thread_num,
            0,
            &prefill_list,
            &decode_list,
            &mut batch_list,
        );
    }

    // Read first tokens
    for batch_idx in 0..batch_size {
        let input_len = all_input_ids[batch_idx].len();
        let token = unsafe {
            *sequences
                .as_mut_ptr()
                .add(batch_idx * sequence_length + input_len)
        };
        generated_tokens[batch_idx].push(token);
        eprintln!(
            "Batch {batch_idx} Token 1: {token} ({})",
            tokenizer.decode(vec![token as u32]).unwrap_or_default()
        );
    }

    // ============ Decode ============
    for step in 1..max_tokens {
        let all_eos = batch_list.iter().all(|s| matches!(s.phase, Phase::Eos));
        if all_eos {
            break;
        }

        let mut step_decode_slices: Vec<SequenceSlice> = vec![];
        for batch_idx in 0..batch_size {
            if matches!(batch_list[batch_idx].phase, Phase::Eos) {
                continue;
            }
            step_decode_slices.push(SequenceSlice {
                batch_index: batch_idx,
                sequence_index: batch_list[batch_idx].sequence_index,
                token_start_index: batch_idx,
                length: 1,
                last_token_flag: true,
            });
        }
        let decode_count = step_decode_slices.len();
        if decode_count == 0 {
            break;
        }
        let empty_prefill: Vec<Vec<SequenceSlice>> = vec![];

        for (_index, operator) in operator_queue.iter().enumerate().take(phase2_start) {
            operator.run(
                0,
                decode_count,
                thread_num,
                0,
                &empty_prefill,
                &step_decode_slices,
                &mut batch_list,
            );
        }

        for (_index, operator) in operator_queue.iter().enumerate().skip(phase2_start) {
            operator.run(
                0,
                decode_count,
                thread_num,
                0,
                &empty_prefill,
                &step_decode_slices,
                &mut batch_list,
            );
        }

        for batch_idx in 0..batch_size {
            if matches!(batch_list[batch_idx].phase, Phase::Eos) {
                continue;
            }
            let token = unsafe {
                *sequences
                    .as_mut_ptr()
                    .add(batch_list[batch_idx].sequence_index)
            };
            generated_tokens[batch_idx].push(token);
            let decoded = tokenizer.decode(vec![token as u32]).unwrap_or_default();
            eprintln!("Batch {batch_idx} Token {}: {token} ({decoded})", step + 1);
        }
    }

    // Build per-sequence results
    let mut results = Vec::new();
    for batch_idx in 0..batch_size {
        let decoded_text = tokenizer
            .decode(
                generated_tokens[batch_idx]
                    .iter()
                    .map(|t| *t as u32)
                    .collect::<Vec<_>>(),
            )
            .unwrap_or_default();
        results.push(json!({
            "batch_index": batch_idx,
            "prompt": prompts[batch_idx],
            "input_ids": all_input_ids[batch_idx],
            "generated_token_ids": generated_tokens[batch_idx],
            "generated_text": decoded_text,
        }));
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "model_dir": model_dir,
            "batch_size": batch_size,
            "max_new_tokens": max_tokens,
            "hidden_size": hidden_size,
            "results": results,
        }))?
    );

    Ok(())
}
