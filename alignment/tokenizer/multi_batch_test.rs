#![feature(f16)]

use ellm::common::sequence_slice::SequenceSlice;
use ellm::mem_mgr::allocator::AlignedBox;
use ellm::mem_mgr::mem_pool::GlobalMemPool;
use ellm::operators::operator::Operator;
use ellm::runtime::chat_template::ChatTemplate;
use ellm::runtime::model_loader::SafeTensorsLoader;
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
        .unwrap_or_else(|| "6".to_string())
        .parse()
        .unwrap_or(6);

    let config = Config::load_from_file(format!("{model_dir}/config.json"))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let generation_config =
        GenerationConfig::load_from_file(format!("{model_dir}/generation_config.json")).ok();

    let tokenizer_path = format!("{model_dir}/tokenizer.json");
    let tokenizer_config_path = format!("{model_dir}/tokenizer_config.json");
    let chat_template_path = format!("{model_dir}/chat_template.jinja");

    let chat_template =
        ChatTemplate::from_model_files(&chat_template_path, &tokenizer_config_path)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let tokenizer = load_tiktoken(&tokenizer_path, &tokenizer_config_path)
        .map_err(|e| anyhow::anyhow!(e))?;

    let messages_batch = [
        [("user", "hello")],
        [("user", "What is 2+2?")],
    ];

    let mut all_input_ids: Vec<Vec<u32>> = vec![];
    for messages in &messages_batch {
        let prompt = chat_template
            .apply_chat_template(messages, true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let ids = tokenizer.encode_with_special_tokens(&prompt);
        eprintln!("Prompt '{}': {} tokens", messages[0].1, ids.len());
        all_input_ids.push(ids);
    }

    let batch_size = messages_batch.len();
    let total_input_tokens: usize = all_input_ids.iter().map(|ids| ids.len()).sum();
    let max_input_len = all_input_ids.iter().map(|ids| ids.len()).max().unwrap();
    let sequence_length = max_input_len + max_tokens + 8;
    let chunk_size = total_input_tokens + batch_size * max_tokens;
    let top_k = 1;

    eprintln!("Batches={batch_size} total_tokens={total_input_tokens} seq_len={sequence_length} chunk={chunk_size}");

    eprintln!("loading f16 weights");
    let params = SafeTensorsLoader::new(&model_dir)?.load_all_weights_f16()?;
    f16::init_global_strict(params);

    eprintln!("building rotary embeddings");
    let position_vec = RotaryEmbedding::new(
        config.head_dim, config.rotary_dim, config.max_position_embeddings,
        config.rope_theta as f32, config.rope_scaling.clone(),
    ).forward::<f16>();

    let eos_ids = generation_config.as_ref()
        .and_then(GenerationConfig::eos_token_ids)
        .filter(|ids| !ids.is_empty())
        .unwrap_or(config.eos_token_ids);

    eprintln!("building model graph");
    let mut model = Model::<f16>::new(
        &config, position_vec, chunk_size, sequence_length,
        batch_size, top_k, eos_ids,
    );

    let sequences = AlignedBox::allocate_init(sequence_length * batch_size, 0usize);
    for (batch_idx, ids) in all_input_ids.iter().enumerate() {
        for (pos, token_id) in ids.iter().enumerate() {
            unsafe {
                *sequences.as_mut_ptr().add(batch_idx * sequence_length + pos) = *token_id as usize;
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

    // Build prefill lists: one group with all batch slices
    // token_start_index: position in contiguous hidden state buffer
    let mut prefill_group: Vec<SequenceSlice> = vec![];
    let mut decode_slices: Vec<SequenceSlice> = vec![];
    let mut batch_list: Vec<SequenceState> = vec![];
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
        prefill_group.push(slice.clone());
        decode_slices.push(slice.clone());
        batch_list.push(SequenceState {
            sequence_index: 0, kv_index: 0,
            filling_length: input_len, phase: Phase::Prefill,
            notify: Arc::new(tokio::sync::Notify::new()),
        });
        token_offset += input_len;
    }

    let prefill_list = vec![prefill_group];
    let prefill_size = total_input_tokens;
    let decode_size = batch_size;
    // Must match MatMulTopK's internal thread_max (detect_threads) for stride alignment
    let thread_num: usize = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).max(1);
    let mut generated_tokens: Vec<Vec<usize>> = vec![vec![]; batch_size];

    // ===== Prefill Phase 1: layers + norm =====
    for thread_id in 0..thread_num {
        for (_index, operator) in operator_queue.iter().enumerate().take(phase2_start) {
            operator.run(prefill_size, decode_size, thread_num, thread_id, &prefill_list, &decode_slices, &mut batch_list);
        }
    }

    // Manual lift: copy last token's norm to batch's start position
    f16::with_global(|pool| {
        let norm_size = chunk_size * 1024;
        let norm_ptr = pool.get("model.norm_hidden.output", &vec![norm_size]);
        let mut offset = 0usize;
        for batch_idx in 0..batch_size {
            let input_len = all_input_ids[batch_idx].len();
            let last_pos = offset + input_len - 1;
            let dst_pos = batch_idx;
            unsafe {
                std::ptr::copy(norm_ptr.add(last_pos * 1024), norm_ptr.add(dst_pos * 1024), 1024);
            }
            offset += input_len;
        }
    });

    // ===== Prefill Phase 2: MatMulTopK + TopKSoftmax =====
    // Use original decode_slices (with full prefill lengths) so TopKSoftmax
    // correctly transitions batch state from Prefill to Decode
    for thread_id in 0..thread_num {
        for (_index, operator) in operator_queue.iter().enumerate().skip(phase2_start) {
            operator.run(0, batch_size, thread_num, thread_id, &prefill_list, &decode_slices, &mut batch_list);
        }
    }

    // Read first tokens (written at sequences[batch * stride + input_len] by TopKSoftmax)
    for batch_idx in 0..batch_size {
        let input_len = all_input_ids[batch_idx].len();
        let tok = unsafe { *sequences.as_mut_ptr().add(batch_idx * sequence_length + input_len) };
        generated_tokens[batch_idx].push(tok);
        eprintln!("Batch {batch_idx} Token 1: {tok} ({})",
            tokenizer.decode(vec![tok as u32]).unwrap_or_default());
    }

    // ===== Decode Rounds =====
    for step in 1..max_tokens {
        let all_eos = batch_list.iter().all(|s| matches!(s.phase, Phase::Eos));
        if all_eos { break; }

        let mut step_decode_slices: Vec<SequenceSlice> = vec![];
        for batch_idx in 0..batch_size {
            if matches!(batch_list[batch_idx].phase, Phase::Eos) { continue; }
            step_decode_slices.push(SequenceSlice {
                batch_index: batch_idx,
                sequence_index: batch_list[batch_idx].sequence_index,
                token_start_index: batch_idx,
                length: 1,
                last_token_flag: true,
            });
        }
        let step_decode_size = step_decode_slices.len();
        if step_decode_size == 0 { break; }
        let empty_prefill: Vec<Vec<SequenceSlice>> = vec![];

        // Phase 1: layers + norm
        for thread_id in 0..thread_num {
            for (_index, operator) in operator_queue.iter().enumerate().take(phase2_start) {
                operator.run(0, step_decode_size, thread_num, thread_id, &empty_prefill, &step_decode_slices, &mut batch_list);
            }
        }

        // Phase 2: MatMulTopK + TopKSoftmax
        for thread_id in 0..thread_num {
            for (_index, operator) in operator_queue.iter().enumerate().skip(phase2_start) {
                operator.run(0, step_decode_size, thread_num, thread_id, &empty_prefill, &step_decode_slices, &mut batch_list);
            }
        }

        for batch_idx in 0..batch_size {
            if matches!(batch_list[batch_idx].phase, Phase::Eos) { continue; }
            let tok = unsafe { *sequences.as_mut_ptr().add(batch_idx * sequence_length + batch_list[batch_idx].sequence_index) };
            generated_tokens[batch_idx].push(tok);
            let decoded = tokenizer.decode(vec![tok as u32]).unwrap_or_default();
            eprintln!("Batch {batch_idx} Token {}: {tok} ({decoded})", step + 1);
        }
    }

    eprintln!("\n=== Results ===");
    for (batch_idx, tokens) in generated_tokens.iter().enumerate() {
        let decoded = tokenizer.decode(tokens.iter().map(|t| *t as u32).collect::<Vec<_>>()).unwrap_or_default();
        eprintln!("Batch {batch_idx}: tokens={tokens:?}");
        eprintln!("  text: {decoded:?}");
    }

    println!("{}", serde_json::to_string_pretty(&json!({
        "generated_token_ids": generated_tokens,
    }))?);

    Ok(())
}
