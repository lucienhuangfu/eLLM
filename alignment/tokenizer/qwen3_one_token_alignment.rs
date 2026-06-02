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
use std::time::Instant;

fn operator_name(operator: &Operator<f16>) -> &'static str {
    match operator {
        Operator::AddRMSZipMap(_) => "AddRMSZipMap",
        Operator::AddZipMap(_) => "AddZipMap",
        Operator::Attention(_) => "Attention",
        Operator::ExpertsMatMulDown(_) => "ExpertsMatMulDown",
        Operator::ExpertsMatMulSilu(_) => "ExpertsMatMulSilu",
        Operator::ExpertsMergeAdd(_) => "ExpertsMergeAdd",
        Operator::MatMulSigmoid(_) => "MatMulSigmoid",
        Operator::ExpertsSoftmaxNorm(_) => "ExpertsSoftmaxNorm",
        Operator::ExpertsTopkNorm(_) => "ExpertsTopkNorm",
        Operator::LiftVector(_) => "LiftVector",
        Operator::LookupRMSMap(_) => "LookupRMSMap",
        Operator::MatMul(_) => "MatMul",
        Operator::MatMul3(_) => "MatMul3",
        Operator::MatMulAdd(_) => "MatMulAdd",
        Operator::MatMulTopK(_) => "MatMulTopK",
        Operator::RMSMap(_) => "RMSMap",
        Operator::SigmoidMap(_) => "SigmoidMap",
        Operator::SiluMulZipMap(_) => "SiluMulZipMap",
        Operator::FakeEcho(_) => "FakeEcho",
        Operator::TopKSoftmax(_) => "TopKSoftmax",
    }
}

fn write_f16_tensor(path: &std::path::Path, ptr: *mut f16, len: usize) -> std::io::Result<()> {
    let byte_size = len * std::mem::size_of::<f16>();
    unsafe {
        let bytes: &[u8] = std::slice::from_raw_parts(ptr as *const u8, byte_size);
        std::fs::write(path, bytes)
    }
}

fn dump_pool_tensor(
    dump_dir: &std::path::Path,
    name: &str,
    file_name: &str,
    len: usize,
) {
    f16::with_global(|pool| {
        let ptr = pool.get(name, &vec![len]);
        let path = dump_dir.join(file_name);
        let _ = write_f16_tensor(&path, ptr, len);
    });
}

fn main() -> anyhow::Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/Qwen3-0.6B".to_string());
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

    let sequence_length = input_ids.len() + 1;
    let batch_size = 1;
    let chunk_size = input_ids.len();
    let top_k = 1;

    let load_start = Instant::now();
    eprintln!("loading f16 weights");
    let params = SafeTensorsLoader::new(&model_dir)?.load_all_weights_f16_parallel()?;
    eprintln!("loaded {} tensors in {:.2?}", params.len(), load_start.elapsed());
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

    let build_start = Instant::now();
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

    let sequences = AlignedBox::allocate_init(sequence_length * batch_size, 0usize);
    for (index, token_id) in input_ids.iter().enumerate() {
        unsafe {
            *sequences.as_mut_ptr().add(index) = *token_id as usize;
        }
    }

    let mut batch_temperature = vec![1.0f16; batch_size];
    eprintln!("calling model.forward");
    let (_output_indices, _output_tensor) =
        model.forward(sequences.as_mut_ptr(), batch_temperature.as_mut_ptr());
    eprintln!("operator queue ready in {:.2?}", build_start.elapsed());
    let operator_queue = f16::take_operator_queue();

    let slice = SequenceSlice {
        batch_index: 0,
        sequence_index: 0,
        token_start_index: 0,
        length: input_ids.len(),
        last_token_flag: true,
    };
    let prefill_list = vec![vec![slice.clone()]];
    let decode_list = vec![slice];
    let mut batch_list = vec![SequenceState {
        sequence_index: 0,
        kv_index: 0,
        filling_length: input_ids.len(),
        phase: Phase::Prefill,
        notify: Arc::new(tokio::sync::Notify::new()),
    }];

    let token_count = input_ids.len();
    let hidden_size = config.hidden_size;

    // Dump helper is kept for debugging purposes but not currently used
    #[allow(dead_code)]
    fn dump_tensor(
        pool: &mut ellm::mem_mgr::mem_pool::MemPool<f16>,
        name: &str,
        size: usize,
        path: &std::path::Path,
    ) {
        let data_ptr = pool.get(name, &vec![size]);
        let byte_size = size * std::mem::size_of::<f16>();
        unsafe {
            let bytes: &[u8] = std::slice::from_raw_parts(data_ptr as *const u8, byte_size);
            std::fs::write(path, bytes).ok();
        }
    }

    // Split into two phases: prefill for layers+norm, decode for final projection
    let phase2_start = operator_queue
        .iter()
        .position(|op| matches!(op, Operator::MatMulTopK(_)))
        .unwrap_or(operator_queue.len());

    // Phase 1: Layer operators + final RMS norm (simulating scheduler's prefill round)
    let dump_dir = std::path::Path::new("alignment/tokenizer/dump");
    std::fs::create_dir_all(dump_dir)?;
    let mut completed_layers = 0usize;
    let mut previous_operator_name = "";
    let run_start = Instant::now();
    for (index, operator) in operator_queue.iter().enumerate().take(phase2_start) {
        let current_operator_name = operator_name(operator);
        eprintln!("running operator[{index}] {}", operator_name(operator));
        operator.run(
            input_ids.len(),
            1,
            1,
            0,
            &prefill_list,
            &decode_list,
            &mut batch_list,
        );

        let size = token_count * hidden_size;
        let num_experts = config
            .layers
            .get(completed_layers)
            .and_then(|layer| match &layer.ffn {
                ellm::transformer::config::FfnKind::SparseMoe { num_experts, .. } => {
                    Some(*num_experts)
                }
                _ => None,
            })
            .unwrap_or(128);
        let num_experts_per_tok = config
            .layers
            .get(completed_layers)
            .and_then(|layer| match &layer.ffn {
                ellm::transformer::config::FfnKind::SparseMoe {
                    num_experts_per_tok,
                    ..
                } => Some(*num_experts_per_tok),
                _ => None,
            })
            .unwrap_or(8);

        if matches!(operator, Operator::LookupRMSMap(_)) {
            dump_pool_tensor(
                dump_dir,
                "model.layers.0.output_hidden",
                "rust_layer00_input.bin",
                size,
            );
            dump_pool_tensor(
                dump_dir,
                "model.layers.0.output_normal",
                "rust_layer00_post_input_norm.bin",
                size,
            );
        } else if matches!(operator, Operator::RMSMap(_))
            && matches!(
                operator_queue.get(index + 1),
                Some(Operator::MatMul3(_))
            )
            && completed_layers < config.num_hidden_layers
        {
            let tensor_name = format!("model.layers.{completed_layers}.input_layernorm.output");
            let file_name = format!("rust_layer{completed_layers:02}_post_input_norm.bin");
            dump_pool_tensor(dump_dir, &tensor_name, &file_name, size);
        } else if matches!(operator, Operator::MatMulAdd(_))
            && completed_layers < config.num_hidden_layers
        {
            let tensor_name = format!("model.layers.{completed_layers}.self_attn");
            let file_name = format!("rust_layer{completed_layers:02}_attn_residual.bin");
            dump_pool_tensor(dump_dir, &tensor_name, &file_name, size);
        } else if matches!(operator, Operator::RMSMap(_))
            && previous_operator_name == "MatMulAdd"
            && completed_layers < config.num_hidden_layers
        {
            let tensor_name =
                format!("model.layers.{completed_layers}.post_attention_layernorm.output");
            let file_name = format!("rust_layer{completed_layers:02}_post_attn_norm.bin");
            dump_pool_tensor(dump_dir, &tensor_name, &file_name, size);
        }
        // Dump router gate logits: MatMul followed by ExpertsSoftmaxNorm
        else if matches!(operator, Operator::MatMul(_))
            && matches!(
                operator_queue.get(index + 1),
                Some(Operator::ExpertsSoftmaxNorm(_))
            )
            && completed_layers < config.num_hidden_layers
        {
            let tensor_name = format!("model.layers.{completed_layers}.mlp.gate.output");
            let file_name = format!("rust_layer{completed_layers:02}_router_logits.bin");
            let router_size = token_count * num_experts;
            dump_pool_tensor(dump_dir, &tensor_name, &file_name, router_size);
        }
        // Dump routing weights and expert indices after ExpertsSoftmaxNorm
        // The topk_values_ptr stores per-token top-k weights (token-major, sorted desc)
        // The routing.topk_indices stores per-token expert indices (token-major, sorted desc)
        else if matches!(operator, Operator::ExpertsSoftmaxNorm(_))
            && completed_layers < config.num_hidden_layers
        {
            if let Operator::ExpertsSoftmaxNorm(ref softmax_op) = operator {
                let topk_count = token_count * num_experts_per_tok;
                // Dump per-token routing weights (token-major, sorted by weight desc)
                let weights_ptr = softmax_op.topk_values_ptr.ptr;
                let weights_path = dump_dir.join(format!(
                    "rust_layer{completed_layers:02}_routing_weights.bin"
                ));
                let _ = write_f16_tensor(&weights_path, weights_ptr, topk_count);
                // Dump per-token selected expert indices (token-major, sorted by weight desc)
                let indices_ptr = softmax_op.routing.topk_indices.ptr;
                let indices_path = dump_dir.join(format!(
                    "rust_layer{completed_layers:02}_selected_experts.bin"
                ));
                unsafe {
                    let byte_size = topk_count * std::mem::size_of::<usize>();
                    let bytes: &[u8] =
                        std::slice::from_raw_parts(indices_ptr as *const u8, byte_size);
                    std::fs::write(&indices_path, bytes).ok();
                }
            }
        }
        // Dump per-expert down projection output
        else if matches!(operator, Operator::ExpertsMatMulDown(_))
            && completed_layers < config.num_hidden_layers
        {
            // down_proj output: [token_count, num_experts_per_tok, hidden_size]
            let tensor_name = format!("model.layers.{completed_layers}.mlp.down_proj.output.output");
            let file_name = format!("rust_layer{completed_layers:02}_mlp_output.bin");
            let mlp_out_size = token_count * num_experts_per_tok * hidden_size;
            dump_pool_tensor(dump_dir, &tensor_name, &file_name, mlp_out_size);
        } else if matches!(operator, Operator::RMSMap(_))
            && matches!(
                operator_queue.get(index + 1),
                Some(Operator::LiftVector(_))
            )
            && completed_layers == config.num_hidden_layers
        {
            dump_pool_tensor(
                dump_dir,
                "model.norm_hidden.output",
                "rust_final_norm.bin",
                size,
            );
        }

        if matches!(operator, Operator::ExpertsMergeAdd(_))
            && completed_layers < config.num_hidden_layers
        {
            f16::with_global(|pool| {
                let tensor_name = format!("model.layers.{completed_layers}.mlp.output.output");
                let size = token_count * hidden_size;
                let ptr = pool.get(&tensor_name, &vec![size]);
                let path = dump_dir.join(format!("rust_layer{completed_layers:02}_output.bin"));
                let _ = write_f16_tensor(&path, ptr, size);
            });
            completed_layers += 1;
        }
        previous_operator_name = current_operator_name;
    }

    // Manual LiftVector: copy last token's normed hidden state to position 0
    // This simulates what LiftVector does during real decode, so MatMulTopK
    // can process the correct token in decode mode (prefill=0, decode=1).
    f16::with_global(|pool| {
        let norm_size = token_count * hidden_size;
        let norm_ptr = pool.get("model.norm_hidden.output", &vec![norm_size]);
        let src_offset = (token_count - 1) * hidden_size;
        eprintln!(
            "Manual lift: copying norm[{src_offset}..{}] to norm[0..{hidden_size}]",
            src_offset + hidden_size
        );
        unsafe {
            std::ptr::copy(norm_ptr.add(src_offset), norm_ptr, hidden_size);
        }
    });

    // Phase 2: MatMulTopK + TopKSoftmax (decode mode)
    for (index, operator) in operator_queue.iter().enumerate().skip(phase2_start) {
        eprintln!("running operator[{index}] {}", operator_name(operator));
        operator.run(0, 1, 1, 0, &prefill_list, &decode_list, &mut batch_list);
    }
    eprintln!("ran operators in {:.2?}", run_start.elapsed());

    let next_token_id = unsafe { *sequences.as_mut_ptr().add(input_ids.len()) };
    eprintln!(">>> next_token_id = {next_token_id}");

    // Dump final norm (must happen after ALL operators, so keep it here)
    // Skip to avoid crash
    // f16::with_global(|pool| {
    //     let tensor_name = "model.norm_hidden.output";
    //     let size = token_count * 1024;
    //     dump_tensor(pool, tensor_name, size,
    //         &dump_dir.join("rust_final_norm.bin"));
    // });

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "model_dir": model_dir,
            "messages": messages
                .iter()
                .map(|(role, content)| json!({"role": role, "content": content}))
                .collect::<Vec<_>>(),
            "rendered_prompt": prompt,
            "input_ids": input_ids,
            "next_token_id": next_token_id,
            "next_token": tokenizer.decode(vec![next_token_id as u32]).unwrap_or_default(),
            "dump_dir": dump_dir,
            "dumped_layers": completed_layers,
            "hidden_size": hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "phase": format!("{:?}", batch_list[0].phase),
            "kv_index": batch_list[0].kv_index,
        }))?
    );

    Ok(())
}
