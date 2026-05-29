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
    let chat_template =
        ChatTemplate::from_model_files(&chat_template_path, &tokenizer_config_path)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let prompt = chat_template
        .apply_chat_template(&messages, true)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let tokenizer = load_tiktoken(&tokenizer_path, &tokenizer_config_path)
        .map_err(|e| anyhow::anyhow!(e))?;
    let input_ids = tokenizer.encode_with_special_tokens(&prompt);

    let sequence_length = input_ids.len() + 1;
    let batch_size = 1;
    let chunk_size = input_ids.len();
    let top_k = 1;

    eprintln!("loading f16 weights");
    let params = SafeTensorsLoader::new(&model_dir)?.load_all_weights_f16()?;
    eprintln!("loaded {} tensors", params.len());
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
        .and_then(GenerationConfig::eos_token_ids)
        .filter(|ids| !ids.is_empty())
        .unwrap_or(config.eos_token_ids);

    eprintln!("building model graph");
    let mut model = Model::<f16>::new(
        &config,
        position_vec,
        chunk_size,
        sequence_length,
        batch_size,
        top_k, 1.0f16, 0.0f16, false,
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
    eprintln!("operator queue ready");
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

    // Dump helper is kept for debugging purposes but not currently used
    #[allow(dead_code)]
    fn dump_tensor(pool: &mut ellm::mem_mgr::mem_pool::MemPool<f16>, name: &str, size: usize, path: &std::path::Path) {
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
    for (index, operator) in operator_queue.iter().enumerate().take(phase2_start) {
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
    }

    // Dump final norm BEFORE manual lift (all 15 tokens intact)
    f16::with_global(|pool| {
        let norm_size = token_count * 1024;
        let norm_ptr = pool.get("model.norm_hidden.output", &vec![norm_size]);
        let byte_size = norm_size * std::mem::size_of::<f16>();
        let first_val = unsafe { *norm_ptr };
        let last_val = unsafe { *norm_ptr.add(norm_size - 1) };
        eprintln!(
            "dumping final norm: {norm_size} elements, first={first_val:.4?}, last={last_val:.4?}"
        );
        unsafe {
            let bytes: &[u8] = std::slice::from_raw_parts(norm_ptr as *const u8, byte_size);
            std::fs::write(dump_dir.join("rust_final_norm.bin"), bytes).ok();
        }
    });

    // Manual LiftVector: copy last token's normed hidden state to position 0
    // This simulates what LiftVector does during real decode, so MatMulTopK
    // can process the correct token in decode mode (prefill=0, decode=1).
    f16::with_global(|pool| {
        let norm_size = token_count * 1024;
        let norm_ptr = pool.get("model.norm_hidden.output", &vec![norm_size]);
        let src_offset = (token_count - 1) * 1024;
        eprintln!(
            "Manual lift: copying norm[{src_offset}..{}] to norm[0..1024]",
            src_offset + 1024
        );
        unsafe {
            std::ptr::copy(norm_ptr.add(src_offset), norm_ptr, 1024);
        }
    });

    // Phase 2: MatMulTopK + TopKSoftmax (decode mode)
    for (index, operator) in operator_queue.iter().enumerate().skip(phase2_start) {
        eprintln!("running operator[{index}] {}", operator_name(operator));
        operator.run(
            0,
            1,
            1,
            0,
            &prefill_list,
            &decode_list,
            &mut batch_list,
        );
    }

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
            "phase": format!("{:?}", batch_list[0].phase),
            "kv_index": batch_list[0].kv_index,
        }))?
    );

    Ok(())
}
