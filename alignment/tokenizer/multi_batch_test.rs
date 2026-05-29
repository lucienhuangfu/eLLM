#![feature(f16)]
#![feature(sync_unsafe_cell)]

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
use std::cell::SyncUnsafeCell;
use std::f16;
use std::sync::{Arc, Barrier};

fn main() {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/Qwen3-0.6B".to_string());
    let max_tokens: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "6".to_string())
        .parse()
        .unwrap_or(6);

    let config = Config::load_from_file(format!("{model_dir}/config.json")).unwrap();
    let gen_cfg =
        GenerationConfig::load_from_file(format!("{model_dir}/generation_config.json")).ok();

    let tokenizer_path = format!("{model_dir}/tokenizer.json");
    let tokenizer_config_path = format!("{model_dir}/tokenizer_config.json");
    let chat_template_path = format!("{model_dir}/chat_template.jinja");

    let chat_template = ChatTemplate::from_model_files(&chat_template_path, &tokenizer_config_path)
        .ok()
        .unwrap();
    let tokenizer = load_tiktoken(&tokenizer_path, &tokenizer_config_path).unwrap();

    let messages_batch = [
        [("user", "Tell me a short joke about programming.")],
        [(
            "user",
            "What is the difference between stack and heap memory?",
        )],
    ];

    let mut all_input_ids: Vec<Vec<u32>> = vec![];
    for messages in &messages_batch {
        let prompt = chat_template.apply_chat_template(messages, true).unwrap();
        let ids = tokenizer.encode_with_special_tokens(&prompt);
        println!("Prompt '{}': {} tokens", messages[0].1, ids.len());
        all_input_ids.push(ids);
    }

    let batch_size = messages_batch.len();
    let total_input: usize = all_input_ids.iter().map(|ids| ids.len()).sum();
    let max_input: usize = all_input_ids.iter().map(|ids| ids.len()).max().unwrap();
    let sequence_length = max_input + max_tokens + 8;
    let chunk_size = total_input + batch_size * max_tokens;
    let top_k = 1;

    println!("Batches={batch_size} total_tokens={total_input} seq_len={sequence_length} chunk={chunk_size}");

    let params = SafeTensorsLoader::new(&model_dir)
        .unwrap()
        .load_all_weights_f16()
        .unwrap();
    f16::init_global_strict(params);

    let position_vec = RotaryEmbedding::new(
        config.head_dim,
        config.rotary_dim,
        config.max_position_embeddings,
        config.rope_theta as f32,
        config.rope_scaling.clone(),
    )
    .forward::<f16>();

    let eos_ids = gen_cfg
        .as_ref()
        .and_then(|g| g.eos_token_id_list.clone())
        .filter(|ids| !ids.is_empty())
        .unwrap_or(config.eos_token_ids.clone());

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

    let mut sequences = AlignedBox::allocate_init(sequence_length * batch_size, 0usize);
    for (batch_idx, ids) in all_input_ids.iter().enumerate() {
        for (pos, token_id) in ids.iter().enumerate() {
            unsafe {
                *sequences
                    .as_mut_ptr()
                    .add(batch_idx * sequence_length + pos) = *token_id as usize;
            }
        }
    }

    let mut temperatures = vec![1.0f16; batch_size];
    let (_indices, _values) = model.forward(sequences.as_mut_ptr(), temperatures.as_mut_ptr());
    let operator_queue: Vec<Operator<f16>> = f16::take_operator_queue();

    let phase2_start = operator_queue
        .iter()
        .position(|op| matches!(op, Operator::MatMulTopK(_)))
        .unwrap_or(operator_queue.len());

    // Build lists
    let mut token_offset = 0usize;
    let mut prefill_group = Vec::new();
    let mut decode_slices = Vec::new();
    let mut batch_list_vec = Vec::new();

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
        decode_slices.push(slice);
        batch_list_vec.push(SequenceState {
            sequence_index: 0,
            kv_index: 0,
            filling_length: input_len,
            phase: Phase::Prefill,
            notify: Arc::new(tokio::sync::Notify::new()),
        });
        token_offset += input_len;
    }

    let prefill_list = vec![prefill_group];
    let prefill_size = total_input;
    let decode_size = batch_size;

    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    let thread_num = core_ids.len().max(1);
    println!("Threads: {thread_num}");

    // Shared state (same pattern as ServingRunner)
    let queue: Arc<[Operator<f16>]> = operator_queue.into();
    let batch_list = Arc::new(SyncUnsafeCell::new(batch_list_vec));
    let pf_list = Arc::new(prefill_list);
    let dc_list = Arc::new(decode_slices);
    let seq_ptr = sequences.as_mut_ptr();

    // ===== PREFILL PHASE 1 =====
    println!("Prefill Phase 1...");
    {
        let barrier = Arc::new(Barrier::new(thread_num));
        let mut handles = Vec::new();
        for tid in 0..thread_num {
            let q = Arc::clone(&queue);
            let b = Arc::clone(&barrier);
            let bl = Arc::clone(&batch_list);
            let pf = Arc::clone(&pf_list);
            let dc = Arc::clone(&dc_list);
            handles.push(std::thread::spawn(move || {
                for (_idx, op) in q.iter().enumerate().take(phase2_start) {
                    b.wait();
                    let bl_ref = unsafe { &mut *bl.get() };
                    op.run(prefill_size, decode_size, thread_num, tid, &pf, &dc, bl_ref);
                    b.wait();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    // Manual lift
    f16::with_global(|pool| {
        let norm_size = chunk_size * 1024;
        let norm_ptr = pool.get("model.norm_hidden.output", &vec![norm_size]);
        let mut offset = 0usize;
        for batch_idx in 0..batch_size {
            let input_len = all_input_ids[batch_idx].len();
            unsafe {
                std::ptr::copy(
                    norm_ptr.add((offset + input_len - 1) * 1024),
                    norm_ptr.add(batch_idx * 1024),
                    1024,
                );
            }
            offset += input_len;
        }
    });

    // ===== PREFILL PHASE 2 =====
    println!("Prefill Phase 2...");
    {
        let barrier = Arc::new(Barrier::new(thread_num));
        let mut handles = Vec::new();
        for tid in 0..thread_num {
            let q = Arc::clone(&queue);
            let b = Arc::clone(&barrier);
            let bl = Arc::clone(&batch_list);
            let pf = Arc::clone(&pf_list);
            let dc = Arc::clone(&dc_list);
            handles.push(std::thread::spawn(move || {
                for (_idx, op) in q.iter().enumerate().skip(phase2_start) {
                    b.wait();
                    let bl_ref = unsafe { &mut *bl.get() };
                    op.run(0, decode_size, thread_num, tid, &pf, &dc, bl_ref);
                    b.wait();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    // Read first tokens
    let mut generated_tokens: Vec<Vec<usize>> = vec![vec![]; batch_size];
    for batch_idx in 0..batch_size {
        let input_len = all_input_ids[batch_idx].len();
        let tok = unsafe { *seq_ptr.add(batch_idx * sequence_length + input_len) };
        generated_tokens[batch_idx].push(tok);
        println!(
            "Batch {batch_idx} Token 1: {tok} ({})",
            tokenizer.decode(vec![tok as u32]).unwrap_or_default()
        );
    }

    // ===== DECODE LOOP =====
    for _step in 1..max_tokens {
        let bl_ref = unsafe { &mut *batch_list.get() };
        if bl_ref.iter().all(|s| matches!(s.phase, Phase::Eos)) {
            break;
        }

        let mut step_slices = Vec::new();
        for batch_idx in 0..batch_size {
            if matches!(bl_ref[batch_idx].phase, Phase::Eos) {
                continue;
            }
            step_slices.push(SequenceSlice {
                batch_index: batch_idx,
                sequence_index: bl_ref[batch_idx].sequence_index,
                token_start_index: batch_idx,
                length: 1,
                last_token_flag: true,
            });
        }
        let step_count = step_slices.len();
        if step_count == 0 {
            break;
        }
        let step_dc = Arc::new(step_slices);
        let empty_pf: Arc<Vec<Vec<SequenceSlice>>> = Arc::new(vec![]);

        // Phase 1
        {
            let barrier = Arc::new(Barrier::new(thread_num));
            let mut handles = Vec::new();
            for tid in 0..thread_num {
                let q = Arc::clone(&queue);
                let b = Arc::clone(&barrier);
                let bl = Arc::clone(&batch_list);
                let pf = Arc::clone(&empty_pf);
                let dc = Arc::clone(&step_dc);
                handles.push(std::thread::spawn(move || {
                    for (_idx, op) in q.iter().enumerate().take(phase2_start) {
                        b.wait();
                        let bl_ref = unsafe { &mut *bl.get() };
                        op.run(0, step_count, thread_num, tid, &pf, &dc, bl_ref);
                        b.wait();
                    }
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        }

        // Phase 2
        {
            let barrier = Arc::new(Barrier::new(thread_num));
            let mut handles = Vec::new();
            for tid in 0..thread_num {
                let q = Arc::clone(&queue);
                let b = Arc::clone(&barrier);
                let bl = Arc::clone(&batch_list);
                let pf = Arc::clone(&empty_pf);
                let dc = Arc::clone(&step_dc);
                handles.push(std::thread::spawn(move || {
                    for (_idx, op) in q.iter().enumerate().skip(phase2_start) {
                        b.wait();
                        let bl_ref = unsafe { &mut *bl.get() };
                        op.run(0, step_count, thread_num, tid, &pf, &dc, bl_ref);
                        b.wait();
                    }
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        }

        for batch_idx in 0..batch_size {
            if matches!(bl_ref[batch_idx].phase, Phase::Eos) {
                continue;
            }
            let tok = unsafe {
                *seq_ptr.add(batch_idx * sequence_length + bl_ref[batch_idx].sequence_index)
            };
            generated_tokens[batch_idx].push(tok);
        }
    }

    println!("\nResults:");
    for (batch_idx, tokens) in generated_tokens.iter().enumerate() {
        let text = tokenizer
            .decode(tokens.iter().map(|t| *t as u32).collect::<Vec<_>>())
            .unwrap_or_default();
        println!("Batch {batch_idx}: {} tokens -> {text:?}", tokens.len());
    }
}
