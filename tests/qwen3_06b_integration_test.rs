#![feature(f16)]

use ellm::mem_mgr::allocator::AlignedBox;
use ellm::mem_mgr::mem_pool::GlobalMemPool;
use ellm::runtime::{model_loader::SafeTensorsLoader, Config};
use ellm::transformer::model::Model;
use ellm::transformer::rope::RotaryEmbedding;

#[test]
#[ignore = "Requires models/Qwen3-0.6B to be present"]
fn test_define_and_load_qwen3_06b() {
    let model_dir = "models/Qwen3-0.6B";

    println!("Loading Qwen3-0.6B configuration...");
    let config = Config::load_from_file(format!("{}/config.json", model_dir)).unwrap();
    println!("Configuration loaded: {:?}", config.family);

    println!("Creating rotary embedding...");
    let position_vec = RotaryEmbedding::new(
        config.head_dim,
        config.rotary_dim,
        config.max_position_embeddings,
        config.rope_theta as f32,
        config.rope_scaling.clone(),
    )
    .forward::<f32>();

    let sequence_length = 128;
    let batch_size = 1;
    let topk_size = 8;
    let eos_id = config.eos_token_id;

    println!("Loading model weights from safetensors...");
    let loader = SafeTensorsLoader::new(model_dir).unwrap();
    let weights = loader.load_all_weights::<f32>().unwrap();
    let weights_count = weights.len();

    println!("Initializing global memory pool with weights...");
    GlobalMemPool::init_global(weights);

    println!("Defining model structure...");
    let model = Model::<f32>::new(
        &config,
        position_vec,
        sequence_length, // chunk_size
        sequence_length, // sequence_length
        batch_size,
        topk_size,
        vec![eos_id],
    );

    println!("Model defined successfully");
    println!("Number of layers: {}", model.layers.len());
    println!("Hidden size: {}", model.hidden_size);
    println!(
        "Global memory pool initialized with weights ({} tensors)",
        weights_count
    );

    println!("Qwen3-0.6B model defined and loaded!");
}

#[test]
#[ignore = "Requires models/Qwen3-0.6B to be present"]
fn test_define_and_load_qwen3_06b_f16() {
    if !std::arch::is_x86_feature_detected!("avx512fp16") {
        eprintln!("skip test_define_and_load_qwen3_06b_f16: avx512fp16 not detected");
        return;
    }

    let model_dir = "models/Qwen3-0.6B";

    println!("Loading Qwen3-0.6B configuration...");
    let config = Config::load_from_file(format!("{}/config.json", model_dir)).unwrap();
    println!("Configuration loaded: {:?}", config.family);

    println!("Creating rotary embedding...");
    let position_vec = RotaryEmbedding::new(
        config.head_dim,
        config.rotary_dim,
        config.max_position_embeddings,
        config.rope_theta as f32,
        config.rope_scaling.clone(),
    )
    .forward::<f16>();

    let sequence_length = 128;
    let batch_size = 1;
    let topk_size = 8;
    let eos_id = config.eos_token_id;

    println!("Loading model weights from safetensors...");
    let loader = SafeTensorsLoader::new(model_dir).unwrap();
    let weights = loader.load_all_weights::<f16>().unwrap();
    let weights_count = weights.len();

    println!("Initializing global memory pool with weights...");
    GlobalMemPool::init_global(weights);

    println!("Defining model structure...");
    let model = Model::<f16>::new(
        &config,
        position_vec,
        sequence_length, // chunk_size
        sequence_length, // sequence_length
        batch_size,
        topk_size,
        vec![eos_id],
    );

    println!("Model defined successfully");
    println!("Number of layers: {}", model.layers.len());
    println!("Hidden size: {}", model.hidden_size);
    println!(
        "Global memory pool initialized with weights ({} tensors)",
        weights_count
    );

    println!("Qwen3-0.6B model defined and loaded!");
}
