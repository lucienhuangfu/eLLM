// === alignment/rope_alignment_test.rs ===
use std::fs;

#[path = "../src/transformer/rope.rs"]
mod rope;
use rope::RotaryEmbedding;

fn write_npy(path: &str, data: &[f32], shape: &[usize]) {
    let descr = npy::to_numpy_file(path, data, shape).unwrap();
}

fn main() {
    println!("===== RoPE Alignment Test =====");

    // Test 1: Basic RoPE
    println!("\n--- Test 1: Basic RoPE ---");
    let rope = RotaryEmbedding::new(64, 64, 16, 10000.0, None);
    let output = rope.forward::<f32>();
    println!("Output length: {}", output.len());
    write_npy("alignment/dump/rust_rope_basic.npy", &output, &[16, 64]);

    // Test 2: Partial Rotary
    println!("\n--- Test 2: Partial Rotary ---");
    let rope_partial = RotaryEmbedding::new(8, 4, 2, 10000.0, None);
    let output_partial = rope_partial.forward::<f32>();
    println!("Output length: {}", output_partial.len());
    write_npy(
        "alignment/dump/rust_rope_partial.npy",
        &output_partial,
        &[2, 8],
    );

    // Test 3: Yarn Scaling
    println!("\n--- Test 3: Yarn Scaling ---");
    let mut rope_scaling = std::collections::HashMap::new();
    rope_scaling.insert(
        "rope_type".to_string(),
        serde_json::Value::String("yarn".to_string()),
    );
    rope_scaling.insert("factor".to_string(), serde_json::Value::Number(4.into()));
    rope_scaling.insert(
        "original_max_position_embeddings".to_string(),
        serde_json::Value::Number(2.into()),
    );
    rope_scaling.insert(
        "attention_factor".to_string(),
        serde_json::Value::Number(1.25.into()),
    );

    let rope_yarn = RotaryEmbedding::new(8, 8, 16, 10000.0, Some(rope_scaling));
    let output_yarn = rope_yarn.forward::<f32>();
    println!("Output length: {}", output_yarn.len());
    println!("Attention scaling: {}", rope_yarn.attention_scaling);
    write_npy("alignment/dump/rust_rope_yarn.npy", &output_yarn, &[16, 8]);

    println!("\n--- Done ---");
}
