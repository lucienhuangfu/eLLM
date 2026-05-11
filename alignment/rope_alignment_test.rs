
// === alignment/rope_alignment_test.rs ===
use std::fs;

// Import our RoPE implementation
#[path = "../src/transformer/rope.rs"]
mod rope;
use rope::RotaryEmbedding;

fn main() {
    println!("===== RoPE Alignment Test =====");

    // Test 1: Basic RoPE
    println!("\n--- Test 1: Basic RoPE ---");
    let rope = RotaryEmbedding::new(64, 64, 16, 10000.0, None);
    let output = rope.forward::<f32>();
    println!("Output length: {}", output.len());

    // Save as raw binary for easy comparison
    let bytes: Vec<u8> = output
        .iter()
        .flat_map(|&x| x.to_le_bytes())
        .collect();
    fs::write("alignment/dump/rust_rope_basic.bin", bytes).unwrap();

    // Test 2: Partial Rotary
    println!("\n--- Test 2: Partial Rotary ---");
    let rope_partial = RotaryEmbedding::new(8, 4, 2, 10000.0, None);
    let output_partial = rope_partial.forward::<f32>();
    println!("Output length: {}", output_partial.len());

    let bytes_partial: Vec<u8> = output_partial
        .iter()
        .flat_map(|&x| x.to_le_bytes())
        .collect();
    fs::write("alignment/dump/rust_rope_partial.bin", bytes_partial).unwrap();

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

    let bytes_yarn: Vec<u8> = output_yarn
        .iter()
        .flat_map(|&x| x.to_le_bytes())
        .collect();
    fs::write("alignment/dump/rust_rope_yarn.bin", bytes_yarn).unwrap();

    println!("\n--- Done ---");
}
