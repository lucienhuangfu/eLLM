
// === alignment/rust/rope.rs ===
use std::env;
use std::fs::File;
use std::path::Path;

use ndarray::Array2;
use ndarray_npy::WriteNpyExt;

// Import our RoPE implementation
#[path = "../../src/transformer/rope.rs"]
mod rope;
use rope::RotaryEmbedding;

fn main() {
    println!("===== RoPE =====");

    // Standard case
    let head_dim = 64;
    let rotary_dim = 64;
    let max_sequence_length = 16;
    let theta = 10000.0;

    println!("Generating Rust output with:");
    println!("  head_dim = {}", head_dim);
    println!("  rotary_dim = {}", rotary_dim);
    println!("  max_sequence_length = {}", max_sequence_length);
    println!("  theta = {}", theta);

    let rope_emb = RotaryEmbedding::new(head_dim, rotary_dim, max_sequence_length, theta, None);
    let output = rope_emb.forward::<f32>();

    // Reshape to [max_sequence_length, head_dim]
    let output_reshaped = Array2::from_shape_vec((max_sequence_length, head_dim), output).unwrap();

    // Save to file
    let path = Path::new("alignment/dump/rust_rope_output.npy");
    let mut file = File::create(path).unwrap();
    output_reshaped.write_npy(&mut file).unwrap();

    println!("\nSaved to alignment/dump/rust_rope_output.npy");
    println!("Shape: {:?}", output_reshaped.shape());

    // Partial rotary case
    let head_dim_partial = 8;
    let rotary_dim_partial = 4;
    let max_sequence_length_partial = 2;

    let rope_emb_partial = RotaryEmbedding::new(
        head_dim_partial,
        rotary_dim_partial,
        max_sequence_length_partial,
        theta,
        None,
    );
    let output_partial = rope_emb_partial.forward::<f32>();
    let output_reshaped_partial =
        Array2::from_shape_vec((max_sequence_length_partial, head_dim_partial), output_partial)
            .unwrap();

    let path_partial = Path::new("alignment/dump/rust_rope_output_partial.npy");
    let mut file_partial = File::create(path_partial).unwrap();
    output_reshaped_partial.write_npy(&mut file_partial).unwrap();

    println!("Saved partial case to alignment/dump/rust_rope_output_partial.npy");
    println!("Shape: {:?}", output_reshaped_partial.shape());

    // Yarn scaling case
    let head_dim_yarn = 8;
    let rotary_dim_yarn = 8;
    let max_sequence_length_yarn = 16;
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

    let rope_emb_yarn = RotaryEmbedding::new(
        head_dim_yarn,
        rotary_dim_yarn,
        max_sequence_length_yarn,
        theta,
        Some(rope_scaling),
    );
    let output_yarn = rope_emb_yarn.forward::<f32>();
    let output_reshaped_yarn =
        Array2::from_shape_vec((max_sequence_length_yarn, head_dim_yarn), output_yarn).unwrap();

    let path_yarn = Path::new("alignment/dump/rust_rope_output_yarn.npy");
    let mut file_yarn = File::create(path_yarn).unwrap();
    output_reshaped_yarn.write_npy(&mut file_yarn).unwrap();

    println!("Saved yarn case to alignment/dump/rust_rope_output_yarn.npy");
    println!("Shape: {:?}", output_reshaped_yarn.shape());
}
