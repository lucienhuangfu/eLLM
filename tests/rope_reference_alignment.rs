use std::fs;
use std::path::PathBuf;

use ellm::transformer::rope::RotaryEmbedding;
use serde_json::Value;

fn case_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("reference")
        .join("hf")
        .join("cases")
        .join("rope_case_min.json")
}

#[test]
fn rope_reference_case_min_matches_expected_layout() {
    let case_text = fs::read_to_string(case_path()).expect("failed to read rope case");
    let case: Value = serde_json::from_str(&case_text).expect("failed to parse rope case");

    let head_dim = case["head_dim"].as_u64().expect("missing head_dim") as usize;
    let rotary_dim = case["rotary_dim"].as_u64().expect("missing rotary_dim") as usize;
    let max_sequence_length = case["max_sequence_length"]
        .as_u64()
        .expect("missing max_sequence_length") as usize;
    let theta = case["theta"].as_f64().expect("missing theta") as f32;

    let rope = RotaryEmbedding::new(head_dim, rotary_dim, max_sequence_length, theta, None);
    let values = rope.forward::<f32>();

    let expected: [f32; 16] = [
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.540_302_3,
        0.841_470_96,
        0.999_95,
        0.009_999_833,
        1.0,
        0.0,
        1.0,
        0.0,
    ];

    assert_eq!(values.len(), expected.len());
    for (idx, (lhs, rhs)) in values.iter().zip(expected.iter()).enumerate() {
        assert!(
            (lhs - rhs).abs() < 1e-5,
            "mismatch at index {idx}: lhs={lhs} rhs={rhs}"
        );
    }
}
