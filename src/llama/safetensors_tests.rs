#[cfg(test)]
mod safetensors_tests {
    use super::*;
    use std::collections::HashMap;
    use std::f16;

    #[test]
    fn test_config_deserialization() {
        let config_json = r#"
        {
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "max_position_embeddings": 8192,
            "rms_norm_eps": 1e-5
        }
        "#;

        let config: SimpleConfig = serde_json::from_str(config_json).unwrap();
        assert_eq!(config.model_type, "llama");
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, Some(8));
        assert_eq!(config.vocab_size, 128256);
        assert_eq!(config.max_position_embeddings, 8192);
        assert_eq!(config.rms_norm_eps, 1e-5);
    }

    #[test]
    fn test_f16_conversion() {
        let f32_val = 3.14159f32;
        let f16_val = f32_val as f16;

        // 验证转换不会失败
        assert!(f16_val.is_finite());

        // 测试数组转换
        let f32_array = vec![1.0f32, 2.0f32, 3.0f32];
        let f16_array: Vec<f16> = f32_array.iter().map(|&x| x as f16).collect();

        assert_eq!(f16_array.len(), 3);
        assert_eq!(f16_array[0] as f32, 1.0f32);
        assert_eq!(f16_array[1] as f32, 2.0f32);
        assert_eq!(f16_array[2] as f32, 3.0f32);
    }

    #[test]
    fn test_layer_name_generation() {
        let layer_id = 5;
        let expected_names = [
            format!("model.layers.{}.self_attn.q_proj.weight", layer_id),
            format!("model.layers.{}.self_attn.k_proj.weight", layer_id),
            format!("model.layers.{}.self_attn.v_proj.weight", layer_id),
            format!("model.layers.{}.self_attn.o_proj.weight", layer_id),
            format!("model.layers.{}.mlp.gate_proj.weight", layer_id),
            format!("model.layers.{}.mlp.up_proj.weight", layer_id),
            format!("model.layers.{}.mlp.down_proj.weight", layer_id),
            format!("model.layers.{}.input_layernorm.weight", layer_id),
            format!("model.layers.{}.post_attention_layernorm.weight", layer_id),
        ];

        assert_eq!(expected_names[0], "model.layers.5.self_attn.q_proj.weight");
        assert_eq!(expected_names[4], "model.layers.5.mlp.gate_proj.weight");
        assert_eq!(expected_names[7], "model.layers.5.input_layernorm.weight");
    }

    #[test]
    fn test_memory_calculation() {
        // 模拟权重数据
        let mut weights = HashMap::new();
        weights.insert("layer1".to_string(), vec![0.0f16; 1000]);
        weights.insert("layer2".to_string(), vec![0.0f16; 2000]);
        weights.insert("layer3".to_string(), vec![0.0f16; 3000]);

        let total_params: usize = weights.values().map(|v| v.len()).sum();
        let total_memory_bytes = total_params * 2; // f16 = 2 bytes
        let total_memory_mb = total_memory_bytes as f64 / (1024.0 * 1024.0);

        assert_eq!(total_params, 6000);
        assert_eq!(total_memory_bytes, 12000);
        assert!((total_memory_mb - 0.011444).abs() < 0.001); // ~0.011 MB
    }

    #[test]
    fn test_bf16_conversion_logic() {
        // 测试BF16转换逻辑（模拟）
        use half::bf16;

        let bf16_val = bf16::from_f32(3.14159f32);
        let f32_intermediate = bf16_val.to_f32();
        let final_f16 = f32_intermediate as f16;

        assert!(final_f16.is_finite());
        assert!((final_f16 as f32 - 3.14159f32).abs() < 0.1); // 允许精度损失
    }
}
