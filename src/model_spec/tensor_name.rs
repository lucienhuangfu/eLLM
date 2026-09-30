use super::ModelName;
use crate::model_family::qwen3_moe::Config;

#[derive(Debug, Clone)]
pub struct ModelTensorNames {
    pub scope: String,
    pub token_embedding: String,
    pub position_embedding: String,
    pub lm_head: String,
    pub norm_weight: String,
}

#[derive(Debug, Clone)]
pub struct AttentionTensorNames {
    pub scope: String,
    pub q_proj: String,
    pub k_proj: String,
    pub v_proj: String,
    pub o_proj: String,
    pub q_norm: String,
    pub k_norm: String,
}

#[derive(Debug, Clone)]
pub struct DenseMlpTensorNames {
    pub scope: String,
    pub gate_proj: String,
    pub up_proj: String,
    pub down_proj: String,
}

impl DenseMlpTensorNames {
    pub fn new(scope: &str) -> Self {
        Self {
            gate_proj: format!("{}.gate_proj.weight", scope),
            up_proj: format!("{}.up_proj.weight", scope),
            down_proj: format!("{}.down_proj.weight", scope),
            scope: scope.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SparseMoeTensorNames {
    pub scope: String,
    pub router_gate: String,
    pub router_bias: Option<String>,
    pub experts_gate_proj: String,
    pub experts_up_proj: String,
    pub experts_down_proj: String,
}

impl SparseMoeTensorNames {
    pub fn new(scope: &str, use_routing_bias: bool) -> Self {
        Self {
            router_gate: format!("{}.gate.weight", scope),
            router_bias: if use_routing_bias {
                Some(format!("{}.e_score_correction_bias", scope))
            } else {
                None
            },
            experts_gate_proj: format!("{}.experts.gate_proj.weight", scope),
            experts_up_proj: format!("{}.experts.up_proj.weight", scope),
            experts_down_proj: format!("{}.experts.down_proj.weight", scope),
            scope: scope.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayerTensorNames {
    pub scope: String,
    pub attention: AttentionTensorNames,
    pub ffn_scope: String,
    pub input_layernorm: String,
    pub post_attention_layernorm: String,
}

pub fn model_tensor_names(config: &Config) -> ModelTensorNames {
    match config.family {
        ModelName::Qwen
        | ModelName::Llama
        | ModelName::Mixtral
        | ModelName::MiniMax
        | ModelName::MiniMaxM2
        | ModelName::Unknown(_) => {
            let token_embedding = String::from("model.embed_tokens.weight");
            let lm_head = if config.tie_word_embeddings {
                token_embedding.clone()
            } else {
                String::from("lm_head.weight")
            };

            ModelTensorNames {
                scope: String::from("model"),
                token_embedding,
                position_embedding: String::from("model.position_embedding.weight"),
                lm_head,
                norm_weight: String::from("model.norm.weight"),
            }
        }
    }
}

pub fn layer_tensor_names(config: &Config, layer_idx: usize) -> LayerTensorNames {
    let model_names = model_tensor_names(config);
    let scope = format!("{}.layers.{}", model_names.scope, layer_idx);
    let attention_scope = format!("{}.self_attn", scope);

    let attention = AttentionTensorNames {
        scope: attention_scope.clone(),
        q_proj: format!("{}.q_proj.weight", attention_scope),
        k_proj: format!("{}.k_proj.weight", attention_scope),
        v_proj: format!("{}.v_proj.weight", attention_scope),
        o_proj: format!("{}.o_proj.weight", attention_scope),
        q_norm: format!("{}.q_norm.weight", attention_scope),
        k_norm: format!("{}.k_norm.weight", attention_scope),
    };

    let ffn_scope = format!("{}.mlp", scope);

    LayerTensorNames {
        input_layernorm: format!("{}.input_layernorm.weight", scope),
        post_attention_layernorm: format!("{}.post_attention_layernorm.weight", scope),
        scope,
        attention,
        ffn_scope,
    }
}
