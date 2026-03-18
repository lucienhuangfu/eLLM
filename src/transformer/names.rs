use super::config::{Config, FfnKind, ModelFamily};

#[derive(Debug, Clone)]
pub struct ModelTensorNames {
    pub scope: String,
    pub token_embedding: String,
    pub position_embedding: String,
    pub lm_head: String,
}

#[derive(Debug, Clone)]
pub struct AttentionTensorNames {
    pub scope: String,
    pub q_proj: String,
    pub k_proj: String,
    pub v_proj: String,
    pub o_proj: String,
}

#[derive(Debug, Clone)]
pub struct DenseMlpTensorNames {
    pub scope: String,
    pub gate_proj: String,
    pub up_proj: String,
    pub down_proj: String,
}

#[derive(Debug, Clone)]
pub struct SparseMoeTensorNames {
    pub scope: String,
    pub router_gate: String,
    pub experts_gate_proj: String,
    pub experts_up_proj: String,
    pub experts_down_proj: String,
}

#[derive(Debug, Clone)]
pub enum FfnTensorNames {
    Dense(DenseMlpTensorNames),
    SparseMoe(SparseMoeTensorNames),
}

#[derive(Debug, Clone)]
pub struct LayerTensorNames {
    pub scope: String,
    pub attention: AttentionTensorNames,
    pub ffn: FfnTensorNames,
}

pub fn model_tensor_names(config: &Config) -> ModelTensorNames {
    match config.family {
        ModelFamily::Qwen
        | ModelFamily::Llama
        | ModelFamily::Mixtral
        | ModelFamily::MiniMax
        | ModelFamily::MiniMaxM2
        | ModelFamily::Unknown(_) => ModelTensorNames {
            scope: String::from("model"),
            token_embedding: String::from("model.embed_tokens.weight"),
            position_embedding: String::from("model.position_embedding.weight"),
            lm_head: String::from("lm_head.weight"),
        },
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
    };

    let ffn = match &config.layers[layer_idx].ffn {
        FfnKind::Dense { .. } => {
            let ffn_scope = format!("{}.mlp", scope);
            FfnTensorNames::Dense(DenseMlpTensorNames {
                scope: ffn_scope.clone(),
                gate_proj: format!("{}.gate_proj.weight", ffn_scope),
                up_proj: format!("{}.up_proj.weight", ffn_scope),
                down_proj: format!("{}.down_proj.weight", ffn_scope),
            })
        }
        FfnKind::SparseMoe { .. } => {
            let ffn_scope = format!("{}.mlp", scope);
            FfnTensorNames::SparseMoe(SparseMoeTensorNames {
                scope: ffn_scope.clone(),
                router_gate: format!("{}.gate.weight", ffn_scope),
                experts_gate_proj: format!("{}.experts.gate_proj.weight", ffn_scope),
                experts_up_proj: format!("{}.experts.up_proj.weight", ffn_scope),
                experts_down_proj: format!("{}.experts.down_proj.weight", ffn_scope),
            })
        }
    };

    LayerTensorNames {
        scope,
        attention,
        ffn,
    }
}