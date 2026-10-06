//! Tensor-name composition for the transformer stack.
//!
//! Only the *scope hierarchy* is centralized here; leaf weight names
//! (`.q_proj.weight`, `.gate_proj.weight`, `.gate.weight`, `.experts.*`, ...) are
//! formatted inside each operator from the scope it is handed, co-located with
//! their sole consumer. This keeps `model_spec` free of any reverse dependency
//! on a concrete model family `Config`.

/// Root scope for model-level tensors (`model.embed_tokens.weight`, ...).
pub const MODEL_SCOPE: &str = "model";

/// Scope of a single decoder layer, e.g. `model.layers.3`.
pub fn layer_scope(layer_idx: usize) -> String {
    format!("{MODEL_SCOPE}.layers.{layer_idx}")
}

pub fn token_embedding_name() -> &'static str {
    "model.embed_tokens.weight"
}

pub fn position_embedding_name() -> &'static str {
    "model.position_embedding.weight"
}

pub fn norm_weight_name() -> &'static str {
    "model.norm.weight"
}

/// `lm_head` is tied to the token embedding when `tie_word_embeddings` is set.
pub fn lm_head_name(tie_word_embeddings: bool) -> String {
    if tie_word_embeddings {
        token_embedding_name().to_string()
    } else {
        "lm_head.weight".to_string()
    }
}

/// HF safetensors keys of one `GatedDeltaAttention` (linear attention) block,
/// scoped at `model.layers.{i}.linear_attn`. Leaf names must equal the HF keys
/// exactly so weight loading fills them; `A_log` keeps its capital `A`.
#[derive(Debug, Clone)]
pub struct GatedDeltaAttentionTensorNames {
    pub scope: String,
    pub in_proj_qkv: String,
    pub in_proj_z: String,
    pub in_proj_b: String,
    pub in_proj_a: String,
    pub dt_bias: String,
    pub a_log: String,
    pub conv1d: String,
    pub norm: String,
    pub out_proj: String,
}

/// Build the linear-attention tensor names for `layer_idx`, mirroring the HF
/// `Qwen3_5DecoderLayer.linear_attn` submodule keys.
pub fn gated_delta_attention_tensor_names(layer_idx: usize) -> GatedDeltaAttentionTensorNames {
    let scope = format!("{}.linear_attn", layer_scope(layer_idx));
    GatedDeltaAttentionTensorNames {
        in_proj_qkv: format!("{scope}.in_proj_qkv.weight"),
        in_proj_z: format!("{scope}.in_proj_z.weight"),
        in_proj_b: format!("{scope}.in_proj_b.weight"),
        in_proj_a: format!("{scope}.in_proj_a.weight"),
        dt_bias: format!("{scope}.dt_bias"),
        a_log: format!("{scope}.A_log"),
        conv1d: format!("{scope}.conv1d.weight"),
        norm: format!("{scope}.norm.weight"),
        out_proj: format!("{scope}.out_proj.weight"),
        scope,
    }
}
