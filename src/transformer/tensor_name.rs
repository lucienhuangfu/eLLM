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
