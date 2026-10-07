//! Family-agnostic, dtype-agnostic model *blueprint primitives*: plain
//! serializable per-layer data (`AttentionKind`, `FfnKind`, `RouterScoringKind`,
//! `LayerSpec`) plus the whole-model `TextConfig` shared by every family.
//! Per-family defaults are injected via `FamilyProfile` (values supplied by each
//! `model_family` submodule); the runtime `<T>` modules live in `transformer`
//! and are built from these via a single exhaustive match.

mod attention_kind;
mod ffn_kind;
mod layer_spec;
mod model_name;
mod router_scoring;
mod text_config;

pub use crate::config::HfConfig;
pub use attention_kind::AttentionKind;
pub use ffn_kind::FfnKind;
pub use layer_spec::LayerSpec;
pub use model_name::ModelName;
pub use router_scoring::RouterScoringKind;
pub use text_config::{FamilyProfile, LinearDefaults, TextConfig};
