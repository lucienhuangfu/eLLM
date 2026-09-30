//! Family-agnostic, dtype-agnostic model *blueprint primitives*: plain
//! serializable per-layer data (`AttentionKind`, `FfnKind`, `RouterScoringKind`,
//! `LayerPlan`). The whole-model `Config` lives with each model family
//! (`model_family::qwen3_moe`); the runtime `<T>` module enums live in
//! `transformer` and are built from these via a single exhaustive match.

mod attention_kind;
mod ffn_kind;
mod layer_plan;
mod model_name;
mod router_scoring;
pub mod tensor_name;

pub use crate::config::HfConfig;
pub use attention_kind::AttentionKind;
pub use ffn_kind::FfnKind;
pub(crate) use ffn_kind::FfnResolveParams;
pub use layer_plan::LayerPlan;
pub use model_name::ModelName;
pub use router_scoring::RouterScoringKind;
