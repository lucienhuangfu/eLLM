//! Family-agnostic, dtype-agnostic model *blueprint primitives*: plain
//! serializable per-layer data (`FfnKind`, `RouterScoringKind`, `LayerSpec`)
//! plus the whole-model `TextConfig` shared by every family.
//! Per-family defaults are injected via `FamilyProfile` (values supplied by each
//! `model_family` submodule); the runtime `<T>` modules live in `transformer`
//! and are built from these.

mod layer_spec;
mod model_name;
mod text_config;

pub use crate::config::HfConfig;
pub use crate::transformer::mlp::{FfnKind, RouterScoringKind};
pub use layer_spec::LayerSpec;
pub use model_name::ModelName;
pub use text_config::{FamilyProfile, LinearDefaults, TextConfig};
