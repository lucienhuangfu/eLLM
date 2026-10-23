pub mod full_attention;
pub mod linear_attention {
    pub mod fused_gated_delta_rule;
    pub mod recurrent_gated_delta_rule;

    pub use fused_gated_delta_rule::FusedGatedDeltaRule;
    pub use recurrent_gated_delta_rule::RecurrentGatedDeltaRule;
}

pub use full_attention::Attention;
pub use linear_attention::FusedGatedDeltaRule;
pub use linear_attention::RecurrentGatedDeltaRule;
