pub mod full_attention;
pub mod linear_attention {
    pub mod recurrent_gated_delta_rule;

    pub use recurrent_gated_delta_rule::RecurrentGatedDeltaRule;
}

pub use full_attention::Attention;
pub use linear_attention::RecurrentGatedDeltaRule;
