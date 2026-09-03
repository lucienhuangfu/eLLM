pub mod assign;
pub mod fake_echo;
pub mod operator;
pub mod send_sync_ptr;
pub use operator::Operator;

pub mod conv {
    pub mod causal_conv1d_silu;

    pub use causal_conv1d_silu::CausalConv1dSilu;
}

pub mod elementwise {
    pub mod add_zip;
    pub mod complex_zip;
    pub mod sigmoid_map;
    pub mod silu_mul_zip;
}

pub mod expert {
    pub mod sparse_expert {
        pub mod expert_matmul_mul;
        pub mod expert_matmul_silu_mul_matmul;
        pub mod expert_merge_add;

        pub use expert_matmul_mul::ExpertMatMulDown;
        pub use expert_matmul_silu_mul_matmul::ExpertMatMulSilu;
        pub use expert_merge_add::ExpertMergeAdd;
    }

    pub mod shared_expert {
        pub mod shared_expert_matmul_mul;
        pub mod shared_expert_matmul_silu_mul_matmul;
        pub mod shared_expert_merge_add;

        pub use shared_expert_matmul_mul::SharedExpertMatMulDown;
        pub use shared_expert_matmul_silu_mul_matmul::SharedExpertMatMulSilu;
        pub use shared_expert_merge_add::SharedExpertMergeAdd;
    }

    pub mod expert_routing;
    pub mod expert_topk_norm;

    #[allow(non_snake_case)]
    pub use sparse_expert::ExpertMatMulDown as ExpertsMatMulDown;
    #[allow(non_snake_case)]
    pub use sparse_expert::ExpertMatMulSilu as ExpertsMatMulSilu;
    #[allow(non_snake_case)]
    pub use sparse_expert::ExpertMergeAdd as ExpertsMergeAdd;

    #[allow(non_snake_case)]
    pub use shared_expert::SharedExpertMatMulDown as SharedExpertsMatMulDown;
    #[allow(non_snake_case)]
    pub use shared_expert::SharedExpertMatMulSilu as SharedExpertsMatMulSilu;
    #[allow(non_snake_case)]
    pub use shared_expert::SharedExpertMergeAdd as SharedExpertsMergeAdd;
}

pub mod expert_imports {
    #[allow(non_snake_case)]
    pub use super::expert::sparse_expert::ExpertMatMulDown as ExpertsMatMulDown;
    #[allow(non_snake_case)]
    pub use super::expert::sparse_expert::ExpertMatMulSilu as ExpertsMatMulSilu;
    #[allow(non_snake_case)]
    pub use super::expert::sparse_expert::ExpertMergeAdd as ExpertsMergeAdd;
}

pub mod full_attention {
    pub mod attention;
}

pub mod lift_vector;

pub mod linear {
    pub use super::full_attention::attention::Attention;
    pub use super::matmul::matmul::MatMul;
    pub use super::matmul::matmul3::MatMul3;
    pub use super::matmul::matmul_add::MatMulAdd;
    pub use super::matmul::matmul_proj::MatMulProj;
    pub use super::matmul::matmul_sigmoid::MatMulSigmoid;
}

pub mod matmul {
    pub mod matmul;
    pub mod matmul3;
    pub mod matmul_add;
    pub mod matmul_proj;
    pub mod matmul_sigmoid;
    pub mod matmul_topk;
}

pub mod movement {
    pub use super::lift_vector::LiftVector;
}

pub mod linear_attention {
    pub mod recurrent_gated_delta_rule;

    pub use recurrent_gated_delta_rule::RecurrentGatedDeltaRule;
}

pub mod testing {
    pub use super::fake_echo::FakeEcho;
}

pub mod normalization {
    pub mod add_rms_zip;
    pub mod lookup_rms_map;
    pub mod rms_map;
}

pub mod routing {
    pub use super::expert::expert_topk_norm::ExpertTopkNorm;
    pub use super::matmul::matmul_sigmoid::MatMulSigmoid;
    pub use super::matmul::matmul_topk::MatMulTopK;
    pub use super::softmax::softmax_norm::ExpertsSoftmaxNorm;
    pub use super::softmax::topk_softmax::TopKSoftmax;

    #[allow(non_snake_case)]
    pub use super::expert::expert_topk_norm::ExpertTopkNorm as ExpertsTopkNorm;
}

pub mod softmax {
    pub mod softmax_norm;
    pub mod topk_softmax;
}

pub mod transform {
    pub use super::elementwise::add_zip::AddZipMap;
    pub use super::elementwise::complex_zip::ComplexZipMap;
    pub use super::elementwise::sigmoid_map::SigmoidMap;
    pub use super::elementwise::silu_mul_zip::SiluMulZipMap;
    pub use super::normalization::add_rms_zip::AddRMSZipMap;
    pub use super::normalization::lookup_rms_map::LookupRMSMap;
    pub use super::normalization::rms_map::RMSMap;
}
pub mod traits {
    pub mod conv;
    pub mod expert;
    pub mod linear;
    pub mod linear_attention;
    pub mod map;
    pub mod softmax;

    pub use conv::CausalConvTrait;
    pub use expert::{
        ExpertsDownTrait, ExpertsSiluTrait, MoeMergeTrait, SharedExpertsDownTrait,
        SharedExpertsSiluTrait, SharedMergeAddTrait,
    };
    pub use linear::{
        AttentionTrait, MatMulAddTrait, MatMulProjTrait, MatMulSigmoidTrait, MatMulTrait,
        MatMulkqvTrait,
    };
    pub use linear_attention::RecurrentGatedDeltaRuleTrait;
    pub use map::{MapTrait, ZipMapTrait};
    pub use softmax::{ExpertsTopkNormTrait, MatMulTopKTrait, SoftmaxTrait, TopKSoftmaxTrait};
}
