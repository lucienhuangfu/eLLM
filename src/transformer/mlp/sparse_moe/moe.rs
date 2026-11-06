use std::ops::{AddAssign, Neg, Sub};

use crate::config::TextConfig;
use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::num_traits::{Exp, NegInfinity, Sigmoid, Sqrt};
use crate::tensor::{GlobalOperatorQueue, Tensor};

use super::router::SparseMoeRouter;

#[derive(Clone)]
pub struct SparseMoe<T>
where
    T: Copy + PartialOrd,
{
    num_topk: usize,
    shared_intermediate_size: usize,
    router: SparseMoeRouter<T>,
    pub(crate) experts_gate_weight: Tensor<T>,
    pub(crate) experts_up_weight: Tensor<T>,
    pub(crate) experts_down_weight: Tensor<T>,
    // Shared-expert weights are allocated only when `shared_intermediate_size > 0`, so a
    // routed-only model never registers a `.weight` name that is absent from the checkpoint
    // (strict loading would otherwise panic). 共享专家权重仅在 `shared_intermediate_size > 0`
    // 时分配，routed-only 模型不会登记 ckpt 中缺失的 `.weight` 名（严格加载否则会 panic）。
    pub(crate) shared_gate_weight: Option<Tensor<T>>,
    pub(crate) shared_up_weight: Option<Tensor<T>>,
    pub(crate) shared_down_weight: Option<Tensor<T>>,
    pub(crate) shared_expert_gate_weight: Option<Tensor<T>>,
    scope_name: String,
}

impl<T> SparseMoe<T>
where
    T: Copy
        + PartialOrd
        + Default
        + Sub<Output = T>
        + Neg<Output = T>
        + Exp
        + NegInfinity
        + Sigmoid
        + Sqrt
        + AddAssign
        + GlobalMemPool
        + GlobalOperatorQueue,
{
    pub fn new(config: &TextConfig, scope: &str) -> Self {
        let hidden_size = config.hidden_size;
        let num_experts = config.num_experts;
        let num_topk = config.num_experts_per_tok;
        let shared_intermediate_size = config.shared_intermediate_size;
        let scope_name = scope.to_string();

        let gate_weight = Tensor::zeros(
            vec![num_experts, hidden_size],
            format!("{scope}.gate.weight"),
        );
        let router_bias = if config.use_routing_bias {
            Some(Tensor::zeros(
                vec![num_experts],
                format!("{scope}.e_score_correction_bias"),
            ))
        } else {
            None
        };

        let router = SparseMoeRouter::new(
            hidden_size,
            num_experts,
            num_topk,
            gate_weight,
            router_bias.clone(),
            &config.scoring_func,
            scope_name.clone(),
        );

        let experts_gate_weight = Tensor::zeros(
            vec![num_experts, config.moe_intermediate_size, hidden_size],
            format!("{scope}.experts.gate_proj.weight"),
        );
        let experts_up_weight = Tensor::zeros(
            vec![num_experts, config.moe_intermediate_size, hidden_size],
            format!("{scope}.experts.up_proj.weight"),
        );
        let experts_down_weight = Tensor::zeros(
            vec![num_experts, hidden_size, config.moe_intermediate_size],
            format!("{scope}.experts.down_proj.weight"),
        );

        // Shared-expert weights exist only for shared-expert models. NT layouts line up with
        // the fused MLP operator without transposition: gate/up [Is, H], down [H, Is],
        // shared_expert_gate [H].
        // 共享专家权重仅存在于带共享专家的模型。NT 布局与融合 MLP 算子天然吻合，无需转置。
        let (shared_gate_weight, shared_up_weight, shared_down_weight, shared_expert_gate_weight) =
            if shared_intermediate_size > 0 {
                (
                    Some(Tensor::zeros(
                        vec![shared_intermediate_size, hidden_size],
                        format!("{scope}.shared_expert.gate_proj.weight"),
                    )),
                    Some(Tensor::zeros(
                        vec![shared_intermediate_size, hidden_size],
                        format!("{scope}.shared_expert.up_proj.weight"),
                    )),
                    Some(Tensor::zeros(
                        vec![hidden_size, shared_intermediate_size],
                        format!("{scope}.shared_expert.down_proj.weight"),
                    )),
                    Some(Tensor::zeros(
                        vec![hidden_size],
                        format!("{scope}.shared_expert_gate.weight"),
                    )),
                )
            } else {
                (None, None, None, None)
            };

        Self {
            num_topk,
            shared_intermediate_size,
            router,
            experts_gate_weight,
            experts_up_weight,
            experts_down_weight,
            shared_gate_weight,
            shared_up_weight,
            shared_down_weight,
            shared_expert_gate_weight,
            scope_name,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<T>,
        residual: &Tensor<T>,
        decode_only_flag: bool,
        tensor_name: String,
    ) -> Tensor<T>
    where
        T: Send + 'static,
    {
        let _ = tensor_name;
        let routing = self.router.forward(hidden_states, decode_only_flag);

        // Single fused MLP operator: gate/up + SiLU + down for routed (Phase B) and, when
        // present, the dense shared branch (Phase A). The intermediate never leaves scratch.
        // 单个融合 MLP 算子：routed（Phase B）+ 稠密 shared（Phase A，若有）的 gate/up + SiLU + down，
        // 中间激活不落 scratch 之外。
        let (routed, shared) = hidden_states.experts_mlp_fused(
            &self.experts_gate_weight,
            &self.experts_up_weight,
            &self.experts_down_weight,
            self.shared_gate_weight.as_ref(),
            self.shared_up_weight.as_ref(),
            self.shared_down_weight.as_ref(),
            routing,
            self.num_topk,
            self.shared_intermediate_size,
            decode_only_flag,
            format!("{}.experts_mlp_routed.output", self.scope_name),
            format!("{}.shared_mlp.output", self.scope_name),
        );

        // Single fused merge node for both variants: `shared`/`shared_expert_gate_weight` are
        // `None` when `shared_intermediate_size == 0`, and the operator skips the gated shared
        // term. No routed/shared branch (核心思路 #1: 单一融合路径).
        // 单一融合 merge 节点，两种形态共用：`shared_intermediate_size == 0` 时 shared 权重为
        // `None`，算子跳过带门控的 shared 项。不分 routed/shared 双路径。
        routed.experts_merge_add_fused(
            residual,
            if self.shared_down_weight.is_some() {
                Some(&shared)
            } else {
                None
            },
            self.shared_expert_gate_weight.as_ref(),
            hidden_states,
            routing,
            decode_only_flag,
            format!("{}.output", self.scope_name),
        )
    }
}
