use std::ops::{AddAssign, Neg, Sub};

use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::num_traits::{Exp, FromNumber, NegInfinity, Sigmoid, Sqrt};

use crate::kernel::common::matmul_params::MatMulParams;
use crate::tensor::{GlobalOperatorQueue, Tensor};

use crate::config::TextConfig;
use crate::transformer::decoder_layer::zero_centered_norm;

/// C 类架构事实：注意力层**不带** q/k 头归一化的模型清单。
///
/// 默认视为有 qk-norm，例外的模型才登记——这是 Python 的实际形状：
/// `modeling_qwen3_5.py` 的 `Qwen3_5Attention.__init__` 无条件
/// `self.q_norm = Qwen3_5RMSNorm(head_dim)`，`configuration_qwen3_5.py` 里
/// 连 `use_qk_norm` 这个字段都不存在。实测本仓库四份 ckpt 全部需要
/// q_norm/k_norm，而 Qwen3-0.6B / Qwen3-Coder-30B / Qwen3.8 的 config.json
/// 都没写 `use_qk_norm` 键（只 MiniMax 显式写了 `true`）——即旧
/// `hf.use_qk_norm \|\| profile.qk_norm` 那条判断本来就只能靠家族默认值兼得。
///
/// 本表逐字保住旧行为：qwen / qwen3_moe 旧经 `qwen3_moe::profile()` 置 true，
/// qwen2 / qwen2_moe 旧为中性 `false`，llama / mixtral 从无 profile。新增无
/// qk-norm 的家族（LLaMA / Mistral 系）时在此登记其 `model_type`。
pub(crate) const NO_QK_NORM_MODELS: &[&str] = &["llama", "mixtral", "qwen2", "qwen2_moe"];

/// 本模型的 full attention 是否创建 `q_norm.weight` / `k_norm.weight`。
pub(crate) fn use_qk_norm(model_type: &str) -> bool {
    !NO_QK_NORM_MODELS.contains(&model_type)
}

/// 输出门的门宽倍数：`attn_output_gate` 为真时 HF 的 `q_proj.weight` 按 head
/// 交错存放 query|gate 两半，行数翻倍（modeling_qwen3_5.py L751-L753
/// `config.num_attention_heads * self.head_dim * 2`），forward 里
/// `attn_output * torch.sigmoid(gate)`（L808）。这是数据（JSON 键）而非架构
/// 事实，故读 `config.attn_output_gate`，判定与实现都呆在本文件。
const GATE_ROWS: usize = 2;

// #[derive(Clone)]
pub struct FullAttention<T>
where
    T: Copy + PartialOrd,
{
    sequence_length: usize,
    batch_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    scaling: T,
    use_qk_norm: bool,
    zero_centered: bool,
    is_causal: bool,
    /// Qwen3.5 输出门开关（`config.attn_output_gate`）。
    use_gate: bool,
    q_weight: Tensor<T>,
    k_weight: Tensor<T>,
    v_weight: Tensor<T>,
    pub(super) o_weight: Tensor<T>,
    /// 仅 `use_qk_norm` 时存在：名字为 `{scope}.q_norm.weight` 的权重在
    /// 无 qk-norm 的 ckpt（LLaMA / Qwen2 系）里根本不存在，而严格加载模式会对
    /// 未命中的权重名 panic，故此处必须按否不分配。
    q_norm_weight: Option<Tensor<T>>,
    k_norm_weight: Option<Tensor<T>>,
    pub(super) scope_name: String,
}

impl<T> FullAttention<T>
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
        + FromNumber
        + AddAssign
        + GlobalMemPool
        + GlobalOperatorQueue,
{
    pub fn new(
        config: &TextConfig,
        sequence_length: usize,
        batch_size: usize,
        is_causal: bool,
        scope: &str,
    ) -> Self {
        let head_dim: usize = config.head_dim;
        let scaling = T::from_f32(1.0 / (head_dim as f32).sqrt());
        // C 类架构判断在本文件完成：不看配置字段，只看 `model_type`。
        let use_qk_norm = use_qk_norm(&config.model_type);
        // 输出门是数据事实（config.json 写了 `attn_output_gate`），直接读字段。
        let use_gate = config.attn_output_gate;
        let query_rows = config.num_attention_heads * head_dim;

        Self {
            sequence_length,
            batch_size: batch_size,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: head_dim,
            scaling: scaling,
            use_qk_norm,
            zero_centered: zero_centered_norm(&config.model_type),
            is_causal,
            use_gate,
            // 有门时行宽翻倍：每个 head 的前 head_dim 行为 query、后 head_dim 行为 gate。
            q_weight: Tensor::zeros(
                vec![
                    query_rows * if use_gate { GATE_ROWS } else { 1 },
                    config.hidden_size,
                ],
                format!("{scope}.q_proj.weight"),
            ),
            k_weight: Tensor::zeros(
                vec![config.num_key_value_heads * head_dim, config.hidden_size],
                format!("{scope}.k_proj.weight"),
            ),
            v_weight: Tensor::zeros(
                vec![config.num_key_value_heads * head_dim, config.hidden_size],
                format!("{scope}.v_proj.weight"),
            ),

            o_weight: Tensor::zeros(
                vec![config.hidden_size, config.num_attention_heads * head_dim],
                format!("{scope}.o_proj.weight"),
            ),
            q_norm_weight: use_qk_norm
                .then(|| Tensor::zeros(vec![head_dim], format!("{scope}.q_norm.weight"))),
            k_norm_weight: use_qk_norm
                .then(|| Tensor::zeros(vec![head_dim], format!("{scope}.k_norm.weight"))),
            scope_name: scope.to_string(),
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<T>,
        position_embedding: &Tensor<T>,
        decode_only_flag: bool,
    ) -> Tensor<T>
    where
        T: Send + 'static,
    {
        {
            //println!("hidden_states shape: {:?}", hidden_states.shape);

            // mul_rms_complex 合并operators
            // q [chunk_size, hidden_size]
            // k [batch_size, head_num, sequence_length, head_dim]
            // v [batch_size, head_num, sequence_length, head_dim]
            // use_qk_norm 为 false 时 matmul3 从不解引用这两个指针（其内部两处
            // `if self.use_qk_norm` 为门），借用 q/k 投影权重占位，避免为不存在的
            // 权重名向内存池申请。
            let (q_norm_weight, k_norm_weight) = match (&self.q_norm_weight, &self.k_norm_weight) {
                (Some(q), Some(k)) => (q, k),
                _ => (&self.q_weight, &self.k_weight),
            };
            let (query_states, key_states, value_states, gate_state) = hidden_states.matmul3(
                &self.q_weight,
                &self.k_weight,
                &self.v_weight,
                q_norm_weight,
                k_norm_weight,
                position_embedding,
                self.sequence_length,
                self.batch_size,
                self.num_key_value_heads,
                self.num_attention_heads / self.num_key_value_heads,
                self.head_dim,
                self.use_qk_norm,
                MatMulParams {
                    a_row_step_macro: 3,
                    b_row_step_macro: 128,
                    column_step_macro: 64,
                    a_row_step_micro: 3,
                    b_row_step_micro: 32,
                },
                self.zero_centered,
                self.use_gate,
                self.scope_name.clone(),
            );

            if decode_only_flag {
                hidden_states.lift_vector();
            }

            // q [chunk_size, head_num, group_num, head_dim] <- [chunk_size, hidden_size]
            let view_query_states = query_states.view(vec![
                query_states.shape[0],
                query_states.shape[1],
                self.num_attention_heads,
                self.head_dim,
            ]);
            let view_key_states = key_states.view(vec![
                key_states.shape[0],
                key_states.shape[1],
                self.num_key_value_heads,
                self.head_dim,
            ]);

            //[batch_size, head_num, sequence_num,  head_size] < - [sequence_num, batch_size, head_num, head_size]
            let view_key_position_tensor = view_key_states.permute(vec![1, 2, 0, 3]);

            let view_value_states = value_states.view(vec![
                value_states.shape[0],
                value_states.shape[1],
                self.num_key_value_heads,
                self.head_dim,
            ]);

            let view_value_states2 = view_value_states.permute(vec![1, 2, 0, 3]);
            let thread_num = num_cpus::get().max(1);

            // [chunk_size, head_num, head_size] <- [chunk_size, head_num, group_num, head_dim] [batch_size, head_num, sequence_length, head_dim] [batch_size, head_num, sequence_length, head_dim]
            // 核心注意力输出 [sequence_length, batch_size, num_attention_heads, head_dim]，
            // out_proj + residual 后处理由 attention_block.rs 的 AttentionBlock::forward 统一完成。
            let attn_output = view_query_states.attention(
                &view_key_position_tensor,
                &view_value_states2,
                self.sequence_length,
                self.batch_size,
                self.scaling,
                decode_only_flag,
                self.is_causal,
                thread_num,
                format!("{}.attn_output", self.scope_name),
            );

            // 输出门（缺口 G2）：`attn_output * torch.sigmoid(gate)`
            // （modeling_qwen3_5.py L808）。gate 与 query 同宽，在 MatMul3 里与
            // q/k/v 一同算出。
            //
            // ZipMap 的分块契约是「total_size 行 × head_num 个 head_size 连续块」，
            // 而 total_size 就是活跃行数，故此处先把 [seq, batch, head, dim] 压成
            // 二维 [seq * batch, head * dim]（命中 `[_, cols]` 分支 → head_num = 1），
            // 再逐元素乘门，最后还原四维交给 AttentionBlock 的 out_proj。
            match gate_state {
                Some(gate) => {
                    let rows = attn_output.shape[0];
                    let cols = attn_output.shape[1];
                    let feature = self.num_attention_heads * self.head_dim;
                    let flat = attn_output.view(vec![rows * cols, feature]);
                    let gated =
                        flat.sigmoid_mul(&gate, format!("{}.attn_output_gated", self.scope_name));
                    gated.view(vec![rows, cols, self.num_attention_heads, self.head_dim])
                }
                None => attn_output,
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::mem_mgr::mem_pool::GlobalMemPool;
    use crate::runtime::{SequenceSlice, SlotState};
    use std::collections::HashMap;

    const EMPTY_SLICES: &[SequenceSlice] = &[];

    #[test]
    fn test_self_attention() {
        let sequence_length = 1;
        let batch_size = 3;
        // let hidden_size = 128;
        // let num_attention_heads = 64;
        // let num_kv_heads = 8;
        // let sequence_length = 10;

        let config = crate::config::TextConfig::load(
            r"checkpoints/Qwen3-Coder-30B-A3B-Instruct/config.json",
        )
        .unwrap();

        // let inverse_sqrt_head = 1.0 / (config.hidden_size as f32).sqrt();
        let attention_head_size: usize = config.head_dim;
        // config.hidden_size / config.num_attention_heads;

        f32::init_global(HashMap::new());
        f32::init_operator_queue();

        let self_attention = FullAttention::<f32>::new(
            &config,
            sequence_length,
            batch_size,
            true,
            "model.layers.1.self_attn",
        );

        let hidden_states = Tensor::zeros(
            vec![sequence_length, batch_size, config.hidden_size],
            String::from("model.layers.1.hidden_tensor"),
        );

        let position_embedding = Tensor::zeros(
            vec![config.max_position_embeddings, 1, 1, attention_head_size],
            String::from("model.position_embedding.weight"),
        );

        let output = self_attention.forward(&hidden_states, &position_embedding, false);

        // Add assertions to validate the output
        debug_assert_eq!(
            output.shape,
            vec![
                sequence_length,
                batch_size,
                config.num_attention_heads,
                config.head_dim
            ]
        );

        // Execute the operator queue
        let thread_num: usize = num_cpus::get();
        f32::with_operator_queue(|queue| {
            for operator in queue.iter() {
                for i in 0..thread_num {
                    operator.run(
                        batch_size,
                        0,
                        0,
                        0,
                        thread_num,
                        i,
                        EMPTY_SLICES,
                        &mut Vec::new(),
                    );
                }
            }
        });

        // Add more assertions as needed
    }

    #[test]
    fn test_self_attention_f16() {
        let sequence_length = 1;
        let batch_size = 3;

        let config = crate::config::TextConfig::load(
            r"checkpoints/Qwen3-Coder-30B-A3B-Instruct/config.json",
        )
        .unwrap();

        let attention_head_size: usize = config.head_dim;

        f16::init_global(HashMap::new());
        f16::init_operator_queue();

        let self_attention = FullAttention::<f16>::new(
            &config,
            sequence_length,
            batch_size,
            true,
            "model.layers.1.self_attn",
        );

        let hidden_states = Tensor::zeros(
            vec![sequence_length, batch_size, config.hidden_size],
            String::from("model.layers.1.hidden_tensor"),
        );

        let position_embedding = Tensor::zeros(
            vec![config.max_position_embeddings, 1, 1, attention_head_size],
            String::from("model.position_embedding.weight"),
        );

        let output = self_attention.forward(&hidden_states, &position_embedding, false);

        // Add assertions to validate the output
        debug_assert_eq!(
            output.shape,
            vec![
                sequence_length,
                batch_size,
                config.num_attention_heads,
                config.head_dim
            ]
        );

        // Execute the operator queue
        let thread_num: usize = num_cpus::get();
        f16::with_operator_queue(|queue| {
            for operator in queue.iter() {
                for i in 0..thread_num {
                    operator.run(
                        batch_size,
                        0,
                        0,
                        0,
                        thread_num,
                        i,
                        EMPTY_SLICES,
                        &mut Vec::new(),
                    );
                }
            }
        });
    }

    // ── C 类判定：qk-norm 是架构事实，判断从配置层搬进本文件，测试跟着搬 ──

    #[test]
    fn qwen3_family_has_qk_norm() {
        // 旧 `qwen3_moe::profile()` 的 `qk_norm: matches!("qwen3" | "qwen3_moe")`。
        assert!(use_qk_norm("qwen3"));
        assert!(use_qk_norm("qwen3_moe"));
        // qwen3_5 系（含嵌套子配置的 model_type）无条件带 q_norm，对应
        // `Qwen3_5Attention.__init__`。
        assert!(use_qk_norm("qwen3_5"));
        assert!(use_qk_norm("qwen3_5_text"));
        // MiniMax 的 ckpt 显式写 use_qk_norm: true，现在由默认方向兼得。
        assert!(use_qk_norm("minimax_m2"));
    }

    #[test]
    fn qwen2_and_llama_family_have_no_qk_norm() {
        // 旧为中性 false 的家族：登记在 NO_QK_NORM_MODELS。
        assert!(!use_qk_norm("qwen2"));
        assert!(!use_qk_norm("qwen2_moe"));
        assert!(!use_qk_norm("llama"));
        assert!(!use_qk_norm("mixtral"));
    }

    #[test]
    fn zero_centered_stays_off_until_verified() {
        // 缺口 G4：Python `Qwen3_5RMSNorm` 取 (1 + w)，但未经 alignment 脚本
        // 证实前不得顺手打开（旧 `profile().zero_centered_norm` 亦恒 false）。
        assert!(crate::transformer::decoder_layer::ZERO_CENTERED_MODELS.is_empty());
        assert!(!zero_centered_norm("qwen3_5_text"));
        assert!(!zero_centered_norm("qwen3"));
    }

    // ── 缺口 G2：输出门 `attn_output_gate`（Qwen3.8 阻塞项）────────────

    /// 一份带/不带输出门的最小配置（数形状与 Qwen3.8 同构，只是缩到能住进单测）。
    fn gate_config(use_gate: bool) -> TextConfig {
        TextConfig {
            model_type: "qwen3_5_text".to_string(),
            hidden_size: 512,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 128,
            rotary_dim: 128,
            max_position_embeddings: 4,
            attn_output_gate: use_gate,
            ..Default::default()
        }
    }

    /// G2 的两条可验证后果：`q_proj.weight` 行数翻倍（对应 HF
    /// `num_attention_heads * head_dim * 2`），且建图时多出一个输出门节点
    /// （对应 forward 里的 `attn_output * torch.sigmoid(gate)`）。
    ///
    /// 张量名沿用本文件既有测试的约定（`model.layers.N.*` / `model.position_embedding.weight`），
    /// 因为 MemPool 按名字正则分流，合成前缀会命中「无匹配模式」分支。
    #[test]
    fn attn_output_gate_doubles_q_proj_and_adds_gating_node() {
        f32::init_global(HashMap::new());
        f32::init_operator_queue();

        let sequence_length = 1;
        let batch_size = 1;
        let config = gate_config(true);

        let gated = FullAttention::<f32>::new(
            &config,
            sequence_length,
            batch_size,
            true,
            "model.layers.0.self_attn",
        );
        assert_eq!(
            gated.q_weight.shape[0],
            gated.num_attention_heads * gated.head_dim * GATE_ROWS,
            "attn_output_gate 时 q_proj 行数是 query 列数的两倍"
        );
        assert_eq!(gated.o_weight.shape[1], 4 * 128);

        let hidden_states = Tensor::zeros(
            vec![sequence_length, batch_size, config.hidden_size],
            String::from("model.layers.0.hidden_tensor"),
        );
        let position_embedding = Tensor::zeros(
            vec![config.max_position_embeddings, 1, 1, config.head_dim],
            String::from("model.position_embedding.weight"),
        );
        let output = gated.forward(&hidden_states, &position_embedding, false);
        // 门控只逐元素改值，输出形状不变（仍是 [seq, batch, head_num, head_dim]）。
        assert_eq!(output.shape, vec![sequence_length, batch_size, 4, 128]);

        let kinds: Vec<&str> = f32::with_operator_queue(|queue| {
            queue.iter().map(|operator| operator.kind()).collect()
        });
        assert!(
            kinds.contains(&"SigmoidMulZipMap"),
            "有门时图里必须有输出门节点：{kinds:?}"
        );

        // 无门对照：同一层造法但不加门，图里不得出现该节点，q_proj 不翻倍。
        f32::init_operator_queue();
        let plain = FullAttention::<f32>::new(
            &gate_config(false),
            sequence_length,
            batch_size,
            true,
            "model.layers.1.self_attn",
        );
        assert_eq!(plain.q_weight.shape[0], 4 * 128);
        plain.forward(&hidden_states, &position_embedding, false);
        let kinds: Vec<&str> = f32::with_operator_queue(|queue| {
            queue.iter().map(|operator| operator.kind()).collect()
        });
        assert!(
            !kinds.contains(&"SigmoidMulZipMap"),
            "无门时不得多出输出门节点：{kinds:?}"
        );
    }

    // ── 建图快照（需求 §七的第二层验证）──────────────────────────────
    //
    // `text_config.rs` 的快照锁的是**数据**（层清单、维度、专家数），本测试锁的是
    // 数据喂进判断平面后**每层的最终形态** `(是否线性注意力, 是否分配 q_norm,
    // 是否 zero_centered)`。三谓词即建图点实际使用的那三个：
    // `layer_types[i].contains("linear")`（`AttentionEnum::new` 的行内 `if`）、
    // `use_qk_norm` 与 `zero_centered_norm`（本文件与 `decoder_layer.rs` 的 C 类表）。
    // P1/P2 声明行为等价，故此处的期望值就是重构前 profile/from_hf 给出的形态。

    /// ckpt 缺失时跳过而非失败 —— 仓库不自带权重目录。
    fn load_real_ckpt(dir: &str) -> Option<TextConfig> {
        let path = format!("checkpoints/{dir}/config.json");
        if !std::path::Path::new(&path).exists() {
            eprintln!("skip build-map snapshot: {path} not found");
            return None;
        }
        Some(crate::config::TextConfig::load(&path).unwrap_or_else(|e| panic!("{path}: {e}")))
    }

    /// 每层 `(is_linear, has_qk_norm, zero_centered)`。
    fn build_map(config: &TextConfig) -> Vec<(bool, bool, bool)> {
        let has_qk_norm = use_qk_norm(&config.model_type);
        let zero_centered = zero_centered_norm(&config.model_type);
        config
            .layer_types
            .iter()
            .map(|t| (t.contains("linear"), has_qk_norm, zero_centered))
            .collect()
    }

    #[test]
    fn build_map_snapshot_four_ckpts() {
        // 三份纯因果注意力 ckpt：全 full、都带 qk-norm、都不 zero-centered。
        for (dir, layers) in [
            ("Qwen3-0.6B", 28usize),
            ("Qwen3-Coder-30B-A3B-Instruct", 48),
            ("MiniMax-M2.5", 62),
        ] {
            let Some(config) = load_real_ckpt(dir) else {
                continue;
            };
            let map = build_map(&config);
            assert_eq!(map.len(), layers, "{dir} 层数");
            for (i, &(linear, qk, zero)) in map.iter().enumerate() {
                assert!(!linear, "{dir} layer {i} 不应是线性注意力");
                assert!(qk, "{dir} layer {i} 必须分配 q_norm/k_norm");
                assert!(!zero, "{dir} layer {i}：G4 核实前不得 zero-centered");
            }
        }

        // Qwen3.8：64 层 hybrid，每 4 层最后一层才是 full_attention（interval = 4），
        // 故 48 线性 + 16 全注意力；qk-norm 只作用于那 16 个 full 层。
        let Some(config) = load_real_ckpt("Qwen3.8-27B") else {
            return;
        };
        let map = build_map(&config);
        assert_eq!(map.len(), 64);
        let linear_count = map.iter().filter(|(linear, ..)| *linear).count();
        assert_eq!(linear_count, 48, "Qwen3.8 应有 48 个 GatedDeltaNet 层");
        for (i, &(linear, qk, zero)) in map.iter().enumerate() {
            assert_eq!(linear, (i + 1) % 4 != 0, "Qwen3.8 layer {i} 形态");
            assert!(qk);
            assert!(!zero, "G4 核实前 Qwen3.8 仍按旧 profile 的 false");
        }
    }

    /// 无 qk-norm 的家族必须**不注册** `q_norm.weight` / `k_norm.weight` 两个名字，
    /// 否则严格加载模式会对 ckpt 里不存在的权重名 panic。旧代码无条件分配，
    /// 这正是 `use_qk_norm` 从未真正保护过 llama/qwen2 的原因。
    #[test]
    fn no_qk_norm_model_registers_no_norm_weight_names() {
        f32::init_global(HashMap::new());
        f32::init_operator_queue();

        for model_type in ["qwen2", "llama"] {
            let config = TextConfig {
                model_type: model_type.to_string(),
                ..gate_config(false)
            };
            let scope = format!("model.layers.3.self_attn");
            let attn = FullAttention::<f32>::new(&config, 1, 1, true, &scope);
            // 权重名是在 `Tensor::zeros(.., name)` 那一步登记进内存池的，Option
            // 为 None 即证明这两个名字从未被申请 —— 严格加载模式才不会 panic。
            assert!(
                attn.q_norm_weight.is_none() && attn.k_norm_weight.is_none(),
                "{model_type} 不该创建 q_norm/k_norm 权重及其权重名"
            );
        }

        // 对照组：默认方向（有 qk-norm）仍按名字注册，Qwen3 / MiniMax 依赖它。
        let config = TextConfig {
            model_type: "qwen3".to_string(),
            ..gate_config(false)
        };
        let attn = FullAttention::<f32>::new(&config, 1, 1, true, "model.layers.4.self_attn");
        assert!(attn.q_norm_weight.is_some() && attn.k_norm_weight.is_some());
    }
}
