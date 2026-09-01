pub trait RecurrentGatedDeltaRuleTrait<T> {
    /// One-step gated delta rule recurrence for one token row. For every
    /// v head h in [head_begin, head_end) it runs:
    ///   S_h <- exp(g_h) * S_h            gating decay of the state
    ///   e   <- v - S_h^T * k             delta error      [head_v_dim]
    ///   S_h <- S_h + beta_h * k ⊗ e      delta-rule update
    ///   o_h <- S_h^T * q                 output           [head_v_dim]
    /// The q and k segments are already l2-normalized (q additionally
    /// scaled) by the upstream CausalConv1dSilu epilogue; the q/k head of
    /// v head h is h * num_k_heads / num_v_heads (repeat_interleave).
    /// state_ptr points at this batch's
    /// [num_v_heads, head_k_dim, head_v_dim] block and is updated in
    /// place; output_ptr receives the v-segment row [value_dim].
    /// 对单个 token 行执行一步 gated delta rule 递推：对
    /// [head_begin, head_end) 内的每个 v 头 h 执行：
    ///   S_h <- exp(g_h) * S_h            状态门控衰减
    ///   e   <- v - S_h^T * k             delta 误差        [head_v_dim]
    ///   S_h <- S_h + beta_h * k ⊗ e      delta 规则更新
    ///   o_h <- S_h^T * q                 输出              [head_v_dim]
    /// q 与 k 段已由上游 CausalConv1dSilu 的 epilogue 做 l2 归一化
    /// （q 额外带缩放）；v 头 h 对应的 q/k 头为
    /// h * num_k_heads / num_v_heads（对应 repeat_interleave）。
    /// state_ptr 指向当前 batch 的
    /// [num_v_heads, head_k_dim, head_v_dim] 块，原地更新；
    /// output_ptr 写入该行输出 [value_dim]。
    fn compute(
        &self,
        qkv_row_ptr: *const T, // one token row: [2 * key_dim + value_dim], q | k | v
        g_row_ptr: *const T,   // log gating decay: [num_v_heads]
        beta_row_ptr: *const T, // delta-rule learning rate: [num_v_heads]
        state_ptr: *mut T,     // [num_v_heads, head_k_dim, head_v_dim], updated in place
        output_ptr: *mut T,    // [value_dim]
        head_begin: usize,     // first v head of this thread's block (inclusive)
        head_end: usize,       // last v head of this thread's block (exclusive)
    );
}
