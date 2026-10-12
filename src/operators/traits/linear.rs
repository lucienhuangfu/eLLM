pub trait AttentionTrait<T> {
    #[allow(clippy::too_many_arguments)]
    fn compute(
        &self,
        q_ptr1: *const T,
        k_ptr2: *const T,
        v_ptr3: *const T,
        output_ptr: *mut T,
        row_begin: usize,
        row_end: usize,
        col_begin: usize,
        col_end: usize,
        total_col_end: usize,
        next_sequence_index: usize,
        k_seq_stride: usize,
        v_seq_stride: usize,
        q_seq_stride: usize,
        running_max: &mut [T],
        running_denom: &mut [T],
        scores: &mut [T],
    );

    /// Non-causal (bidirectional) counterpart of [`compute`]: every query row
    /// may attend to all columns up to `total_col_end`, with no triangular mask.
    /// 非因果（双向）版本的 [`compute`]：每个 query 行可见 `total_col_end` 之前的
    /// 全部列，不做三角掩码。
    #[allow(clippy::too_many_arguments)]
    fn compute_non_causal(
        &self,
        _q_ptr1: *const T,
        _k_ptr2: *const T,
        _v_ptr3: *const T,
        _output_ptr: *mut T,
        _row_begin: usize,
        _row_end: usize,
        _col_begin: usize,
        _col_end: usize,
        _total_col_end: usize,
        _next_sequence_index: usize,
        _k_seq_stride: usize,
        _v_seq_stride: usize,
        _q_seq_stride: usize,
        _running_max: &mut [T],
        _running_denom: &mut [T],
        _scores: &mut [T],
    ) {
        // TODO: 非因果（双向）attention：每行可见列 = total_col_end，不做三角掩码；
        // 实现时须绕开 causal_col_prune / 三角切分 / gqa8 等因果专用路径，改均匀切分 + 全列可见。
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_gqa8(
        &self,
        _q_group_ptr: *const T,
        _k_head_ptr: *const T,
        _v_head_ptr: *const T,
        _output_group_ptr: *mut T,
        _row_begin: usize,
        _row_end: usize,
        _total_col_end: usize,
        _sequence_index: usize,
        _k_seq_stride: usize,
        _v_seq_stride: usize,
        _q_seq_stride: usize,
    ) -> bool {
        false
    }
}

pub trait MatMulTrait<T> {
    fn compute(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T);

    fn compute2(
        &self,
        _input_ptr1: *const T,
        _input_ptr2: *const T,
        _output_ptr: *mut T,
        _length: usize,
    ) {
        unreachable!("MatMulTrait::compute2 is not implemented for this operator")
    }
}

pub trait MatMulAddTrait<T> {
    fn compute(
        &self,
        input_ptr1: *const T,
        input_ptr2: *const T,
        input_ptr3: *const T,
        output_ptr: *mut T,
    );

    fn compute_init(
        &self,
        input_ptr1: *const T,
        weight_panel: *const T,
        residual_ptr: *const T,
        output_ptr: *mut T,
        kc: usize,
    );

    fn compute_rows(
        &self,
        input_row: *const T,
        weight_panel: *const T,
        output_row: *mut T,
        kc: usize,
        rows: usize,
    );

    fn compute_rows_init(
        &self,
        input_row: *const T,
        weight_panel: *const T,
        residual_row: *const T,
        output_row: *mut T,
        kc: usize,
        rows: usize,
    );
}

pub trait MatMulSigmoidTrait<T> {
    fn compute(&self, m0: usize, n0: usize, m_blk: usize, n_blk: usize, thread_id: usize);
}

pub trait MatMulkqvTrait<T> {
    fn compute1(
        &self,
        a: *const T,
        b_panel: *const T,
        c: *mut T,
        lda: usize,
        ldc: usize,
        kc: usize,
    );

    /// compute1_init: C = A × B_panel（first K-block, no prior C load）.
    /// Eliminates the external zero-init pass for the 3-row tile path.
    fn compute1_init(
        &self,
        a: *const T,
        b_panel: *const T,
        c: *mut T,
        lda: usize,
        ldc: usize,
        kc: usize,
    );

    fn compute2(&self, c_head: *mut T, rope_head: *const T, ldc: usize);

    fn compute_norm_rope(
        &self,
        c_head: *mut T,
        norm_weight: *const T,
        rope_head: *const T,
        length: usize,
        eps: T,
    );

    fn compute_head_gemv(
        &self,
        a_row: *const T,
        dst_head: *mut T,
        packed_b: *const T,
        head_output_panel: usize,
        output_panel_count: usize,
        reduction_cols: usize,
        reduction_block_cols: usize,
        micro_tile_cols: usize,
        head_dim: usize,
    );
}

pub trait MatMulProjTrait<T> {
    // Plain GEMM on one output segment: C = A @ B_nt^T over output_cols columns.
    // 对一个输出段做纯乘法：C = A @ B_nt^T，共 output_cols 列。
    fn compute(
        &self,
        input_ptr1: *const T,
        input_ptr2: *const T,
        output_ptr: *mut T,
        output_cols: usize,
    );

    // GEMM with the b-segment epilogue fused in place:
    //   beta[h] = sigmoid(b[row, h])
    // 乘法混合 sigmoid，作用于输出的 b 段，原地得到 beta：
    //   beta[h] = sigmoid(b[row, h])
    fn compute_sigmoid_b(
        &self,
        input_ptr1: *const T,
        input_ptr2: *const T,
        output_ptr: *mut T,
        output_cols: usize,
    );

    // GEMM with the a-segment epilogue fused in place:
    //   g[row, h] = -exp(A_log[h]) * softplus(a[row, h] + dt_bias[h])
    // 乘法混合衰减门（softplus + exp 缩放），作用于输出的 a 段，原地得到 g：
    //   g[row, h] = -exp(A_log[h]) * softplus(a[row, h] + dt_bias[h])
    fn compute_gate_a(
        &self,
        input_ptr1: *const T,
        input_ptr2: *const T,
        output_ptr: *mut T,
        output_cols: usize,
    );
}
