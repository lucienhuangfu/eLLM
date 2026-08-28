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

    fn compute_rows(
        &self,
        input_row: *const T,
        weight_panel: *const T,
        output_row: *mut T,
        kc: usize,
        rows: usize,
    );
}

pub trait MatMulSigmoidTrait<T> {
    fn compute(&self, m0: usize, n0: usize, m_blk: usize, n_blk: usize, thread_id: usize);
}

// Dedicated trait for the fused input projection (MatMulProj).
// Provides three compute variants: plain GEMM, sigmoid epilogue on the
// B (weight) side, and sigmoid epilogue on the A (input) side.
// MatMulProj 融合输入投影的专用 trait。
// 提供三种 compute 变体：纯乘法、sigmoid 作用于 B 矩阵、sigmoid 作用于 A 矩阵。
pub trait MatMulProjTrait<T> {
    // Plain GEMM: C = A @ B_nt^T.
    // 纯乘法：C = A @ B_nt^T。
    fn compute(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T);

    // GEMM with the b-segment epilogue fused in place:
    //   beta[h] = sigmoid(b[row, h])
    // 乘法混合 sigmoid，作用于 B 矩阵（b 段），原地得到 beta：
    //   beta[h] = sigmoid(b[row, h])
    fn compute_sigmoid_b(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T);

    // GEMM with the a-segment epilogue fused in place:
    //   g[row, h] = -exp(A_log[h]) * softplus(a[row, h] + dt_bias[h])
    // 乘法混合门控，作用于 A 矩阵（a 段），原地得到 g：
    //   g[row, h] = -exp(A_log[h]) * softplus(a[row, h] + dt_bias[h])
    fn compute_sigmoid_a(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T);
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
