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
        sequence_index: usize,
        k_seq_stride: usize,
        v_seq_stride: usize,
        q_seq_stride: usize,
        running_max: &mut [T],
        running_denom: &mut [T],
        scores: &mut [T],
    );

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
