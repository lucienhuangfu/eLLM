pub trait AttentionAddTrait<T> {
    fn compute(
        &self,
        q_ptr1: *const T,
        k_ptr2: *const T,
        v_ptr3: *const T,
        residual_ptr: *const T,
        output_ptr: *mut T,
        position: usize,
    );
}

pub trait MatMulTrait<T> {
    fn compute(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T);
    fn compute2(
        &self,
        input_ptr1: *const T,
        input_ptr2: *const T,
        output_ptr: *mut T,
        length: usize,
    );
}

pub trait MatMulAddTrait<T> {
    fn compute(
        &self,
        input_ptr1: *const T,
        input_ptr2: *const T,
        input_ptr3: *const T,
        output_ptr: *mut T,
    );
}

pub trait MatMulTopKTrait<T> {
    fn compute(
        &self,
        input_ptr1: *const T,
        input_ptr2: *const T,
        value_ptr: *mut T,
    );
}

// === runner/mul_trait.rs ===
pub trait MatMul3Trait<T> {
    fn compute1(
        &self,
        input_ptr1: *const T,
        input_ptr2: *const T,
        input_ptr3: *const T,
        output_ptr1: *mut T,
        output_ptr2: *mut T,
    );

    fn compute2(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T);
}
// === mul_trait.rs 中新增 ===

pub trait SiluMatmulTrait<T> {
    /// K 方向逐 kc 累加：
    ///   gate_acc/up_acc += A_tile(3×kc) × panel(kc×32)
    fn compute1(
        &self,
        a_tile: *const T,      // A tile: 3×kc，行距=K
        gate_panel: *const T,  // gate 面板: kc×32
        up_panel: *const T,    // up   面板: kc×32
        gate_acc: *mut T,      // gate 累加器: 3×32
        up_acc: *mut T,        // up   累加器: 3×32
    );

    /// 整个 K 累加完成后的收尾：
    ///   C_tile = SiLU(gate_acc) ⊙ up_acc
    fn compute2(
        &self,
        gate_acc: *const T,    // gate 累加器: 3×32
        up_acc: *const T,      // up   累加器: 3×32
        c_tile: *mut T,        // 输出 C tile: 3×32
    );
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

    fn compute2(
        &self,
        c_head: *mut T,
        rope_head: *const T,
        ldc: usize,
    );
}

// === runner/mul_trait.rs ===
pub trait MatMul5Trait<T> {
    fn compute(
        &self,
        input_ptr1: *const T,
        input_ptr2: *const T,
        input_ptr3: *const T,
        input_ptr4: *const T,
        input_ptr5: *const T,
        output_ptr: *mut T,
    );
}
// === mul_trait.rs 中新增 ===

/// MoE down projection 专用：
/// - compute1: 把 A_tile(3×kc) × B_panel(kc×32) 累加到 acc(3×32)
/// - compute2: 对一行 acc_row 乘以 factor 并加到 out_row 上
pub trait ExpertsDownTrait<T> {
    /// GEMM 微核：acc += A_tile × B_panel
    fn compute1(
        &self,
        a_tile: *const T,   // MR×KC（行主，行距=KC）
        b_panel: *const T,  // KC×NR（行主，每行 NR 连续）
        acc: *mut T,        // MR×NR（行主，行距=NR）
    );

    /// 缩放 + 累加：out_row[j] += acc_row[j] * factor，j in [0, len)
    fn compute2(
        &self,
        out_row: *mut T,    // OUT[b, slot, n0..] 的起点
        acc_row: *const T,  // acc 中第 r 行的起点
        factor: *const T,   // 标量因子指针
        len: usize,         // 当前 tile 宽度（n_blk）
    );
}
// === mul_trait.rs 中新增 ===

/// MoE merge（最后一步）专用：
/// 对一整行做 out[h] += add[h]。
pub trait MoeMergeTrait<T> {
    /// 行内加法：out_row[h] += add_row[h], h in [0, len)
    fn merge_add(
        &self,
        out_row: *mut T,    // 输出行（OUT[t, :]），原位累加
        add_row: *const T,  // 要加的行（一个 expert 的输出）
        len: usize,         // hidden_size
    );
}