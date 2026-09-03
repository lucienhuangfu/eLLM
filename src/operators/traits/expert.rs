pub trait ExpertsDownTrait<T> {
    fn compute1(&self, a_tile: *const T, b_panel: *const T, acc: *mut T);

    fn compute1_single(&self, input_row: *const T, b_panel: *const T, acc: *mut T, kc: usize);

    fn compute1_rows(
        &self,
        a_tile: *const T,
        b_panel: *const T,
        acc: *mut T,
        kc: usize,
        rows: usize,
    );

    fn compute2(&self, out_row: *mut T, acc_row: *const T, factor: *const T, len: usize);
}

pub trait ExpertsSiluTrait<T> {
    fn compute1(
        &self,
        a_tile: *const T,
        gate_panel: *const T,
        up_panel: *const T,
        gate_acc: *mut T,
        up_acc: *mut T,
        kc: usize,
    );

    fn compute1_single(
        &self,
        input_row: *const T,
        gate_panel: *const T,
        up_panel: *const T,
        gate_acc: *mut T,
        up_acc: *mut T,
        kc: usize,
    );

    fn compute1_rows(
        &self,
        a_tile: *const T,
        gate_panel: *const T,
        up_panel: *const T,
        gate_acc: *mut T,
        up_acc: *mut T,
        kc: usize,
        rows: usize,
    );

    fn compute2(&self, gate_row: *const T, up_row: *const T, c_row: *mut T);
}

pub trait MoeMergeTrait<T> {
    fn merge_add(&self, out_row: *mut T, add_row: *const T, len: usize);
}

/* ------------------------------------------------------------------ */
/* Shared-expert fused operators: dense shared branch first, then the  */
/* existing sparse routed branch. Scalar compute only (no f16 / SIMD). */
/* shared expert 融合算子：先算稠密 shared 分支，再算现有 sparse 路由分支。 */
/* 仅提供标量 compute（不做 f16 / SIMD 特化）。                            */
/* ------------------------------------------------------------------ */

/// Shared + routed gate/up projection fused with SiLU(gate) * up.
/// shared 与 routed 的 gate/up 投影，融合 SiLU(gate) * up。
///
/// Signatures mirror [`ExpertsSiluTrait`]; a concrete scalar implementation
/// lives in the operator file where the tile geometry (`self.params`) is
/// accessible.
/// 签名与 [`ExpertsSiluTrait`] 一致；标量实现位于算子文件内，
/// 以便访问 tile 几何参数（`self.params`）。
pub trait SharedExpertsSiluTrait<T> {
    fn compute1(
        &self,
        a_tile: *const T,
        gate_panel: *const T,
        up_panel: *const T,
        gate_acc: *mut T,
        up_acc: *mut T,
        kc: usize,
    );

    fn compute1_single(
        &self,
        input_row: *const T,
        gate_panel: *const T,
        up_panel: *const T,
        gate_acc: *mut T,
        up_acc: *mut T,
        kc: usize,
    );

    fn compute1_rows(
        &self,
        a_tile: *const T,
        gate_panel: *const T,
        up_panel: *const T,
        gate_acc: *mut T,
        up_acc: *mut T,
        kc: usize,
        rows: usize,
    );

    fn compute2(&self, gate_row: *const T, up_row: *const T, c_row: *mut T);
}

/// Shared + routed down projection.
/// shared 与 routed 的 down 投影。
///
/// Signatures mirror [`ExpertsDownTrait`].
/// 签名与 [`ExpertsDownTrait`] 一致。
pub trait SharedExpertsDownTrait<T> {
    fn compute1(&self, a_tile: *const T, b_panel: *const T, acc: *mut T);

    fn compute1_single(&self, input_row: *const T, b_panel: *const T, acc: *mut T, kc: usize);

    fn compute1_rows(
        &self,
        a_tile: *const T,
        b_panel: *const T,
        acc: *mut T,
        kc: usize,
        rows: usize,
    );

    fn compute2(&self, out_row: *mut T, acc_row: *const T, factor: *const T, len: usize);
}

/// Merge routed expert outputs, residual, and the gated shared-expert output.
/// 合并 routed expert 输出、residual 以及带门控的 shared expert 输出。
pub trait SharedMergeAddTrait<T> {
    /// out_row[j] += add_row[j].
    fn merge_add(&self, out_row: *mut T, add_row: *const T, len: usize);

    /// out_row[j] += add_row[j] * factor.
    fn merge_add_scaled(&self, out_row: *mut T, add_row: *const T, factor: T, len: usize);
}
