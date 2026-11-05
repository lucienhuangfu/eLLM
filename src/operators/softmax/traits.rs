// === operators/softmax/traits.rs ===
// Shared micro-kernel interface of the softmax family. The implementing
// operator ExpertSoftmaxNorm lives under expert/, but the semantics belong
// to softmax, so the trait stays here.
// softmax 家族的共享微内核接口。实现算子 ExpertSoftmaxNorm 位于 expert/
// 目录，但语义属于 softmax 家族，因此 trait 放在这里。

/// Softmax + top-k normalize micro-kernel interface.
/// softmax + top-k 归一化的微内核接口。
pub trait SoftmaxTrait<T> {
    fn compute(
        &self,
        ptr1: *const T,
        topk_values_ptr: *mut T,
        topk_indices_ptr: *mut usize,
        input_length: usize,
        output_length: usize,
    );
}
