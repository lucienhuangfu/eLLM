// === operators/elementwise/traits.rs ===
// Shared micro-kernel interfaces of the elementwise family: one map
// (single input) and one zip (two inputs). Implemented across
// elementwise/, rms.rs, movement/lookup_rms.rs and softmax/sigmoid_map.rs.
// elementwise 家族的共享微内核接口：单输入 map 与双输入 zip。
// 实现方分布在 elementwise/、rms.rs、movement/lookup_rms.rs 与
// softmax/sigmoid_map.rs。

/// Two-input elementwise (zip) micro-kernel interface.
/// 双输入逐元素（zip）微内核接口。
pub trait ZipMapTrait<T> {
    fn compute(&self, input_ptr1: *const T, input_ptr_2: *const T, output_ptr: *mut T);
}

/// One-input elementwise (map) micro-kernel interface.
/// 单输入逐元素（map）微内核接口。
pub trait MapTrait<T> {
    fn compute(&self, input_ptr: *const T, output_ptr: *mut T, length: usize);
}
