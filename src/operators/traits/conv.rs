pub trait CausalConvTrait<T> {
    /// Fused depthwise causal conv1d + activation for one token row.
    /// Reads the per-channel rolling window, convolves with the channel
    /// weights, applies the activation, and updates the window in place.
    /// 对单个 token 行做融合的 depthwise 因果卷积 + 激活：
    /// 读取逐通道滚动窗口，与通道权重卷积，应用激活，并原地更新窗口。
    fn compute(
        &self,
        input_ptr: *const T,  // one token row: [conv_dim]
        weight_ptr: *const T, // depthwise weight: [conv_dim, kernel_size]
        state_ptr: *mut T,    // rolling window: [conv_dim, kernel_size - 1], updated in place
        output_ptr: *mut T,   // [conv_dim]
    );
}
