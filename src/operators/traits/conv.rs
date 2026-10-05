pub trait CausalConvTrait<T> {
    /// Fused depthwise causal conv1d + activation for one token row.
    /// Reads the per-channel rolling window, convolves with the channel
    /// weights, applies the activation, and updates the window in place.
    /// Channels are independent, so the caller may restrict the work to
    /// the range [channel_begin, channel_end); the pointers stay anchored
    /// at the row start and the callee offsets them internally.
    /// 对单个 token 行做融合的 depthwise 因果卷积 + 激活：
    /// 读取逐通道滚动窗口，与通道权重卷积，应用激活，并原地更新窗口。
    /// 通道间无依赖，调用方可将工作限定在 [channel_begin, channel_end)；
    /// 指针仍指向行首，由实现内部自行偏移。
    fn compute(
        &self,
        input_ptr: *const T,  // one token row: [conv_dim]
        weight_ptr: *const T, // depthwise weight: [conv_dim, kernel_size]
        state_ptr: *mut T,    // rolling window: [conv_dim, kernel_size - 1], updated in place
        output_ptr: *mut T,   // [conv_dim]
        channel_begin: usize, // first channel of this thread's block (inclusive)
        channel_end: usize,   // last channel of this thread's block (exclusive)
    );

    /// Per-head l2-norm (+ query scale) epilogue for one token row, applied
    /// in place after compute. Every head of the q segment [0, key_dim) and
    /// the k segment [key_dim, 2 * key_dim) that lies fully inside
    /// [channel_begin, channel_end) is normalized along its head_k_dim
    /// channels: x *= rsqrt(sum(x^2) + eps); the q heads additionally take
    /// the attention scale 1 / sqrt(head_k_dim). The value segment and any
    /// partial head at the range edges are left untouched. Channel blocks
    /// are head-aligned by scheduling, so every head belongs to exactly one
    /// block.
    /// 对单个 token 行的逐头 l2 归一化（+ query 缩放）epilogue，在 compute
    /// 之后原地执行：q 段 [0, key_dim) 与 k 段 [key_dim, 2 * key_dim) 中
    /// 完整落在 [channel_begin, channel_end) 内的每个头沿其 head_k_dim 个
    /// 通道归一化：x *= rsqrt(sum(x^2) + eps)；q 头额外乘注意力缩放
    /// 1 / sqrt(head_k_dim)。value 段与范围边缘的不完整头不动。
    /// 调度保证通道块按头对齐，每个头恰好属于一个块。
    fn norm_scale_qk(
        &self,
        _qkv_row_ptr: *mut T, // one token row: [conv_dim], normalized in place
        _channel_begin: usize,
        _channel_end: usize,
    ) {
    }
}
