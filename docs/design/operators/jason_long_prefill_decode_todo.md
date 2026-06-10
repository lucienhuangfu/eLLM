# Jason: 长 Prefill 后 Decode 加速 TODO

## 背景

长 prompt 下 prefill 已经通过 GQA8 attention 融合和 AVX512 向量化 softmax 明显下降，但 decode 仍然会随着 prompt 长度上升而变慢。

当前主要现象：

* decode 每步 attention 需要扫描已有 KV cache，prompt 越长，每个 token 的 decode attention 越贵。
* 现有短 slice/decode 路径主要按 head split；Qwen3-Coder-30B-A3B 只有 32 个 Q heads，在 48 线程机器上天然会有线程覆盖不足。
* decode 路径目前没有像长 prefill 一样充分利用 `1 KV head -> 8 Q heads` 的 GQA8 融合。
* 如果简单按 KV column 再切分，需要做 partial softmax 的跨线程归约，否则会破坏 attention 数值语义。

## 下一步方向

已验证但暂不默认开启的方向：

* `ELLM_ATTENTION_DECODE_GQA8=1`：decode 时按 `kv_head` 一次计算 8 个 Q heads。
* 4000 prompt / 16 output 的 step1 profile 中，单纯 GQA8 decode group 让 attention 从约 `0.0385s` 变为约 `0.0482s`。
* 原因是它只暴露 4 个 `kv_head` task，48 线程机器上并行度损失大于 K/V 复用收益。
* 结论：decode 不能只做 GQA8 group fusion，还需要 KV column split 和 partial softmax merge 才可能同时保留并行度和 K/V 复用。

1. 给 decode attention 增加更细 profile：
   * 区分 QK dot、softmax、V accumulate。
   * 打印 decode step 和当前 KV 长度。
   * 对比 prompt 1000/2000/3000/4000 下每步 attention 成本增长。

2. 设计 decode GQA8 kernel：
   * 一个 `kv_head` 下同时计算 8 个 Q heads。
   * 复用同一 K/V 行，减少重复 K/V 读取。
   * 优先保持单线程完整 softmax，先验证内核收益。

3. 再考虑 KV column split：
   * 每个线程处理一段 KV 列。
   * 每段输出 partial max、partial denom、partial weighted V。
   * 通过稳定 online softmax merge 合并 partial 结果。
   * 只有在 GQA8 单线程/少线程路径仍不足时再引入跨线程归约。

4. 检查 KV cache 访问布局：
   * 确认 decode 时 K/V stride 连续性。
   * 确认同一个 `kv_head` 的 K/V 读取不会被 8 个 Q heads 重复扫 8 次。
   * 对长 prompt 下 L2/L3 miss 和内存带宽做单独 profile。

## 验收指标

* 4000 prompt / 512 output 下 decode 总时间下降。
* prompt 从 1000 增到 4000 时，decode 时间斜率降低。
* 输出开头语义正常，不出现明显对齐错乱。
* 不牺牲当前 long prefill 的 GQA8 + AVX512 softmax 优化。
