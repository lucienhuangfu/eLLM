# 05 · 服务初始化

> 对应原 `serving.md` 章节：§7 服务初始化。

## 1. 入口

`initialize_serving_resources(config)` 在 `resources.rs` 中完成所有组件的初始化。

## 2. 初始化步骤

按顺序执行：

1. 加载模型配置（`config.json`）和生成配置（`generation_config.json`）
2. 通过 `SafeTensorsLoader` 加载权重，写入全局内存池
3. 提取采样参数（top_k、top_k_simd、top_p、min_p、do_sample、eos_token_id_list）
4. 确定线程配置（worker_threads = max(total - async_threads, 1)，async_threads = 2）
5. 构建 `BatchSequence`（tokenizer + 序列缓冲区）
6. 构建 `batch_states`（每个槽位的 `SequenceState`）
7. 创建调度组件（`BatchScheduler` + `TokenCounter` + broadcast channel）
8. 创建 RoPE 位置编码，初始化模型，执行一次前向推理（填充算子队列）
9. 创建 `ServingRunner`（从全局队列取出算子列表）

## 3. 返回值

返回 `ServingResources`，包含所有运行时组件。
