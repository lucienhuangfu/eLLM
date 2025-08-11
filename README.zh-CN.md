# eLLM：是一款运行在单路 CPU-only 服务器上的大语言模型推理框架。它：
- 运行满血MoE大模型（Qwen3-480B），具备实时短文本推理能力（100ms/token）
- 支持百万 tokens 的长文本深度思考与理解

🌐 语言版本：[English](README.md) | [简体中文](README.zh-CN.md)

##✅ 重要提示
* 项目正在积极开发中，预计将在 **约 1 个月后** 发布最小原型！  
* 我们目前正在寻找志愿者——如果你感兴趣，请联系 **lucienhuangfu@outlook.com**。

**关键能力**：

* 支持 MoE 模型完整加载，动态专家激活
* 完整支持百万 tokens 上下文 (KV Cache)
* 标准 Attention 推理（token 与全文深度关联）

## 应用场景 1：在线短文本推理
* 搜索问答
* 代码补全
* 客服对话

## 应用场景 2：离线长文本推理（Deep Research）
- 代码审计 / 高危漏洞查找
- 合同审核 / 文档审查  
- 财务报表合规检查
- 文学作品创作 / 延续式写作 

## 竞品对比：eLLM 降低大模型普及门槛

**eLLM 让中小团队也能轻松部署大模型，成本更低、部署更灵活。**

### 无需高性能 GPU 服务器
- 单路CPU-only服务器即可运行 MoE 架构大模型  
- 仅需支持 AVX512-F16 的通用 CPU  
- 可通过加装 DDR5 模块扩展内存 

### 部署简单，适配多种场景
- 本地服务器 / 私有云 / 边缘节点轻松部署  
- 支持按需弹性计算，任务完成后自动释放资源  
- 横向扩展架构，满足高并发推理需求

机器对比：CPU-only 服务器 vs GPU 服务器

| CPU-only 服务器 | 条目 | GPU 服务器 | |
|----------|--------------|------------|------|
|CPU ||CPU|GPU| 
| Xeon 6900| 型号           |   Xeon 8480+     | H20   |
|3|内存容量(TB)|2|0.141|
| 1| 数量          |4        | 8  |
|15|总价(万元) |150| 


## 路线图
* [ ] Qwen (30B，480B)（即将支持）
* [x] LLaMA 2 / 3
* [ ] DeepSeek（即将支持）
* [ ] gpt-oss

## 📄 论文

如果你对技术细节感兴趣，可以阅读我们的论文并引用：

```bibtex
@article{ellm2025,
  title={eLLM: Achieving Lossless Million-Token LLM Inference on CPUs Faster Than GPUs},
  author={Yaguang Huangfu},
  journal={preprint https://www.researchgate.net/publication/393416965},
  year={2025}
}
