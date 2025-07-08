# eLLM: Achieving Lossless Million-Token LLM Inference on CPUs

**eLLM** is an inference framework that enables **lossless** large language models (LLMs) on **CPU-only machines**, supporting **sequences up to millions of tokens**. It outperforms GPU-based inference under comparable per-user cost .

## 🚀 Key Features

* **Million-token inference** on CPUs without memory fragmentation.
* **No GPUs required** — built for general-purpose CPU servers.
* **Elastic computation graph** removes the overhead of dynamic graph construction.
* **Contiguous KV cache memory layout** for efficient hardware prefetching.
* **Competitive or superior performance** to GPU-based systems on long-sequence tasks.

## 🧠 Supported Models

* [x] LLaMA 2 / 3
* [ ] Qwen (coming soon)
* [ ] DeepSeek (coming soon)

## 📄 Paper

If you're interested in the technical details, check out our paper and cite:

> **eLLM: Make Million-Token LLM Inference on CPUs Faster Than on GPUs — Losslessly**
> \[arXiv / conference link coming soon]

If you find eLLM useful in your research, please cite:

```bibtex
@article{ellm2025,
  title={eLLM: Achieving Lossless Million-Token LLM Inference on CPUs Faster Than GPUs},
  author={Yaguang Huangfu},
  journal={preprint https://www.researchgate.net/publication/393416965},
  year={2025}
}
```

## 🧪 Applications

* Long-form QA and document synthesis
* Deep research assistants (multi-document reading)
* Fiction generation with long contexts
* Asynchronous/offline inference tasks
* AI PCs and edge inference (batch size = 1)


## 🔧 Usage

Todo


## 💻 Installation

Todo



## 📦 Coming Soon

* Docker support
* Multi-CPU parallelism
* Benchmark scripts
* More model compatibility

## 📜 License

This project is licensed under the [Apache 2.0 License](LICENSE).




