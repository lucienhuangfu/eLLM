## ✅ Important
* The project is under active development and is expected to be released in about a month!  
* We are currently looking for volunteers — if you're interested, please contact lucienhuangfu@outlook.com.


# eLLM: Achieving Lossless Million-Token LLM Inference on CPUs

* **eLLM** is an inference framework that enables **lossless** large language models (LLMs) on **CPU-only machines**, 
* supporting **sequences up to millions of tokens**. 
* It outperforms GPU-based inference under comparable per-user cost .

## 🚀 Key Features
* **Elastic computation graph** removes the overhead of dynamic graph construction.
* **Contiguous KV cache memory layout** for efficient hardware prefetching.


## 🧠 Supported Models

* [x] LLaMA 2 / 3
* [ ] Qwen (coming soon)
* [ ] DeepSeek (coming soon)

## 📄 Paper

If you're interested in the technical details, check out our paper and cite:

```bibtex
@article{ellm2025,
  title={eLLM: Achieving Lossless Million-Token LLM Inference on CPUs Faster Than GPUs},
  author={Yaguang Huangfu},
  journal={preprint https://www.researchgate.net/publication/393416965},
  year={2025}
}
```

## 🧪 Applications

* Deep research and thinking(multi-document reading)
* Long-form QA and document synthesis
* Fiction generation with long contexts
* Offline inference tasks


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




