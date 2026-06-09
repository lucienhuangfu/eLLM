# `serve`

`vllm serve` 命令用于启动一个 OpenAI 兼容的推理服务。

## JSON CLI 参数

传递 JSON CLI 参数时，以下参数集是等效的：

- `--json-arg '{"key1": "value1", "key2": {"key3": "value2"}}'`
- `--json-arg.key1 value1 --json-arg.key2.key3 value2`

此外，可以使用 `+` 单独传递列表元素：

- `--json-arg '{"key4": ["value3", "value4", "value5"]}'`
- `--json-arg.key4+ value3 --json-arg.key4+='value4,value5'`

## 参数

### 基础参数

#### `--headless`

以无头模式运行。有关多节点数据并行的详细信息，请参阅多节点数据并行文档。

默认值: `False`

#### `--api-server-count`, `-asc`

要运行的 API 服务器进程数。如果未指定，默认为 `data_parallel_size`。

#### `--config`

从配置文件读取 CLI 选项。必须是包含以下选项的 YAML 文件：https://docs.vllm.ai/en/latest/configuration/serve_args.html

#### `--grpc`

启动 gRPC 服务器而不是 HTTP OpenAI 兼容服务器。需要安装：`pip install vllm[grpc]`。

默认值: `False`

#### `--disable-log-stats`

禁用日志统计。

默认值: `False`

#### `--aggregate-engine-logging`

使用数据并行时记录聚合统计信息而不是每个引擎的统计信息。

默认值: `False`

#### `--fail-on-environ-validation`, `--no-fail-on-environ-validation`

如果设置，引擎将在环境验证失败时引发错误。

默认值: `False`

#### `--shutdown-timeout`

关闭超时（秒）。0 = 中止，>0 = 等待。

默认值: `0`

#### `--gdn-prefill-backend`

可能的选择: `flashinfer`, `triton`, `cutedsl`

选择 GDN prefill 后端。

#### `--enable-log-requests`, `--no-enable-log-requests`

启用请求信息日志记录，取决于日志级别：
- INFO: 请求 ID、参数和 LoRA 请求
- DEBUG: 提示输入（例如：文本、token ID）

您可以通过 `VLLM_LOGGING_LEVEL` 设置最低日志级别。

默认值: `False`

### Frontend

OpenAI 兼容前端服务器的参数。

#### `--lora-modules`

LoRA 模块配置。

#### `--chat-template`

聊天模板。

#### `--chat-template-content-format`

可能的选择: `auto`, `openai`, `string`

默认值: `auto`

#### `--trust-request-chat-template`, `--no-trust-request-chat-template`

默认值: `False`

#### `--default-chat-template-kwargs`

应该是有效的 JSON 字符串或单独传递的 JSON 键。

#### `--response-role`

默认值: `assistant`

#### `--return-tokens-as-token-ids`, `--no-return-tokens-as-token-ids`

默认值: `False`

#### `--enable-auto-tool-choice`, `--no-enable-auto-tool-choice`

默认值: `False`

#### `--exclude-tools-when-tool-choice-none`, `--no-exclude-tools-when-tool-choice-none`

默认值: `False`

#### `--tool-call-parser`

工具调用解析器。

#### `--tool-parser-plugin`

默认值: `""`

#### `--tool-server`

工具服务器配置。

#### `--log-config-file`

日志配置文件。

#### `--max-log-len`

最大日志长度。

#### `--enable-prompt-tokens-details`, `--no-enable-prompt-tokens-details`

默认值: `False`

#### `--enable-server-load-tracking`, `--no-enable-server-load-tracking`

默认值: `False`

#### `--enable-force-include-usage`, `--no-enable-force-include-usage`

默认值: `False`

#### `--enable-tokenizer-info-endpoint`, `--no-enable-tokenizer-info-endpoint`

默认值: `False`

#### `--enable-log-outputs`, `--no-enable-log-outputs`

默认值: `False`

#### `--enable-log-deltas`, `--no-enable-log-deltas`

默认值: `True`

#### `--log-error-stack`, `--no-log-error-stack`

默认值: `False`

#### `--tokens-only`, `--no-tokens-only`

默认值: `False`

#### `--fingerprint-mode`

可能的选择: `custom`, `full`, `hash`, `none`

默认值: `full`

#### `--fingerprint-value`

指纹值。

#### `--host`

主机名。

#### `--port`

端口号。

默认值: `8000`

#### `--data-parallel-supervisor-port`

多端口外部负载均衡模式下聚合健康端点的 HTTP 端口。

默认值: `9256`

#### `--dp-supervisor-probe-interval-s`

多端口外部负载均衡模式下聚合健康探测之间的秒数。

默认值: `5.0`

#### `--dp-supervisor-probe-timeout-s`

多端口外部负载均衡模式下，子健康探测因连接错误失败时重试之间等待的秒数。

默认值: `5.0`

#### `--dp-supervisor-probe-failure-threshold`

多端口外部负载均衡模式下，连续连接错误重试次数达到此阈值时，子健康探测被声明为失败。

默认值: `3`

#### `--uds`

Unix 域套接字路径。如果设置，将忽略 host 和 port 参数。

#### `--uvicorn-log-level`

可能的选择: `critical`, `debug`, `error`, `info`, `trace`, `warning`

uvicorn 的日志级别。

默认值: `info`

#### `--disable-uvicorn-access-log`, `--no-disable-uvicorn-access-log`

禁用 uvicorn 访问日志。

默认值: `False`

#### `--disable-access-log-for-endpoints`

逗号分隔的端点路径列表，用于从 uvicorn 访问日志中排除。这对于减少来自健康检查等高频率端点的日志噪音很有用。例如："/health,/metrics,/ping"。设置后，对这些路径的请求的访问日志将被抑制，同时保留其他端点的日志。

#### `--allow-credentials`, `--no-allow-credentials`

允许凭据。

默认值: `False`

#### `--allowed-origins`

允许的来源。

默认值: `['*']`

#### `--allowed-methods`

允许的方法。

默认值: `['*']`

#### `--allowed-headers`

允许的请求头。

默认值: `['*']`

#### `--api-key`

如果提供，服务器将要求在请求头中提供这些密钥之一。

#### `--ssl-keyfile`

SSL 密钥文件的文件路径。

#### `--ssl-certfile`

SSL 证书文件的文件路径。

#### `--ssl-ca-certs`

CA 证书文件。

#### `--enable-ssl-refresh`, `--no-enable-ssl-refresh`

当 SSL 证书文件更改时刷新 SSL 上下文。

默认值: `False`

#### `--ssl-cert-reqs`

是否需要客户端证书（请参阅标准库 ssl 模块）。

默认值: `0`

#### `--ssl-ciphers`

HTTPS 的 SSL 密码套件（仅 TLS 1.2 及以下版本）。例如：'ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-CHACHA20-POLY1305'

#### `--root-path`

应用程序在基于路径的路由代理后面时的 FastAPI root_path。

#### `--middleware`

要应用于应用程序的额外 ASGI 中间件。我们接受多个 `--middleware` 参数。该值应该是导入路径。如果提供的是函数，vLLM 将使用 `@app.middleware('http')` 将其添加到服务器。如果提供的是类，vLLM 将使用 `app.add_middleware()` 将其添加到服务器。

默认值: `[]`

#### `--enable-request-id-headers`, `--no-enable-request-id-headers`

如果指定，API 服务器将在响应中添加 X-Request-Id 头。

默认值: `False`

#### `--disable-fastapi-docs`, `--no-disable-fastapi-docs`

禁用 FastAPI 的 OpenAPI 模式、Swagger UI 和 ReDoc 端点。

默认值: `False`

#### `--h11-max-incomplete-event-size`

h11 解析器的不完整 HTTP 事件（头或正文）的最大大小（字节）。有助于缓解头滥用。默认值：4194304（4 MB）。

默认值: `4194304`

#### `--h11-max-header-count`

h11 解析器允许的请求中 HTTP 头的最大数量。有助于缓解头滥用。默认值：256。

默认值: `256`

#### `--enable-offline-docs`, `--no-enable-offline-docs`

为隔离环境启用离线 FastAPI 文档。使用与 vLLM 捆绑的静态资源。

默认值: `False`

#### `--enable-flash-late-interaction`, `--no-enable-flash-late-interaction`

如果设置，在 API 服务器进程中在 GPU 上运行 pooling score MaxSim。可以显著提高后期交互评分性能。

默认值: `True`

### ModelConfig

模型配置。

#### `--model`

要使用的 Hugging Face 模型的名称或路径。当未指定 `served_model_name` 时，它也用作指标输出中 `model_name` 标签的内容。

默认值: `Qwen/Qwen3-0.6B`

#### `--runner`

可能的选择: `auto`, `draft`, `generate`, `pooling`

要使用的模型运行器类型。每个 vLLM 实例仅支持一种模型运行器，即使同一模型可以用于多种类型。

默认值: `auto`

#### `--convert`

可能的选择: `auto`, `classify`, `embed`, `none`

使用 [vllm.model_executor.models.adapters](https://docs.vllm.ai/en/stable/api/vllm/model_executor/models/adapters/#vllm.model_executor.models.adapters) 中定义的适配器转换模型。最常见的用例是将文本生成模型适配为用于 pooling 任务。

默认值: `auto`

#### `--tokenizer`

要使用的 Hugging Face tokenizer 的名称或路径。如果未指定，将使用模型名称或路径。

#### `--tokenizer-mode`

可能的选择: `auto`, `deepseek_v32`, `deepseek_v4`, `hf`, `mistral`, `slow`

Tokenizer 模式：
- "auto" 将为 Mistral 模型使用 `mistral_common` 中的 tokenizer（如果可用），否则使用 "hf" tokenizer。
- "hf" 将使用快速 tokenizer（如果可用）。
- "slow" 将始终使用慢速 tokenizer。
- "mistral" 将始终使用 `mistral_common` 中的 tokenizer。
- "deepseek_v32" 将始终使用 `deepseek_v32` 中的 tokenizer。
- "deepseek_v4" 将始终使用 `deepseek_v4` 中的 tokenizer。
- "qwen_vl" 将始终使用 `qwen_vl` 中的 tokenizer。
- 其他自定义值可以通过插件支持。

要将支持 HF 快速 tokenizer 的 Rust BPE 后端替换为 [fastokens](https://github.com/crusoecloud/fastokens) 实现，请设置 `VLLM_USE_FASTOKENS=1` — 该覆盖适用于任何加载 HF 快速 tokenizer 的模式（`hf`, `deepseek_v32`, `deepseek_v4`, `qwen_vl`, …）。

默认值: `auto`

#### `--trust-remote-code`, `--no-trust-remote-code`

下载模型和 tokenizer 时信任远程代码（例如来自 HuggingFace）。

默认值: `False`

#### `--dtype`

可能的选择: `auto`, `bfloat16`, `float`, `float16`, `float32`, `half`

模型权重和激活的数据类型：
- "auto" 将为 FP32 和 FP16 模型使用 FP16 精度，为 BF16 模型使用 BF16 精度。
- "half" 用于 FP16。推荐用于 AWQ 量化。
- "float16" 与 "half" 相同。
- "bfloat16" 在精度和范围之间取得平衡。
- "float" 是 FP32 精度的简写。
- "float32" 用于 FP32 精度。

默认值: `auto`

#### `--seed`

用于可重现性的随机种子。

必须设置全局种子，否则不同的张量并行工作器将采样不同的 token，导致结果不一致。

默认值: `0`

#### `--hf-config-path`

要使用的 Hugging Face 配置的名称或路径。如果未指定，将使用模型名称或路径。

#### `--allowed-local-media-path`

允许 API 请求从服务器文件系统指定的目录读取本地图像或视频。这存在安全风险。应仅在可信环境中启用。

默认值: `""`

#### `--allowed-media-domains`

如果设置，只有属于此域的媒体 URL 才能用于多模态输入。

#### `--revision`

要使用的特定模型版本。可以是分支名称、标签名称或提交 ID。如果未指定，将使用默认版本。

#### `--code-revision`

要用于 Hugging Face Hub 上模型代码的特定修订版本。可以是分支名称、标签名称或提交 ID。如果未指定，将使用默认版本。

#### `--tokenizer-revision`

要用于 Hugging Face Hub 上 tokenizer 的特定修订版本。可以是分支名称、标签名称或提交 ID。如果未指定，将使用默认版本。

#### `--max-model-len`

模型上下文长度（提示和输出）。如果未指定，将自动从模型配置中派生。

通过 `--max-model-len` 传递时，支持人类可读格式的 k/m/g/K/M/G。示例：
- 1k -> 1000
- 1K -> 1024
- 25.6k -> 25,600
- -1 或 'auto' -> 自动选择适合 GPU 内存的最大模型长度。如果模型的最大上下文长度适合，将使用它，否则将找到可以容纳的最大长度。

解析人类可读的整数，如 '1k'、'2M' 等。包括带有十进制乘数的十进制值。也接受 -1 或 'auto' 作为自动检测的特殊值。

#### `--quantization`, `-q`

用于量化权重的方法。如果为 `None`，我们首先检查模型配置文件中的 `quantization_config` 属性。如果也为 `None`，我们假设模型权重未量化，并使用 `dtype` 确定权重的数据类型。

#### `--quantization-config`

用户面向的量化配置。包含每层类型规格（linear、moe）和忽略模式；请参阅 :class:`QuantizationConfigArgs`。当 `quantization` 是 `ONLINE_QUANT_SHORTHAND_NAMES` 中的值之一时，从匹配的在线简写自动填充。

API 文档：https://docs.vllm.ai/en/latest/api/vllm/config/#vllm.config.QuantizationConfigArgs

应该是有效的 JSON 字符串或单独传递的 JSON 键。

#### `--allow-deprecated-quantization`, `--no-allow-deprecated-quantization`

是否允许已弃用的量化方法。

默认值: `False`

#### `--enforce-eager`, `--no-enforce-eager`

是否始终使用 eager 模式 PyTorch。如果为 True，我们将禁用 CUDA 图并始终以 eager 模式执行模型。如果为 False，我们将使用 CUDA 图和 eager 执行的混合模式以获得最大性能和灵活性。

默认值: `False`

#### `--enable-return-routed-experts`, `--no-enable-return-routed-experts`

是否返回路由专家。

默认值: `False`

#### `--max-logprobs`

当 `SamplingParams` 中指定 `logprobs` 时返回的最大对数概率数。默认值来自 OpenAI Chat Completions API 的默认值。-1 表示无限制，即允许返回所有（output_length * vocab_size）个 logprobs，这可能导致 OOM。

默认值: `20`

#### `--logprobs-mode`

可能的选择: `processed_logits`, `processed_logprobs`, `raw_logits`, `raw_logprobs`

指示 logprobs 和 prompt_logprobs 中返回的内容。支持的模式：
1) raw_logprobs
2) processed_logprobs
3) raw_logits
4) processed_logits

Raw 表示应用任何 logit 处理器（如禁用词）之前的值。Processed 表示应用所有处理器（包括 temperature 和 top_k/top_p）之后的值。

默认值: `raw_logprobs`

#### `--use-fp64-gumbel`, `--no-use-fp64-gumbel`

是否对采样器使用的 Gumbel 噪声使用 FP64（而不是 FP32）。FP64 减少了 Gumbel-max 采样中出现平局的机会，但代价是在大多数 GPU 上显著降低内核吞吐量。

默认值: `False`

#### `--disable-sliding-window`, `--no-disable-sliding-window`

是否禁用滑动窗口。如果为 True，我们将禁用模型的滑动窗口功能，限制为滑动窗口大小。如果模型不支持滑动窗口，则忽略此参数。

默认值: `False`

#### `--disable-cascade-attn`, `--no-disable-cascade-attn`

禁用 V1 的级联注意力。虽然级联注意力不改变数学正确性，但禁用它可能有助于防止潜在的数值问题。默认为 True，因此用户必须通过将其设置为 False 来选择启用级联注意力。即使设置为 False，级联注意力也只会在启发式判断有益时使用。

默认值: `True`

#### `--skip-tokenizer-init`, `--no-skip-tokenizer-init`

跳过 tokenizer 和 detokenizer 的初始化。期望输入中提供有效的 `prompt_token_ids` 和 `None` 作为提示。生成的输出将包含 token ID。

默认值: `False`

#### `--enable-prompt-embeds`, `--no-enable-prompt-embeds`

如果为 `True`，允许通过 `prompt_embeds` 键传递文本嵌入作为输入。

警告：如果传递的嵌入形状不正确，vLLM 引擎可能会崩溃。仅为受信任用户启用此标志！

默认值: `False`

#### `--served-model-name`

服务模型名称。