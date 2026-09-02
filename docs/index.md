# eLLM Documentation

eLLM is a CPU-only language-model inference runtime. The current runnable path
loads Qwen3-Coder-30B-A3B-Instruct from a fixed local directory and exposes one
OpenAI-style chat-completions endpoint.

## Start here

1. [Install eLLM and download the model](getting_started/installation.md)
2. [Start the service and send one request](getting_started/quickstart.md)
3. [Review runtime configuration](configuration/env_vars.md)
4. [Deploy the process](deployment/index.md)

For debugging, first run [offline inference](serving/offline_inference.md) to
separate model/runtime issues from the HTTP layer. For application integration,
see the [chat server contract](serving/openai_compatible_server.md).

The current API is deliberately narrower than a full OpenAI or vLLM server.
Only behavior documented in these pages should be treated as supported.
