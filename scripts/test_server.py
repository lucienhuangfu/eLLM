import os

from openai import OpenAI


def build_endpoint() -> str:
    """
    本地服务默认配置（基于 src/serving/server/mod.rs）：
    - POST /v1/chat/completions
    - GET  /status

    可通过环境变量覆盖：
    - SERVER_BASE_URL (默认: http://127.0.0.1:8000)
    - OPENAI_API_KEY (可选，占位即可)

    例如（Windows PowerShell）：
        $env:SERVER_BASE_URL = "http://127.0.0.1:8000"
        $env:OPENAI_API_KEY = "EMPTY"

    例如（macOS / Linux）：
        export SERVER_BASE_URL="http://127.0.0.1:8000"
        export OPENAI_API_KEY="EMPTY"
    """
    base_url = os.getenv("SERVER_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    if base_url.endswith("/v1"):
        return base_url
    return f"{base_url}/v1"


def build_client() -> OpenAI:
    base_url = build_endpoint()
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    return OpenAI(base_url=base_url, api_key=api_key)


def run_single_chat() -> None:
    client = build_client()
    model = os.getenv("MODEL_NAME", "local-model")
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "你是一个简洁、专业的助手。"},
        {"role": "user", "content": "请用一句话介绍你自己。"},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            temperature=0.7,
        )
        assistant_text = response.choices[0].message.content or ""
        print(f"助手: {assistant_text or '[空响应]'}")
    except Exception as exc:
        print(f"请求失败: {exc}")


if __name__ == "__main__":
    run_single_chat()
