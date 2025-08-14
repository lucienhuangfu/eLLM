#!/usr/bin/env python3
"""
增强服务器的测试客户端
"""

import requests
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor


def test_non_streaming():
    """测试非流式响应"""
    url = "http://localhost:8000/v1/chat/completions"

    payload = {
        "model": "enhanced-model",
        "messages": [{"role": "user", "content": "你好，请介绍一下人工智能"}],
        "stream": False,
        "temperature": 0.7,
    }

    headers = {"Content-Type": "application/json"}

    print("=== 测试非流式响应 ===")
    start_time = time.time()

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        end_time = time.time()

        if response.status_code == 200:
            data = response.json()
            print(f"✓ 请求成功，耗时: {end_time - start_time:.2f}s")
            print(f"响应内容: {data['choices'][0]['message']['content']}")
            print(f"请求ID: {data['id']}")
        else:
            print(f"✗ 请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")

    except Exception as e:
        print(f"✗ 请求异常: {e}")


def test_streaming():
    """测试流式响应"""
    url = "http://localhost:8000/v1/chat/completions"

    payload = {
        "model": "enhanced-model",
        "messages": [{"role": "user", "content": "请解释量子计算的基本原理"}],
        "stream": True,
        "temperature": 0.8,
    }

    headers = {"Content-Type": "application/json"}

    print("\n=== 测试流式响应 ===")

    try:
        response = requests.post(
            url, json=payload, headers=headers, stream=True, timeout=15
        )

        if response.status_code == 200:
            print("✓ 流式请求成功，接收数据中...")
            content_chunks = []

            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str.strip():
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0]["delta"]
                                    if "content" in delta and delta["content"]:
                                        content_chunks.append(delta["content"])
                                        print(f"收到块: '{delta['content']}'")

                                    if (
                                        data["choices"][0].get("finish_reason")
                                        == "stop"
                                    ):
                                        print("✓ 流式响应完成")
                                        break
                            except json.JSONDecodeError as e:
                                print(f"JSON解析错误: {e}, 数据: {data_str}")

            full_content = "".join(content_chunks)
            print(f"完整内容: {full_content}")

        else:
            print(f"✗ 流式请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")

    except Exception as e:
        print(f"✗ 流式请求异常: {e}")


def test_concurrent_requests():
    """测试并发请求"""
    print("\n=== 测试并发请求 ===")

    def make_request(request_id):
        url = "http://localhost:8000/v1/chat/completions"
        payload = {
            "model": "enhanced-model",
            "messages": [
                {"role": "user", "content": f"这是并发请求 #{request_id}，请简短回复"}
            ],
            "stream": False,
        }

        try:
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=10)
            end_time = time.time()

            if response.status_code == 200:
                data = response.json()
                print(f"✓ 请求 #{request_id} 成功，耗时: {end_time - start_time:.2f}s")
                return True
            else:
                print(f"✗ 请求 #{request_id} 失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ 请求 #{request_id} 异常: {e}")
            return False

    # 使用线程池进行并发测试
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request, i) for i in range(5)]
        results = [future.result() for future in futures]

    success_count = sum(results)
    print(f"并发测试结果: {success_count}/5 个请求成功")


def test_server_status():
    """测试服务器状态"""
    print("\n=== 测试服务器状态 ===")

    try:
        response = requests.get("http://localhost:8000/status", timeout=5)
        if response.status_code == 200:
            print(f"✓ 状态查询成功: {response.text}")
        else:
            print(f"✗ 状态查询失败: {response.status_code}")
    except Exception as e:
        print(f"✗ 状态查询异常: {e}")


def main():
    print("增强服务器测试客户端")
    print("=" * 50)

    # 检查服务器是否可用
    try:
        response = requests.get("http://localhost:8000/status", timeout=5)
        if response.status_code != 200:
            print("✗ 服务器似乎没有运行，请先启动服务器")
            return
    except Exception:
        print("✗ 无法连接到服务器，请确保服务器在 localhost:8000 上运行")
        return

    print("✓ 服务器连接正常")

    # 运行各种测试
    test_server_status()
    test_non_streaming()
    test_streaming()
    test_concurrent_requests()

    print("\n" + "=" * 50)
    print("所有测试完成")


if __name__ == "__main__":
    main()
