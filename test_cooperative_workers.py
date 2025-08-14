#!/usr/bin/env python3
"""
测试协作式worker线程的客户端
"""

import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def test_single_request():
    """测试单个请求"""
    print("=== 测试单个请求 ===")

    payload = {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": "This is a test prompt with multiple words that should be split across workers and then combined back together for the final response",
            }
        ],
        "stream": False,
    }

    start_time = time.time()
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    end_time = time.time()

    print(f"请求耗时: {end_time - start_time:.2f}s")
    print(f"状态码: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"响应内容: {result['choices'][0]['message']['content']}")
        print(f"Request ID: {result['id']}")
    else:
        print(f"错误: {response.text}")

    print()


def test_streaming_request():
    """测试流式请求"""
    print("=== 测试流式请求 ===")

    payload = {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": "Generate a long response that can be streamed back chunk by chunk from multiple workers",
            }
        ],
        "stream": True,
    }

    start_time = time.time()
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        stream=True,
    )

    print(f"状态码: {response.status_code}")
    print("流式响应内容:")

    full_content = ""
    for line in response.iter_lines():
        if line:
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                data_str = line_str[6:]  # Remove 'data: ' prefix
                if data_str.strip():
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0]["delta"]
                            if "content" in delta:
                                content = delta["content"]
                                print(content, end="", flush=True)
                                full_content += content

                            if data["choices"][0].get("finish_reason") == "stop":
                                break
                    except json.JSONDecodeError:
                        continue

    end_time = time.time()
    print(f"\n\n流式请求耗时: {end_time - start_time:.2f}s")
    print(f"完整内容: {full_content}")
    print()


def make_concurrent_request(request_id):
    """并发请求函数"""
    payload = {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": f"Request {request_id}: Generate response for concurrent test with enough words to test worker cooperation",
            }
        ],
        "stream": False,
    }

    start_time = time.time()
    try:
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        end_time = time.time()

        if response.status_code == 200:
            result = response.json()
            return {
                "id": request_id,
                "success": True,
                "time": end_time - start_time,
                "content": result["choices"][0]["message"]["content"][:100]
                + "...",  # 截断显示
            }
        else:
            return {
                "id": request_id,
                "success": False,
                "time": end_time - start_time,
                "error": response.text,
            }
    except Exception as e:
        return {
            "id": request_id,
            "success": False,
            "time": time.time() - start_time,
            "error": str(e),
        }


def test_concurrent_requests():
    """测试并发请求"""
    print("=== 测试并发请求 (5个并发) ===")

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=5) as executor:
        # 提交5个并发请求
        futures = [executor.submit(make_concurrent_request, i) for i in range(5)]

        # 收集结果
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result["success"]:
                print(f"请求 {result['id']} 完成，耗时: {result['time']:.2f}s")
            else:
                print(
                    f"请求 {result['id']} 失败，耗时: {result['time']:.2f}s，错误: {result['error']}"
                )

    end_time = time.time()
    total_time = end_time - start_time

    success_count = sum(1 for r in results if r["success"])
    avg_time = sum(r["time"] for r in results if r["success"]) / max(success_count, 1)

    print(f"\n并发测试总耗时: {total_time:.2f}s")
    print(f"成功请求数: {success_count}/5")
    print(f"平均请求耗时: {avg_time:.2f}s")
    print()


def test_server_status():
    """测试服务器状态"""
    print("=== 测试服务器状态 ===")

    try:
        response = requests.get("http://localhost:8000/status")
        print(f"状态码: {response.status_code}")
        print(f"状态信息: {response.text}")
    except Exception as e:
        print(f"状态查询失败: {e}")

    print()


def main():
    print("开始测试协作式worker线程...")
    print("服务器地址: http://localhost:8000")
    print()

    # 等待服务器启动
    print("等待服务器启动...")
    time.sleep(2)

    # 运行各种测试
    test_server_status()
    test_single_request()
    test_streaming_request()
    test_concurrent_requests()

    print("测试完成！")


if __name__ == "__main__":
    main()
