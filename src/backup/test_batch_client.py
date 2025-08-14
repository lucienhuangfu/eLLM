#!/usr/bin/env python3
"""
批处理服务器测试客户端
演示如何向批处理服务器发送多个请求来触发批处理
"""

import asyncio
import aiohttp
import json
import time
from typing import List
import sys


class BatchTestClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def send_chat_completion(self, prompt: str, stream: bool = False) -> dict:
        """发送单个聊天完成请求"""
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            "temperature": 0.7,
            "max_tokens": 100,
        }

        request_start = time.time()

        if stream:
            return await self._handle_stream_request(url, payload, request_start)
        else:
            return await self._handle_non_stream_request(url, payload, request_start)

    async def _handle_non_stream_request(
        self, url: str, payload: dict, request_start: float
    ) -> dict:
        """处理非流式请求"""
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                duration = time.time() - request_start
                return {
                    "success": True,
                    "duration": duration,
                    "response": result,
                    "content": result["choices"][0]["message"]["content"],
                }
            else:
                return {
                    "success": False,
                    "status": response.status,
                    "error": await response.text(),
                }

    async def _handle_stream_request(
        self, url: str, payload: dict, request_start: float
    ) -> dict:
        """处理流式请求"""
        content_chunks = []
        first_chunk_time = None

        async with self.session.post(
            url, json=payload, headers={"Accept": "text/event-stream"}
        ) as response:
            if response.status != 200:
                return {
                    "success": False,
                    "status": response.status,
                    "error": await response.text(),
                }

            async for line in response.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        if first_chunk_time is None:
                            first_chunk_time = time.time()

                        if chunk["choices"][0]["delta"].get("content"):
                            content_chunks.append(
                                chunk["choices"][0]["delta"]["content"]
                            )

                    except json.JSONDecodeError:
                        continue

            total_duration = time.time() - request_start
            first_chunk_latency = (
                first_chunk_time - request_start if first_chunk_time else None
            )

            return {
                "success": True,
                "duration": total_duration,
                "first_chunk_latency": first_chunk_latency,
                "content": "".join(content_chunks),
                "chunks_count": len(content_chunks),
            }

    async def test_single_request(self, prompt: str, stream: bool = False):
        """测试单个请求"""
        print(f"\n=== 单个请求测试 ({'流式' if stream else '非流式'}) ===")
        print(f"发送prompt: {prompt}")

        result = await self.send_chat_completion(prompt, stream)

        if result["success"]:
            print(f"✅ 请求成功")
            print(f"⏱️ 总耗时: {result['duration']:.3f}s")
            if stream and result.get("first_chunk_latency"):
                print(f"🚀 首chunk延迟: {result['first_chunk_latency']:.3f}s")
                print(f"📦 总chunk数: {result.get('chunks_count', 0)}")
            print(f"💬 回复: {result['content'][:100]}...")
        else:
            print(f"❌ 请求失败: {result}")

    async def test_concurrent_requests(self, prompts: List[str], stream: bool = False):
        """测试并发请求以触发批处理"""
        print(f"\n=== 并发请求测试 ({'流式' if stream else '非流式'}) ===")
        print(f"并发发送 {len(prompts)} 个请求...")

        start_time = time.time()

        # 并发发送所有请求
        tasks = [
            self.send_chat_completion(f"请求{i+1}: {prompt}", stream)
            for i, prompt in enumerate(prompts)
        ]

        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        print(f"🎯 所有请求完成，总耗时: {total_time:.3f}s")

        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        print(f"✅ 成功: {len(successful)}, ❌ 失败: {len(failed)}")

        if successful:
            avg_duration = sum(r["duration"] for r in successful) / len(successful)
            min_duration = min(r["duration"] for r in successful)
            max_duration = max(r["duration"] for r in successful)

            print(f"📊 单个请求耗时统计:")
            print(f"   平均: {avg_duration:.3f}s")
            print(f"   最小: {min_duration:.3f}s")
            print(f"   最大: {max_duration:.3f}s")

            if stream:
                first_chunk_latencies = [
                    r["first_chunk_latency"]
                    for r in successful
                    if r.get("first_chunk_latency")
                ]
                if first_chunk_latencies:
                    avg_first_chunk = sum(first_chunk_latencies) / len(
                        first_chunk_latencies
                    )
                    print(f"🚀 平均首chunk延迟: {avg_first_chunk:.3f}s")

        # 显示部分响应内容
        print(f"\n📝 响应示例:")
        for i, result in enumerate(successful[:3]):
            print(f"  请求{i+1}: {result['content'][:80]}...")

    async def test_sequential_vs_concurrent(self, prompts: List[str]):
        """比较顺序处理和并发处理的性能差异"""
        print(f"\n=== 性能对比测试 ===")

        # 顺序处理
        print(f"🔄 顺序处理 {len(prompts)} 个请求...")
        sequential_start = time.time()
        sequential_results = []
        for i, prompt in enumerate(prompts):
            result = await self.send_chat_completion(f"顺序{i+1}: {prompt}")
            sequential_results.append(result)
        sequential_time = time.time() - sequential_start

        # 并发处理
        print(f"⚡ 并发处理 {len(prompts)} 个请求...")
        concurrent_start = time.time()
        concurrent_tasks = [
            self.send_chat_completion(f"并发{i+1}: {prompt}")
            for i, prompt in enumerate(prompts)
        ]
        concurrent_results = await asyncio.gather(*concurrent_tasks)
        concurrent_time = time.time() - concurrent_start

        # 结果对比
        print(f"\n📈 性能对比结果:")
        print(f"顺序处理: {sequential_time:.3f}s")
        print(f"并发处理: {concurrent_time:.3f}s")
        print(f"性能提升: {(sequential_time/concurrent_time):.2f}x")

        # 吞吐量计算
        sequential_throughput = len(prompts) / sequential_time
        concurrent_throughput = len(prompts) / concurrent_time
        print(f"顺序吞吐量: {sequential_throughput:.2f} req/s")
        print(f"并发吞吐量: {concurrent_throughput:.2f} req/s")

    async def get_server_status(self):
        """获取服务器状态"""
        try:
            async with self.session.get(f"{self.base_url}/status") as response:
                if response.status == 200:
                    status = await response.json()
                    print(f"🟢 服务器状态: {status}")
                    return status
                else:
                    print(f"❌ 无法获取服务器状态: {response.status}")
                    return None
        except Exception as e:
            print(f"❌ 连接服务器失败: {e}")
            return None


async def main():
    print("🚀 批处理服务器测试客户端启动")
    print("=" * 50)

    async with BatchTestClient() as client:
        # 检查服务器状态
        status = await client.get_server_status()
        if not status:
            print("请确保服务器已启动 (cargo run --bin batch_server)")
            return

        # 测试数据
        test_prompts = [
            "什么是人工智能？",
            "请解释机器学习的基本概念",
            "深度学习和传统机器学习有什么区别？",
            "如何优化神经网络的性能？",
            "什么是transformer架构？",
            "请介绍一下注意力机制",
        ]

        # 1. 单个请求测试
        await client.test_single_request("测试单个非流式请求", stream=False)
        await client.test_single_request("测试单个流式请求", stream=True)

        # 2. 小批量并发测试（触发批处理）
        small_batch = test_prompts[:4]  # 刚好达到batch_size
        await client.test_concurrent_requests(small_batch, stream=False)
        await client.test_concurrent_requests(small_batch, stream=True)

        # 3. 大批量并发测试
        await client.test_concurrent_requests(test_prompts, stream=False)

        # 4. 性能对比测试
        await client.test_sequential_vs_concurrent(test_prompts[:4])

        print(f"\n🎉 所有测试完成！")
        print("💡 观察服务器日志可以看到批处理的工作过程")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 测试中断")
    except Exception as e:
        print(f"❌ 测试出错: {e}")
