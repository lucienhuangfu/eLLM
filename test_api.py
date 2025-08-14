#!/usr/bin/env python3

import requests
import json


def test_chat_completion():
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "enhanced-llm",
        "stream": False,
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        if response.status_code == 200:
            print("✅ API call successful!")
        else:
            print("❌ API call failed")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")


def test_server_status():
    try:
        response = requests.get("http://localhost:8000/status", timeout=10)
        print(
            f"Status endpoint - Code: {response.status_code}, Response: {response.text}"
        )
    except requests.exceptions.RequestException as e:
        print(f"❌ Status request failed: {e}")


if __name__ == "__main__":
    print("🧪 Testing Enhanced Server API...")
    print("\n📡 Testing server status...")
    test_server_status()

    print("\n💬 Testing chat completion...")
    test_chat_completion()
