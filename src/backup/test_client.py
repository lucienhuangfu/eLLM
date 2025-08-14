#!/usr/bin/env python3
"""
Simple test client for the OpenAI-compatible API server
"""

import requests
import json
import time


def test_chat_completion(stream=False):
    """Test the chat completion endpoint"""
    url = "http://localhost:8000/v1/chat/completions"

    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "stream": stream,
        "temperature": 0.7,
        "max_tokens": 100,
    }

    headers = {"Content-Type": "application/json"}

    try:
        if stream:
            print("Testing streaming response...")
            response = requests.post(url, json=payload, headers=headers, stream=True)

            if response.status_code == 200:
                print("✓ Streaming request successful")
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode("utf-8")
                        if line_str.startswith("data: "):
                            data_str = line_str[6:]  # Remove 'data: ' prefix
                            try:
                                data = json.loads(data_str)
                                print(f"Received chunk: {json.dumps(data, indent=2)}")
                            except json.JSONDecodeError:
                                print(f"Raw line: {line_str}")
            else:
                print(f"✗ Streaming request failed with status {response.status_code}")
                print(f"Response: {response.text}")
        else:
            print("Testing non-streaming response...")
            response = requests.post(url, json=payload, headers=headers)

            if response.status_code == 200:
                print("✓ Non-streaming request successful")
                data = response.json()
                print(f"Response: {json.dumps(data, indent=2)}")
            else:
                print(
                    f"✗ Non-streaming request failed with status {response.status_code}"
                )
                print(f"Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print("✗ Connection error - make sure the server is running on port 8000")
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    print("Testing OpenAI-compatible API server")
    print("=" * 50)

    # Test non-streaming
    test_chat_completion(stream=False)

    print("\n" + "=" * 50)

    # Test streaming
    test_chat_completion(stream=True)
