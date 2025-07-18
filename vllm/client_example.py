#!/usr/bin/env python3
"""
vLLM Client Example

This script demonstrates how to interact with the vLLM OpenAI-compatible API server.
Make sure the server is running first by executing api_server_example.py
"""

import requests
import json

def test_completion_api():
    """Test the completions API endpoint"""
    url = "http://localhost:8000/v1/completions"
    
    payload = {
        "model": "facebook/opt-125m",
        "prompt": "The quick brown fox",
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    print("Testing Completions API...")
    print(f"Prompt: {payload['prompt']}")
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        generated_text = result['choices'][0]['text']
        print(f"Generated: {generated_text}")
        print(f"\nFull response: {json.dumps(result, indent=2)}")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server.")
        print("Make sure the vLLM server is running (python api_server_example.py)")
    except Exception as e:
        print(f"Error: {e}")

def test_chat_api():
    """Test the chat completions API endpoint"""
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "model": "facebook/opt-125m",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    print("\n\nTesting Chat Completions API...")
    print(f"Message: {payload['messages'][0]['content']}")
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        generated_text = result['choices'][0]['message']['content']
        print(f"Response: {generated_text}")
        print(f"\nFull response: {json.dumps(result, indent=2)}")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server.")
        print("Make sure the vLLM server is running (python api_server_example.py)")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("vLLM Client Example")
    print("=" * 50)
    
    test_completion_api()
    test_chat_api()