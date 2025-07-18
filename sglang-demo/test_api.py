#!/usr/bin/env python3
"""
Test SGLang with OpenAI-compatible API
"""

import requests
import json
import time

def test_chat_completion():
    """Test the chat completion endpoint"""
    url = "http://localhost:30000/v1/chat/completions"
    
    payload = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        print("Chat Completion Response:")
        print(json.dumps(result, indent=2))
        
        # Extract the assistant's response
        if 'choices' in result and len(result['choices']) > 0:
            assistant_response = result['choices'][0]['message']['content']
            print(f"\nAssistant: {assistant_response}")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to SGLang server.")
        print("Make sure the server is running with: ./start_server.sh")
    except Exception as e:
        print(f"Error: {e}")

def test_completion():
    """Test the completion endpoint"""
    url = "http://localhost:30000/v1/completions"
    
    payload = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt": "Once upon a time",
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        print("\nCompletion Response:")
        print(json.dumps(result, indent=2))
        
    except requests.exceptions.ConnectionError:
        print("\nSkipping completion test - server not running")
    except Exception as e:
        print(f"Error in completion test: {e}")

if __name__ == "__main__":
    print("Testing SGLang API endpoints...")
    print("================================")
    
    # Test chat completion
    test_chat_completion()
    
    # Test regular completion
    test_completion()
    
    print("\nTest complete!")