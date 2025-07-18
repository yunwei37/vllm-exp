#!/usr/bin/env python3
"""
vLLM OpenAI-Compatible API Server Example

This script demonstrates how to start the vLLM server with OpenAI-compatible API.
Run this script to start the server, then use curl or any OpenAI-compatible client to interact with it.
"""

import subprocess
import sys

def main():
    print("Starting vLLM OpenAI-compatible API server...")
    print("Server will run on http://localhost:8000")
    print("\nUsing model: facebook/opt-125m (small model for quick testing)")
    print("\nOnce the server starts, you can test it with:")
    print("\ncurl http://localhost:8000/v1/completions \\")
    print("  -H \"Content-Type: application/json\" \\")
    print("  -d '{")
    print("    \"model\": \"facebook/opt-125m\",")
    print("    \"prompt\": \"Hello, world!\",")
    print("    \"max_tokens\": 50,")
    print("    \"temperature\": 0.7")
    print("  }'")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Start the vLLM server
    try:
        subprocess.run([
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", "facebook/opt-125m",
            "--host", "0.0.0.0",
            "--port", "8000"
        ])
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == "__main__":
    main()