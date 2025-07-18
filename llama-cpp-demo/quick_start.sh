#!/bin/bash

echo "Llama.cpp Quick Start Demo"
echo "========================="
echo ""

# Check if we're on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "This script is designed for Linux. Please adjust for your OS."
    exit 1
fi

echo "Step 1: Downloading pre-built llama.cpp binaries..."
cd /root/yunwei37/vllm-exp/llama-cpp-demo

# Download the latest release
if [ ! -f "build/bin/llama-cli" ]; then
    # Get the actual release URL
    wget -q https://github.com/ggerganov/llama.cpp/releases/download/b5935/llama-b5935-bin-ubuntu-x64.zip
    unzip -q llama-b5935-bin-ubuntu-x64.zip
    rm llama-b5935-bin-ubuntu-x64.zip
    echo "Download complete!"
else
    echo "Binaries already downloaded."
fi

echo ""
echo "Step 2: Testing with a small model from Hugging Face..."
echo "This will download and run TinyLlama model"
echo ""

# Run a simple completion
echo "Running a test prompt..."
echo "Note: Using TinyLlama model which is publicly available..."
./build/bin/llama-cli -hf QuantFactory/TinyLlama-1.1B-Chat-v1.0-GGUF -m TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf -p "Once upon a time" -n 50

echo ""
echo "Demo complete! You can now:"
echo "1. Run interactive chat: ./build/bin/llama-cli -hf QuantFactory/TinyLlama-1.1B-Chat-v1.0-GGUF -m TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf -i -ins"
echo "2. Start API server: ./build/bin/llama-server -hf QuantFactory/TinyLlama-1.1B-Chat-v1.0-GGUF -m TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf --host 0.0.0.0 --port 8080"