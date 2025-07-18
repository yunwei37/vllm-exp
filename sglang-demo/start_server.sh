#!/bin/bash

echo "Starting SGLang Server with TinyLlama model..."
echo "============================================="
echo ""

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name --format=csv,noheader
    echo ""
fi

echo "Starting server on port 30000..."
echo "This will download the TinyLlama model if not already cached."
echo ""

# Start the server
python -m sglang.launch_server \
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype float16

# Alternative for CPU-only (slower):
# python -m sglang.launch_server \
#     --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
#     --host 0.0.0.0 \
#     --port 30000 \
#     --device cpu