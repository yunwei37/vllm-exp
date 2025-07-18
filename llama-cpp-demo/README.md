# Llama.cpp Quick Start Demo

This demo shows how to quickly get started with llama.cpp.

## Installation

### Option 1: Using Pre-built Binaries (Recommended for quick start)

```bash
# Download llama.cpp latest release
wget https://github.com/ggerganov/llama.cpp/releases/latest/download/llama-cli-linux-x64.zip
unzip llama-cli-linux-x64.zip
```

### Option 2: Build from Source

```bash
# Clone the repository
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build
cmake -B build
cmake --build build --config Release

# Binaries will be in build/bin/
```

## Running the Demo

### 1. Download a Model

For this demo, we'll use a small model from Hugging Face:

```bash
# Run a model directly from Hugging Face (requires internet)
./llama-cli -hf ggml-org/gemma-2b-it-GGUF
```

### 2. Interactive Chat

```bash
# Start interactive chat
./llama-cli -hf ggml-org/gemma-2b-it-GGUF -i -ins
```

### 3. API Server Mode

```bash
# Start OpenAI-compatible API server
./llama-server -hf ggml-org/gemma-2b-it-GGUF --host 0.0.0.0 --port 8080
```

Then test with curl:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-2b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Notes

- GGUF is the required model format for llama.cpp
- Models can be downloaded from Hugging Face or converted from other formats
- Check the full documentation at: https://github.com/ggerganov/llama.cpp