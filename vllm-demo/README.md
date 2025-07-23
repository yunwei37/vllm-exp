# vLLM Quick Start Guide

This directory contains a vLLM (Very Large Language Model) setup with examples for getting started quickly.

## What is vLLM?

vLLM is a fast and easy-to-use library for LLM inference and serving. It provides:
- High-throughput serving with PagedAttention
- Continuous batching of incoming requests
- Optimized CUDA kernels
- OpenAI-compatible API server

## Prerequisites

- Linux operating system
- Python 3.9 - 3.12
- NVIDIA GPU with compute capability ≥ 7.0 (e.g., V100, T4, RTX20xx, A100, L4, H100)
- CUDA 12.1 (vLLM is compiled with CUDA 12.1)

## Installation

A virtual environment has already been created with vLLM installed. To activate it:

```bash
source venv/bin/activate
```

If you need to reinstall or update vLLM:

```bash
pip install --upgrade vllm
```

## Quick Start Examples

### 1. Offline Batch Inference

Run the basic offline inference example:

```bash
python quickstart_example.py
```

This demonstrates:
- Loading a model (facebook/opt-125m)
- Setting sampling parameters
- Generating text for multiple prompts
- Processing outputs

### 2. OpenAI-Compatible API Server

Start the vLLM server:

```bash
python api_server_example.py
```

The server will run on `http://localhost:8000` with OpenAI-compatible endpoints.

### 3. Client Example

Once the server is running, test it with the client:

```bash
python client_example.py
```

Or use curl:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "Hello, world!",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

## Available Models

The examples use `facebook/opt-125m` - a small model for quick testing. You can use any Hugging Face model that's compatible with vLLM:

```python
# Examples of other models you can use:
llm = LLM(model="meta-llama/Llama-2-7b-hf")
llm = LLM(model="mistralai/Mistral-7B-v0.1")
llm = LLM(model="google/gemma-2b")
```

Note: Some models require accepting their license on Hugging Face and providing an access token.

## Key Features

### Sampling Parameters

Control text generation with various parameters:

```python
sampling_params = SamplingParams(
    temperature=0.8,      # Randomness (0.0 = deterministic, 1.0 = more random)
    top_p=0.95,          # Nucleus sampling threshold
    max_tokens=100,      # Maximum tokens to generate
    top_k=10,            # Top-k sampling
    presence_penalty=0.0, # Penalize repeated tokens
    frequency_penalty=0.0 # Penalize frequent tokens
)
```

### API Endpoints

When running the server, these endpoints are available:

- `GET /v1/models` - List available models
- `POST /v1/completions` - Text completion (OpenAI-compatible)
- `POST /v1/chat/completions` - Chat completion (OpenAI-compatible)
- `POST /v1/embeddings` - Generate embeddings

### Performance Tips

1. **Batch Processing**: vLLM automatically batches requests for optimal throughput
2. **Memory Management**: Uses PagedAttention to manage KV cache efficiently
3. **GPU Utilization**: Maximizes GPU memory usage for better performance

## Troubleshooting

### CUDA Version Mismatch
If you encounter CUDA version issues:
```bash
# Check your CUDA version
nvidia-smi

# Install vLLM for specific CUDA version
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
```

### Out of Memory
For large models, you may need to:
- Use a smaller model
- Reduce `max_model_len` parameter
- Use tensor parallelism across multiple GPUs

### Import Errors
Ensure the virtual environment is activated:
```bash
source venv/bin/activate
which python  # Should show the venv Python
```

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [Supported Models List](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Next Steps

1. Try different models from Hugging Face
2. Experiment with sampling parameters
3. Build a production API with authentication
4. Implement streaming responses
5. Set up monitoring and logging
6. Deploy with Docker or Kubernetes

Happy serving with vLLM!