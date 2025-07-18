# SGLang Quick Start Demo

SGLang is a fast serving framework for large language models and vision language models.

## Installation

### Option 1: Using pip (Recommended)

```bash
# Upgrade pip and install uv
pip install --upgrade pip
pip install uv

# Install SGLang with all dependencies
uv pip install "sglang[all]>=0.4.9.post2"
```

### Option 2: From Source

```bash
git clone -b v0.4.9.post2 https://github.com/sgl-project/sglang.git
cd sglang
pip install -e "python[all]"
```

## Running the Demo

### 1. Start the SGLang Server

```bash
# Start server with a small model
python -m sglang.launch_server --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host 0.0.0.0 --port 30000
```

### 2. Test with Python Client

```python
import sglang as sgl

@sgl.function
def hello_sglang(s):
    s += "Hello! " + sgl.gen("response", max_tokens=50)

# Use the function
state = hello_sglang.run()
print(state["response"])
```

### 3. Test with OpenAI-compatible API

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 50
  }'
```

## Notes

- SGLang requires a GPU for optimal performance
- Supports various model formats and frameworks
- Provides OpenAI-compatible API endpoints
- Check the documentation at https://docs.sglang.ai/ for more details