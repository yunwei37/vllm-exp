# 3rd_party - issues

**Total Issues**: 4
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 4

### Label Distribution

- bug-unconfirmed: 4 issues
- 3rd party: 4 issues
- stale: 3 issues
- medium severity: 2 issues
- low severity: 1 issues

---

## Issue #N/A: Bug: High CPU usage and bad output with flash attention on ROCm

**Link**: https://github.com/ggml-org/llama.cpp/issues/8893
**State**: closed
**Created**: 2024-08-06T18:02:29+00:00
**Closed**: 2024-10-06T01:07:37+00:00
**Comments**: 8
**Labels**: bug-unconfirmed, stale, medium severity, 3rd party

### Description

### What happened?

While flash attention works well for my python ui (https://github.com/curvedinf/dir-assistant/) on an nvidia system, it produces bad results on my AMD system. My AMD system has a 7900XT with ROCm 6.1.2 on Ubuntu 22.04. When FA is off, inference is almost instant with high t/s and no CPU usage. When FA is on, the CPU utilization is 100% for 1-2 minutes, and then tokens are generated slowly and are incorrect (in my case, it always produces "################################" for a long time).

### Name and Version

Using python bindings via llama-cpp-python 0.2.84. Readme says llama.cpp version is [ggerganov/llama.cpp@4730faca618ff9cee0780580145e3cbe86f24876](https://github.com/ggerganov/llama.cpp/commit/4730faca618ff9cee0780580145e3cbe86f24876)

### What operating system are you seeing the problem on?

Linux

### Relevant log output

_No response_

---

## Issue #N/A: Bug: Model generating result sometimes but most of the times it doesnt 

**Link**: https://github.com/ggml-org/llama.cpp/issues/8269
**State**: closed
**Created**: 2024-07-03T06:24:30+00:00
**Closed**: 2024-07-03T08:27:34+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, medium severity, 3rd party

### Description

### What happened?

So, I am currently working with the local 'mistral-7b-q4' gguf model using 'llamacpp'. Although I can confirm that the model is active on the server but still I have encountered some issues during testing. Specifically, when I provide a small prompt , the model occasionally generates a response, but more often than not , it produces an empty string for the same prompt.

At this stage, I am unsure whether this behaviour is a result of the latest update, an expected characteristic of the model, or if there might be an error in my approach. 

This is how I am calling the model:

`

# Load LLM
llm = LlamaCpp(
    model_path="Models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_gpu_layers=-1,
    temperature=0.1,
    top_p=0.7,
    n_ctx=16000,
    max_tokens=4096,
    frequency_penalty=0.2,
    presence_penalty=0.5,
    stop=["\n"]
)`

these are the outputs I am getting. First one is without a response and second one is with the reponse. 
![Microsof

[... truncated for brevity ...]

---

## Issue #N/A: Bug: CUDA error: out of memory - Phi-3 Mini 128k prompted with 20k+ tokens on 4GB GPU

**Link**: https://github.com/ggml-org/llama.cpp/issues/7885
**State**: closed
**Created**: 2024-06-11T19:50:26+00:00
**Closed**: 2024-08-01T01:07:07+00:00
**Comments**: 28
**Labels**: bug-unconfirmed, stale, 3rd party

### Description

### What happened?

I get a CUDA out of memory error when sending large prompt (about 20k+ tokens) to Phi-3 Mini 128k model on laptop with Nvidia A2000 4GB RAM. At first about 3.3GB GPU RAM and 8GB CPU RAM is used by ollama, then the GPU ram usage slowly rises (3.4, 3.5GB etc.) and after about a minute it throws the error when GPU ram is exhaused probably (3.9GB is latest in task manager). The inference does not return any token (as answer) before crashing. Attaching server log. Using on Win11 + Ollama 0.1.42 + VS Code (1.90.0) + Continue plugin (v0.8.40).

The expected behavior would be not crashing and maybe rellocating the memory somehow so that GPU memory does not get exhausted. I want to disable GPU usage in ollama (to test for CPU inference only - I have 64GB RAM) but I am not able to find out how to turn the GPU off (even though I saw there is a command for it recently - am not able to find it again).

Actual error:
```
CUDA error: out of memory
  current device: 0, in fu

[... truncated for brevity ...]

---

## Issue #N/A: Bug:  Error while running model file (.gguf ) in LM Studio

**Link**: https://github.com/ggml-org/llama.cpp/issues/7779
**State**: closed
**Created**: 2024-06-05T20:15:20+00:00
**Closed**: 2024-07-21T01:07:09+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale, low severity, 3rd party

### Description

### What happened?

I'm encountering an error while trying to run a model in LM Studio. Below are the details of the error:

{
  "title": "Failed to load model",
  "cause": "llama.cpp error: 'error loading model architecture: unknown model architecture: 'clip''",
  "errorData": {
    "n_ctx": 2048,
    "n_batch": 512,
    "n_gpu_layers": 10
  },
  "data": {
    "memory": {
      "ram_capacity": "15.92 GB",
      "ram_unused": "7.46 GB"
    },
    "gpu": {
      "gpu_names": [
        "NVIDIA GeForce 940MX"
      ],
      "vram_recommended_capacity": "2.00 GB",
      "vram_unused": "1.64 GB"
    },
    "os": {
      "platform": "win32",
      "version": "10.0.22631",
      "supports_avx2": true
    },
    "app": {
      "version": "0.2.24",
      "downloadsDir": "C:\\Users\\hp\\.cache\\lm-studio\\models"
    },
    "model": {}
  }
}


![image](https://github.com/ggerganov/llama.cpp/assets/82998682/f408d74f-dee0-4347-983f-1c7d6c8e87e3)


### Name and 

[... truncated for brevity ...]

---

