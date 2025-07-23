# moderate_impact_3to10 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 7
- Closed Issues: 23

### Label Distribution

- bug: 12 issues
- stale: 10 issues
- feature request: 4 issues
- new-model: 2 issues
- installation: 2 issues
- torch.compile: 1 issues
- keep-open: 1 issues
- RFC: 1 issues
- documentation: 1 issues

---

## Issue #N/A: [Bug]: nrt_tensor_allocate status=4 message="Allocation Failure" on AWS Neuron

**Link**: https://github.com/vllm-project/vllm/issues/12443
**State**: open
**Created**: 2025-01-26T10:43:56+00:00
**Comments**: 6
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 01-26 10:23:53 __init__.py:183] Automatically detected platform neuron.
Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.10.12 (main, Jan 17 2025, 14:35:34) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-6.8.0-1021-aws-x86_64-with-glibc2.35
Is CUDA available: False
CUDA runtime version: No CUDA
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: No CUDA
Nvidia driver version: No CUDA
cuDNN version: No CUDA
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                     

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: ValueError: Asking to pad but the tokenizer does not have a padding token. Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`

**Link**: https://github.com/vllm-project/vllm/issues/3497
**State**: closed
**Created**: 2024-03-19T11:29:16+00:00
**Closed**: 2024-11-29T02:07:39+00:00
**Comments**: 2
**Labels**: bug, stale

### Description

### Your current environment

```text
python3 benchmarks/benchmark_throughput.py  --backend hf --hf-max-batch-size 20 --model /data/pretrain_models/Baichuan2-7B-Chat --trust-remote-code --input-len 512 --output-len 2048 --num-prompts 20
```


### 🐛 Describe the bug

Warning: import flash_attn rotary fail, please install FlashAttention rotary to get higher efficiency https://github.com/Dao-AILab/flash-attention/tree/main/csrc/rotary
Warning: import flash_attn rms_norm fail, please install FlashAttention layer_norm to get higher efficiency https://github.com/Dao-AILab/flash-attention/tree/main/csrc/layer_norm
Warning: import flash_attn fail, please install FlashAttention to get higher efficiency https://github.com/Dao-AILab/flash-attention
Loading checkpoint shards:   0%|                                                                                                                                                                        | 0/8 [00:00<?, ?it/s]/data/luhairong/anaconda

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: MLPSpeculator Tensor Parallel support

**Link**: https://github.com/vllm-project/vllm/issues/5809
**State**: closed
**Created**: 2024-06-25T01:10:59+00:00
**Closed**: 2024-07-02T14:20:30+00:00
**Comments**: 3
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

`MLPSpeculator`-based speculative decoding was recently added in https://github.com/vllm-project/vllm/pull/4947, but the initial integration only covers single GPU usage.

There will soon be "speculator" models available for larger target models that require multiple GPUs so we would like to ensure that TP can be used.

The first part of this issue would be testing it out in conjunction with https://github.com/vllm-project/vllm/pull/5414 and making necessary adjustments so that it will work with TP=1 for the speculator and TP=N for the target model.

Following this we can look at having the speculator itself run with TP>1, but that may be more involved since it will require some distributed coordination of the sampling of each speculated token in the MLPSpeculator loop. It might be possible to avoid additional communication here by the having the sampler used by the speculator model use a dedicated `torch.Generator` for its sampling and 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: 模型运行期间，报错TimeoutError: RPC call to execute_model timed out.，导致模型退出。

**Link**: https://github.com/vllm-project/vllm/issues/19197
**State**: open
**Created**: 2025-06-05T09:19:47+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
Collecting environment information...
==============================
        System Info
==============================
OS                           : Ubuntu 24.04.2 LTS (x86_64)
GCC version                  : (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
Clang version                : Could not collect
CMake version                : Could not collect
Libc version                 : glibc-2.39

==============================
       PyTorch Info
==============================
PyTorch version              : 2.7.0+cu126
Is debug build               : False
CUDA used to build PyTorch   : 12.6
ROCM used to build PyTorch   : N/A

==============================
      Python Environment
==============================
Python version               : 3.12.9 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:27) [GCC 11.2.0] (64-bit runtime)
Python platform              : Linux-6.6.87.

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: sm75 can not serve qwen3 bnb 4bit model

**Link**: https://github.com/vllm-project/vllm/issues/17337
**State**: open
**Created**: 2025-04-29T01:03:34+00:00
**Comments**: 9
**Labels**: bug

### Description

### Your current environment

<details>
<summary>docker image v0.8.5</summary>


vllm-openai-1  | (VllmWorkerProcess pid=149) WARNING 04-28 18:00:58 [utils.py:168] The model class Qwen3MoeForCausalLM has not defined `packed_modules_mapping`, this may lead to incorrect mapping of quantized or ignored modules
vllm-openai-1  | WARNING 04-28 18:00:58 [utils.py:168] The model class Qwen3MoeForCausalLM has not defined `packed_modules_mapping`, this may lead to incorrect mapping of quantized or ignored modules
vllm-openai-1  | (VllmWorkerProcess pid=149) ERROR 04-28 18:00:58 [multiproc_worker_utils.py:238] Exception in worker VllmWorkerProcess while processing method load_model.
vllm-openai-1  | (VllmWorkerProcess pid=149) ERROR 04-28 18:00:58 [multiproc_worker_utils.py:238] Traceback (most recent call last):
vllm-openai-1  | (VllmWorkerProcess pid=149) ERROR 04-28 18:00:58 [multiproc_worker_utils.py:238]   File "/usr/local/lib/python3.12/dist-packages/vllm/executor/multiproc_worker_utils.py"

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]:Slim Attention (lossless 2x reduction in KV cache size)

**Link**: https://github.com/vllm-project/vllm/issues/14937
**State**: open
**Created**: 2025-03-17T08:20:19+00:00
**Comments**: 2
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

**Feature Description**
Slim attention can reduce KV cache size by a factor of 2 without a loss of accuracy as it's lossless: https://arxiv.org/pdf/2503.05840

**Motivation**
Allows you to run with larger context sizes at the same (V)RAM usage or allows you to cram the same context into less (V)RAM. Furthermore, it improves performance at long context sizes.

**Possible Implementation**
No response


---

## Issue #N/A: vLLM on OpenShift/Kubernetes Manifests

**Link**: https://github.com/vllm-project/vllm/issues/2314
**State**: closed
**Created**: 2024-01-01T11:50:39+00:00
**Closed**: 2024-09-20T20:09:30+00:00
**Comments**: 2

### Description

Does anyone have a sample manifest on how to deploy vLLM on OpenShift or Kubernetes?

---

## Issue #N/A: [Feature]: Support attention backend with FlexAttention

**Link**: https://github.com/vllm-project/vllm/issues/7315
**State**: closed
**Created**: 2024-08-08T19:52:26+00:00
**Closed**: 2025-02-14T01:59:26+00:00
**Comments**: 10
**Labels**: feature request, torch.compile, stale

### Description

### 🚀 The feature, motivation and pitch

FlexAttention was proposed as a performant attention implementation leveraging `torch.compile` with easy APIs for adding support for complex attention variants such as Causal, [Relative Positional Embeddings](https://paperswithcode.com/method/relative-position-encodings), [Alibi](https://paperswithcode.com/method/alibi), [Sliding Window Attention](https://mistral.ai/news/announcing-mistral-7b/), [PrefixLM](https://twitter.com/andersonbcdefg/status/1800907703688339569), [Document Masking/Sample Packing/Jagged Tensors](https://github.com/pytorch/torchtune/pull/875), [Tanh Soft-Capping](https://twitter.com/LysandreJik/status/1807779471891538199), [PagedAttention](https://arxiv.org/abs/2309.06180), etc.

https://pytorch.org/blog/flexattention/

While it is not the fastest attention backend (yet!) it is clearly performant enough while enabling much more flexibility than current compiled backends to easily implement attention features we need fo

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Build and publish Neuron docker image

**Link**: https://github.com/vllm-project/vllm/issues/4838
**State**: open
**Created**: 2024-05-15T15:27:17+00:00
**Comments**: 4
**Labels**: feature request, keep-open

### Description

### 🚀 The feature, motivation and pitch

It seems like the current docker images don't support Neuron (Inferentia).
It would be very helpful if there was a tested, managed Neuron docker image to use.
While at the same subject, it would be even better if some documentation would be added on running vLlm Neuron using containers.

### Alternatives

DJL?

### Additional context

_No response_

---

## Issue #N/A: Refactor Mixtral to reuse code from MegaBlocks

**Link**: https://github.com/vllm-project/vllm/issues/2032
**State**: closed
**Created**: 2023-12-11T19:04:24+00:00
**Closed**: 2024-03-25T11:32:24+00:00
**Comments**: 5

### Description

Hello! A [portion](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/mixtral.py#L223-L326) of the MoE implementation for Mixtral is copied directly from MegaBlocks. It's somewhat error prone code and I've been meaning to factor out helpers for it, which we could reuse to avoid having this duplicated in vLLM. If this is interesting to you I'll send a PR :)

---

## Issue #N/A: How to install from source with CUDA 11.8 instead of 12.1?

**Link**: https://github.com/vllm-project/vllm/issues/2072
**State**: closed
**Created**: 2023-12-13T01:49:45+00:00
**Closed**: 2024-03-28T12:02:21+00:00
**Comments**: 4

### Description

No description provided.

---

## Issue #N/A: [Bug]: lora base_model.model.lm_head.base_layer.weight is not supported 

**Link**: https://github.com/vllm-project/vllm/issues/4186
**State**: open
**Created**: 2024-04-19T02:36:50+00:00
**Comments**: 11
**Labels**: bug, stale

### Description

### Your current environment

```text
Collecting environment information...
PyTorch version: 2.2.2+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Debian GNU/Linux 12 (bookworm) (x86_64)
GCC version: (Debian 12.2.0-14) 12.2.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.36

Python version: 3.11.9 (main, Apr 10 2024, 14:54:51) [GCC 12.2.0] (64-bit runtime)
Python platform: Linux-6.5.0-15-generic-x86_64-with-glibc2.36
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA H100 80GB HBM3
GPU 1: NVIDIA H100 80GB HBM3
GPU 2: NVIDIA H100 80GB HBM3
GPU 3: NVIDIA H100 80GB HBM3

Nvidia driver version: 550.54.14
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                       x86_64
CPU op-mode(s):

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: "Loading safetensors checkpoint shards" runs twice when serving model

**Link**: https://github.com/vllm-project/vllm/issues/13765
**State**: closed
**Created**: 2025-02-24T14:03:29+00:00
**Closed**: 2025-06-24T06:19:51+00:00
**Comments**: 3
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Amazon Linux 2023.6.20250115 (x86_64)
GCC version: (GCC) 11.4.1 20230605 (Red Hat 11.4.1-2)
Clang version: Could not collect
CMake version: version 3.22.2
Libc version: glibc-2.34

Python version: 3.12.8 | packaged by conda-forge | (main, Dec  5 2024, 14:24:40) [GCC 13.3.0] (64-bit runtime)
Python platform: Linux-6.1.119-129.201.amzn2023.x86_64-x86_64-with-glibc2.34
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA L4
Nvidia driver version: 560.35.03
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                   

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: ARM vLLM container build failing due to outlines v0.1.9

**Link**: https://github.com/vllm-project/vllm/issues/11178
**State**: closed
**Created**: 2024-12-13T19:33:28+00:00
**Closed**: 2024-12-14T07:46:20+00:00
**Comments**: 4
**Labels**: bug

### Description

### Your current environment

M2 Mac with Docker Desktop

### Model Input Dumps

_No response_

### 🐛 Describe the bug

Building ARM vLLM Docker image with `docker build -f Dockerfile.arm -t vllm-cpu-env --shm-size=4g .` leads to the following error:
```
24.95   copying python/outlines_core/fsm/regex.py -> build/lib.linux-aarch64-cpython-310/outlines_core/fsm
24.95   running build_ext
24.95   running build_rust
24.95   error: can't find Rust compiler
24.95 
24.95   If you are using an outdated pip version, it is possible a prebuilt wheel is available for this package but pip is not able to install from it. Installing from the wheel would avoid the need for a Rust compiler.
24.95 
24.95   To update pip, run:
24.95 
24.95       pip install --upgrade pip
24.95 
24.95   and then retry package installation.
24.95 
24.95   If you did intend to build this package from source, try installing a Rust compiler from your system package manager and ensure it is on the PATH during ins

[... truncated for brevity ...]

---

## Issue #N/A: [New Model]: Mistral-Nemo

**Link**: https://github.com/vllm-project/vllm/issues/6563
**State**: closed
**Created**: 2024-07-19T06:51:46+00:00
**Closed**: 2024-07-19T13:06:56+00:00
**Comments**: 2
**Labels**: new-model

### Description

### The model to consider.

https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407

### The closest model vllm already supports.

- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py
- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/mixtral.py

### What's your difficulty of supporting the model you want?

_No response_

---

## Issue #N/A: GPU KV cache usage: 100.0%以后就卡住

**Link**: https://github.com/vllm-project/vllm/issues/1206
**State**: closed
**Created**: 2023-09-28T01:13:13+00:00
**Closed**: 2024-12-01T02:16:12+00:00
**Comments**: 21
**Labels**: bug, stale

### Description

GPU KV cache usage: 100.0%以后就卡住，GPU使用率也将为0，无法继续提供服务，请问有什么解决办法吗？

---

## Issue #N/A: [RFC]: should we use `VLLM_WORKER_MULTIPROC_METHOD=spawn` by default for openai-compatible api server ?

**Link**: https://github.com/vllm-project/vllm/issues/15681
**State**: closed
**Created**: 2025-03-28T06:52:19+00:00
**Closed**: 2025-03-28T09:53:59+00:00
**Comments**: 2
**Labels**: RFC

### Description

### Motivation.

I’ve recently encountered this error:

```
RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
```

And I always have to specify `VLLM_WORKER_MULTIPROC_METHOD=spawn` to bypass the issue, so I investigated the cause.

vllm forces `VLLM_WORKER_MULTIPROC_METHOD=spawn` when `torch.cuda.is_initialized()` is True, but that’s not enough.

In PyTorch, when the main process runs `poison_fork` and then forks a subprocess, the subprocess's `in_bad_fork` gets set to true. Therefore, when the subprocess attempts to reinitialize CUDA, it throws an error.

https://github.com/pytorch/pytorch/blob/v2.6.0/torch/csrc/cuda/Module.cpp#L63-L79

However, `poison_fork` doesn't only run during `torch._C._cuda_init`. It also runs during `torch._C._cuda_getDeviceCount` and `torch._C._cuda_getArchFlags`.

So if we use the fork method, we can't call methods like `torch.cuda.is_available()` or `torch.cuda.device_count

[... truncated for brevity ...]

---

## Issue #N/A: [New Model]: Supporting DBRX from Databricks

**Link**: https://github.com/vllm-project/vllm/issues/3658
**State**: closed
**Created**: 2024-03-27T13:00:35+00:00
**Closed**: 2024-03-27T20:01:47+00:00
**Comments**: 1
**Labels**: new-model

### Description

### The model to consider.

Databricks has released [DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm), which consists of 2 models

- [dbrx](https://huggingface.co/databricks/dbrx-base)
- [dbrx-instruct](https://huggingface.co/databricks/dbrx-instruct)

It's a 132B parameter MoE model. Might be useful.

### The closest model vllm already supports.

_No response_

### What's your difficulty of supporting the model you want?

It seems that they have a custom script in their files, might need custom implementation on that regard.

---

## Issue #N/A: [Bug]: No output / Repeated outputs when using Gemma 3  on vLLM

**Link**: https://github.com/vllm-project/vllm/issues/20341
**State**: open
**Created**: 2025-07-01T22:35:38+00:00
**Comments**: 16
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
Your output of `python collect_env.py` here
```

</details>


### 🐛 Describe the bug

I'm running the google/gemma-3-27b-it model with vLLM using the OpenAI-compatible API server.


CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 python /opt/VLLM/vllm/vllm/entrypoints/openai/api_server.py \
--model /opt/MODELS/gemma-3-27b-it/ \
--max-model-len 32000 \
--host 10.12.112.168 \
--port 9005 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.9


Then, I send a standard request to the /v1/chat/completions endpoint using Python:


import requests
import json

url = "http://10.12.112.168:9005/v1/chat/completions"

data = {
    "model": "/opt/MODELS/gemma-3-27b-it/",
    "messages": [
        {"role": "user", "content": "hello"}
    ],
    "temperature": 0.1,
    "max_tokens": 500,
    "enable_thinking": False
}

headers = {
    "Content-Type": "application/json"
}

response = requests.

[... truncated for brevity ...]

---

## Issue #N/A: why online seving slower than offline serving??

**Link**: https://github.com/vllm-project/vllm/issues/2019
**State**: closed
**Created**: 2023-12-11T12:50:58+00:00
**Closed**: 2024-10-26T16:44:55+00:00
**Comments**: 13

### Description

1. offline serving
![image](https://github.com/vllm-project/vllm/assets/43260218/87e216b5-9064-4c2a-a021-cac08e22795d)

2. online serving(fastapi)
![image](https://github.com/vllm-project/vllm/assets/43260218/322cc4a4-a78f-4212-a266-d586e8e2969d)
![image](https://github.com/vllm-project/vllm/assets/43260218/49c9cf76-ca3f-4362-95d8-191cbbdd3543)
log: INFO 12-11 21:50:36 llm_engine.py:649] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.4%, CPU KV cache usage: 0.0%
INFO 12-11 21:50:41 async_llm_engine.py:111] Finished request 261ddff3312f44cd8ee1c52a6acd10e6.


Why is the speed 2 seconds slower when displayed as fastapi??
parameters is same, prompt is same

"Open-Orca/Mistral-7B-OpenOrca" this model same issue
and any llama2 model same issue

python : 3.10.12
[my library list.txt](https://github.com/vllm-project/vllm/files/13641002/my.library.list.txt)

cuda_version :

[... truncated for brevity ...]

---

## Issue #N/A: [Doc]: Outdated docs on AutoAWQ

**Link**: https://github.com/vllm-project/vllm/issues/6653
**State**: closed
**Created**: 2024-07-22T17:47:58+00:00
**Closed**: 2024-11-24T02:07:33+00:00
**Comments**: 2
**Labels**: documentation, stale

### Description

### 📚 The doc issue

The AutoAWQ content is outdated. Particularly, the warning is now obsolete due to the optimization in #6612 which makes AWQ run much faster in vLLM.

### Suggest a potential alternative/fix

The warning should be updated to reflect newer versions of vLLM.

---

## Issue #N/A: [Bug]: Tokenization Mismatch Between HuggingFace and vLLM

**Link**: https://github.com/vllm-project/vllm/issues/8904
**State**: closed
**Created**: 2024-09-27T12:43:11+00:00
**Closed**: 2025-01-26T01:59:45+00:00
**Comments**: 4
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.10.14 (main, May  6 2024, 19:42:50) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-1061-nvidia-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.1.105
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA H100 80GB HBM3
Nvidia driver version: 550.90.07
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.9.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.9.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8.9.0
/usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.8.9

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vllm:num_requests_waiting is not being published at /metrics endpoint

**Link**: https://github.com/vllm-project/vllm/issues/7918
**State**: closed
**Created**: 2024-08-27T16:10:57+00:00
**Closed**: 2024-12-28T01:59:07+00:00
**Comments**: 4
**Labels**: bug, stale

### Description

### 🐛 Describe the bug

Data for vllm:num_requests_waiting is missing.

vllm:num_requests_waiting is not being published at /metrics endpoint

docker image for vllm : vllm-openai:v0.5.3.post1

```
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="/data/models/model-gemma2-a100/experiment-it1"} 0.0

```

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: Does VLLM currently support QWEN LoRa model ？

**Link**: https://github.com/vllm-project/vllm/issues/3201
**State**: closed
**Created**: 2024-03-05T12:22:41+00:00
**Closed**: 2024-11-30T02:02:14+00:00
**Comments**: 7
**Labels**: stale

### Description

I  use the multi-LoRA for offline inference:
sql_lora_path = "/home/zyn/models/slot_lora_gd"

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="/home/models/dem_14b/base",
          enable_lora=True,
          trust_remote_code=True)

sampling_params = SamplingParams(temperature=0,
                                 max_tokens=256,
                                 stop=["[/assistant]"])

prompts = [
    "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",
    "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",
]

outputs = llm.generate(prompts,

[... truncated for brevity ...]

---

## Issue #N/A: Support for  Smaug-72B-v0.1 on vLLM

**Link**: https://github.com/vllm-project/vllm/issues/2917
**State**: closed
**Created**: 2024-02-19T03:25:19+00:00
**Closed**: 2024-04-08T19:42:37+00:00
**Comments**: 12

### Description

No description provided.

---

## Issue #N/A: Repeated answer: When I use vllm with opt-13b, the generated text is not end until the max length, with the repeated answer

**Link**: https://github.com/vllm-project/vllm/issues/1958
**State**: closed
**Created**: 2023-12-07T08:02:58+00:00
**Closed**: 2024-03-25T10:10:52+00:00
**Comments**: 11

### Description

hi, I use vllm in greedy SamplingType, I meet  the repeated answer: 
`sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048)`

`from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Give three tips for staying healthy.",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048)

# Create an LLM.
llm = LLM(model="/workspace/opt-13b/", tensor_parallel_size=4)
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}", len(output.out
puts[0].token_ids))`

The answer is follow, the same text is repreated, also to other prompts.
![截屏2023-12-07 下午3 59 51](https://github.com/vllm-project/vllm/assets/8213143/dd316200-1c6b-420e-abbf-2fd07dbd8283)

So is there any way to solve this problem?
Thanks.

---

## Issue #N/A: [bug]vllm deployement is failing on nvidia because of numpy2.0 upgrade

**Link**: https://github.com/vllm-project/vllm/issues/5594
**State**: closed
**Created**: 2024-06-17T07:19:41+00:00
**Closed**: 2024-06-17T10:10:28+00:00
**Comments**: 2
**Labels**: installation

### Description

### Your current environment

# python3.9 -c "import vllm; print(vllm.__version__)"

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.0.0 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "<string>", line 1, in <module>
  File "/usr/local/lib64/python3.9/site-packages/vllm/__init__.py", line 3, in <module>
    from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
  File "/usr/local/lib64/python3.9/site-packages/vllm/engine/arg_utils.py", line 6, in <module>
    from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, ModelConfig,
  File "/usr/local/lib64/python3.9/site-packages/vllm/conf

[... truncated for brevity ...]

---

## Issue #N/A: os.environ['CUDA_VISIBLE_DEVICES'] = '1' does not work in jupyter

**Link**: https://github.com/vllm-project/vllm/issues/571
**State**: closed
**Created**: 2023-07-25T09:46:53+00:00
**Closed**: 2023-08-08T09:04:51+00:00
**Comments**: 1

### Description

As the title says, it is invalid to specify the GPU through `CUDA_VISIBLE_DEVICES` in jupyter, and only 'GPU:0' will still be used; but it is effective when using `CUDA_VISIBLE_DEVICES=1 python *.py`

---

## Issue #N/A: cache_kernel.cu does not compile using pip install -e . on source code

**Link**: https://github.com/vllm-project/vllm/issues/651
**State**: closed
**Created**: 2023-08-02T19:54:02+00:00
**Closed**: 2024-04-20T12:10:56+00:00
**Comments**: 3
**Labels**: installation

### Description

Neither in docker (with the suggested docker), nor on my own environment, I get to compilte the cache_kernel.cu.

NCVV = 11.8, also using the PyTorch 2.0.1 CUDA 11.8 package.

At first, it didn't install at all because of the myproject.toml pointing towards a pytorch in pip that is not cuda enabled.
After removing the toml file, I ran into these errors:

vllm\csrc\cache_kernels.cu(41): error: expression must be a pointer to a complete object type

vllm\csrc\cache_kernels.cu(42): error: expression must be a pointer to a complete object type

vllm\csrc\cache_kernels.cu(96): error: expression must have a constant value
vllm\csrc\cache_kernels.cu(96): note #2689-D: the value of variable "num_layers"
    (86): here cannot be used as a constant

vllm\csrc\cache_kernels.cu(97): error: expression must have a constant value
vllm\csrc\cache_kernels.cu(97): note #2689-D: the value of variable "num_layers"
    (86): here cannot be used as a constant


---

## Issue #N/A: [Bug]: OpenAI server unexpected shutdown

**Link**: https://github.com/vllm-project/vllm/issues/6629
**State**: closed
**Created**: 2024-07-22T02:31:50+00:00
**Closed**: 2024-11-24T02:07:40+00:00
**Comments**: 3
**Labels**: bug, stale

### Description

### Your current environment

```text
The output of `python collect_env.py`
```
Collecting environment information...
PyTorch version: 2.3.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: Could not collect
CMake version: version 3.29.5
Libc version: glibc-2.31

Python version: 3.9.19 (main, May  6 2024, 19:43:03)  [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-107-generic-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY

Nvidia driver version: 535.183.01
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

Versions of relevant libraries:
[pip3] numpy==1.26.4
[pip3] nvidia-nccl-cu12==2.20.5
[pip3] torch==2.3.0
[pip3] torcheval==0.0.7
[pip3] transformers==4.41.2
[pip3] tr

[... truncated for brevity ...]

---

