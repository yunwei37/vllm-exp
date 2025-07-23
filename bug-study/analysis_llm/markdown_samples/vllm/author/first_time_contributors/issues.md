# first_time_contributors - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 3
- Closed Issues: 27

### Label Distribution

- bug: 14 issues
- stale: 11 issues
- installation: 3 issues
- usage: 3 issues
- feature request: 2 issues
- documentation: 2 issues
- help wanted: 2 issues
- good first issue: 2 issues
- tool-calling: 1 issues
- misc: 1 issues

---

## Issue #N/A: [Bug]: Installation Issue with torch Version Conflict on vllm v0.5.0.post1

**Link**: https://github.com/vllm-project/vllm/issues/5576
**State**: closed
**Created**: 2024-06-16T07:08:06+00:00
**Closed**: 2025-02-11T16:37:08+00:00
**Comments**: 6
**Labels**: bug

### Description

### Your current environment

```text
Collecting environment information...
PyTorch version: 2.3.1
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.2 LTS (x86_64)
GCC version: (Ubuntu 11.3.0-1ubuntu1~22.04.1) 11.3.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 19:46:43) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.4.0-165-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA A800-SXM4-80GB
Nvidia driver version: 535.54.03
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.9.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.9.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8.9.0
/usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: Kimi-VL-A3B failed to be deployed using vllm mirroring

**Link**: https://github.com/vllm-project/vllm/issues/16715
**State**: closed
**Created**: 2025-04-16T10:11:58+00:00
**Closed**: 2025-04-17T12:53:32+00:00
**Comments**: 28
**Labels**: installation

### Description

### Your current environment

model：Kimi-VL-A3B-Thinking
image：vllm-openai：latest    
vllm version:0.8.4

1.docker pull vllm/vllm-openai
里面的vllm version:0.8.4
2.docker run --gpus all -v /mnt/data1/LargeLanguageModels/qwen:/model --ipc=host --network=host  --name kimi-vl -it --entrypoint vllm/vllm-openai ：latest  bash
3. 在容器中如下命令启动大模型

> CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \

    --port 3000 \
    --served-model-name kimi-vl \
    --trust-remote-code \
    --model /models/Kimi-VL-A3B-Thinking/Kimi-VL-A3B-Thinking \
    --tensor-parallel-size 1 \
    --max-num-batched-tokens 131072 \
    --max-model-len 131072 \
    --max-num-seqs 512 \
    --limit-mm-per-prompt image=256 \
    --disable-mm-preprocessor-cache

出现报错

> Traceback (most recent call last):
  File "/usr/local/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/local/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, 

[... truncated for brevity ...]

---

## Issue #N/A: ImportError: cannot import name 'activation_ops'

**Link**: https://github.com/vllm-project/vllm/issues/705
**State**: closed
**Created**: 2023-08-08T13:39:56+00:00
**Closed**: 2023-09-11T15:55:40+00:00
**Comments**: 9

### Description

I cloned the repository

---

## Issue #N/A: [Bug]:  ValueError: Expected a torch.device with a specified index or an integer, but got:cuda

**Link**: https://github.com/vllm-project/vllm/issues/14500
**State**: closed
**Created**: 2025-03-08T18:54:49+00:00
**Closed**: 2025-07-07T02:14:18+00:00
**Comments**: 2
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 03-09 02:38:05 [__init__.py:264] Automatically detected platform rocm.
WARNING 03-09 02:38:05 [rocm.py:25] Failed to import from amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
Collecting environment information...
PyTorch version: 2.4.0+rocm6.3.4.git7cecbf6d
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.3.42134-a9a80e791

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (conda-forge gcc 12.1.0-17) 12.1.0
Clang version: Could not collect
CMake version: version 3.31.6
Libc version: glibc-2.35

Python version: 3.10.16 (main, Dec 11 2024, 16:24:50) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: AMD Radeon RX 7900 XT (gfx1100)
Nvidia driver version:

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: The driver_worker gets stuck 100% of the time, when using Medusa with TP > 1

**Link**: https://github.com/vllm-project/vllm/issues/9573
**State**: closed
**Created**: 2024-10-22T02:43:08+00:00
**Closed**: 2025-03-07T02:03:48+00:00
**Comments**: 4
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.4.0+cu121 
OS: Ubuntu 22.04.3 LTS (x86_64)
Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] (64-bit runtime) 

GPU models and configuration:  
GPU 0: NVIDIA A800-SXM4-80GB 
GPU 1: NVIDIA A800-SXM4-80GB 

CPU: 
Architecture:                    x86_64 

Versions of relevant libraries: 
[pip3] numpy==1.26.4 
[pip3] nvidia-cublas-cu12==12.1.3.1 
[pip3] nvidia-cuda-cupti-cu12==12.1.105 
[pip3] nvidia-cuda-nvrtc-cu12==12.1.105 
[pip3] nvidia-cuda-runtime-cu12==12.1.105 
[pip3] nvidia-cudnn-cu12==9.1.0.70 
[pip3] nvidia-cufft-cu12==11.0.2.54 
[pip3] nvidia-curand-cu12==10.3.2.106 
[pip3] nvidia-cusolver-cu12==11.4.5.107 
[pip3] nvidia-cusparse-cu12==12.1.0.106 
[pip3] nvidia-dali-cuda120==1.33.0 
[pip3] nvidia-ml-py==12.560.30 
[pip3] nvidia-nccl-cu12==2.20.5 
[pip3] nvidia-nvjitlink-cu12==12.6.68 
[pip3] nvidia-nvtx-cu1

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: vLLM v0.9.1 Free Memory on device on startup is less than desired GPU memory utilization

**Link**: https://github.com/vllm-project/vllm/issues/20305
**State**: closed
**Created**: 2025-07-01T07:21:49+00:00
**Closed**: 2025-07-03T11:26:17+00:00
**Comments**: 0
**Labels**: usage

### Description

### Your current environment

```text
```

### How would you like to use vllm

I created a container with these arguments by using vLLM v0.9.1 by usnig Dockerode.

```text
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v /home/openzeka/.cache/huggingface:/root/.cache/huggingface:rw \
  -e HF_TOKEN=<TOKEN> \
  vllm/vllm-openai:v0.9.1 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 1 \
  --tensor-parallel-size 2 \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```
But after that I got an error like. My GPUs are free to use.
```text
[multiproc_executor.py:492] ValueError: Free memory on device (31.44/47.38 GiB) on startup is less than desired GPU memory utilization (0.95, 45.01 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.
ERROR 06-30 13:53:19 [core.py:515] EngineCore failed to start.
ERROR 06-30 13:53:19 [core.py:515] Traceback (most recent call last):
ERROR 06-30 13:53:19 [core.py:515]   File "/usr/local/lib/pyth

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: FastAPI Swagger /docs does not working correctly

**Link**: https://github.com/vllm-project/vllm/issues/3795
**State**: closed
**Created**: 2024-04-02T11:56:41+00:00
**Closed**: 2024-04-20T00:16:11+00:00
**Comments**: 2
**Labels**: bug

### Description

### Your current environment

The latest vLLM docker: vllm/vllm-openai:v0.4.0

Issue seems to exist as far back as 0.2.7.


### 🐛 Describe the bug

Visiting **http://localhost:8080/docs** when running the server offline shows a blank page however loads with the proper title for the page. I haven't yet attempted this with an online / non-firewalled running instance of vLLM.

Inspecting the page with developer tools shows some html, however there are errors in downloading .js files from a cdn so I suspect that is the problem. 

Is there a way to enable support for the Swagger documentation offline? 

Since I'm unable to test, http://localhost:8080/docs properly functioning with online instances?


---

## Issue #N/A: [Installation]: building CPU docker image crashes my machine 

**Link**: https://github.com/vllm-project/vllm/issues/8083
**State**: closed
**Created**: 2024-09-02T12:19:16+00:00
**Closed**: 2025-01-02T01:59:28+00:00
**Comments**: 5
**Labels**: installation, stale

### Description

### Your current environment

title says it all, when running 

```bash
docker build -f Dockerfile.cpu -t vllm-cpu-env --shm-size=4g .
```

the build process gets up to [this line](https://github.com/vllm-project/vllm/blob/e2b2aa5a0fdd3e682dd1fbd62e2ba81b8aa054d2/Dockerfile.cpu#L44
) and subsequently freezes my pc indefinitely.
 
 No error messages to share..


PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0
Clang version: Could not collect
CMake version: version 3.30.2
Libc version: glibc-2.35

Python version: 3.10.12 (main, Jul 29 2024, 16:56:48) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-6.8.0-40-generic-x86_64-with-glibc2.35
Is CUDA available: False
CUDA runtime version: No CUDA
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: No CUDA
Nvidia driver version: No CUDA
cuDNN version: 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: AssertionError: MolmoForCausalLM does not support LoRA yet.

**Link**: https://github.com/vllm-project/vllm/issues/11431
**State**: closed
**Created**: 2024-12-23T10:38:18+00:00
**Closed**: 2024-12-31T01:33:08+00:00
**Comments**: 3
**Labels**: feature request

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.10.12 (main, Sep 11 2024, 15:47:36) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-122-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.5.82
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA RTX A6000
GPU 1: NVIDIA RTX A6000

Nvidia driver version: 555.42.02
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Add support for `GPTNeoXForSequenceClassification`

**Link**: https://github.com/vllm-project/vllm/issues/8152
**State**: closed
**Created**: 2024-09-04T10:45:45+00:00
**Closed**: 2025-01-04T01:58:20+00:00
**Comments**: 2
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

Currently vLLM does not support `GPTNeoXForSequenceClassification` architecture. Many reward models that are used in RLHF training have similar architecture (causalLM + linear projection on top). Supporting this architecture can make training and evaluation of RLHF methods way faster.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: Duplicate Token `<s>` in Tokenizer Encoded Token ids

**Link**: https://github.com/vllm-project/vllm/issues/2899
**State**: closed
**Created**: 2024-02-17T06:39:51+00:00
**Closed**: 2024-06-04T00:17:02+00:00
**Comments**: 3
**Labels**: usage

### Description

When working on tokenizer result for `llama-2-7b-chat-hf` model, I noticed that the `prompt_token_ids` generated in [this place](https://github.com/vllm-project/vllm/blob/5f08050d8d0bfcdaced0fe706cdfc9e311e0f263/vllm/engine/llm_engine.py#L385C13-L385C29) would generate an extra token `<s>` in the beginning of the sentence.

For example for the follow prompt `<s>[INST] what is the color of the snow? [/INST]` , hf tokenizer can directly tokenize it to
```
['<s>', '▁[', 'INST', ']', '▁what', '▁is', '▁the', '▁color', '▁of', '▁the', '▁snow', '?', '▁[', '/', 'INST', ']']
[1, 518, 25580, 29962, 825, 338, 278, 2927, 310, 278, 15007, 29973, 518, 29914, 25580, 29962]
```
but for the very same prompt vllm would generate tokenized prompt ids as follows
```
[1, 1, 518, 25580, 29962, 825, 338, 278, 2927, 310, 278, 15007, 29973, 518, 29914, 25580, 29962]
```
which has an extra token `1`, aka `<s>` in the beginning.

Looking forward to have someone help me confirm if this is designated be

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: working with openai-agents sdk an use Runner.run_streamed() got fucntion call error

**Link**: https://github.com/vllm-project/vllm/issues/15256
**State**: open
**Created**: 2025-03-21T00:10:12+00:00
**Comments**: 3
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

2025-03-21 08:04:16 (61.7 KB/s) - ‘collect_env.py’ saved [26257/26257]

INFO 03-21 08:04:21 [__init__.py:256] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.30.2
Libc version: glibc-2.35

Python version: 3.10.12 (main, Feb  4 2025, 14:57:36) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.6.20
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 3090
Nvidia driver version: 560.94
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.9

[... truncated for brevity ...]

---

## Issue #N/A: awq CUDA error: an illegal memory access was encountered

**Link**: https://github.com/vllm-project/vllm/issues/1830
**State**: closed
**Created**: 2023-11-29T08:57:46+00:00
**Closed**: 2023-11-30T09:14:47+00:00
**Comments**: 3

### Description

hi, 

I get an "an illegal memory access was encountered" error when inference [deepseek-coder-33B-base-AWQ](https://huggingface.co/TheBloke/deepseek-coder-33B-base-AWQ),  which is a Llama2 (GQA) architecture model, but the smaller model is fine([deepseek-coder-6.7B-base-AWQ](https://huggingface.co/TheBloke/deepseek-coder-6.7B-base-AWQ)), the relevant information as follows:

## Environment
python==3.8
torch==2.0.1+cu118
transformers==4.34.1
vllm==0.2.2

## Code
````
from vllm import LLM, SamplingParams
import torch

model_path = "deepseek-coder-33b-base-awq"

sampling_params = SamplingParams(temperature=0.0, 
                                      n=1,
                                      use_beam_search=False,
                                      top_p=1, top_k=-1, max_tokens=200, 
                                      skip_special_tokens=False, 
                                      stop_token_ids=stop_token_ids)

llm = LLM(model=model_path, quantization="aw

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Temperature is ignored in vLLM 0.8.0/0.8.1

**Link**: https://github.com/vllm-project/vllm/issues/15241
**State**: closed
**Created**: 2025-03-20T18:30:40+00:00
**Closed**: 2025-03-21T11:01:03+00:00
**Comments**: 11
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
```
</details>

### Description
In vLLM 0.7 and before, using a high temperature (10) with a random input string **always** returns "max_tokens" number of tokens (random output of the correct length)
With a temperature of 0, it returns something similar to "It seems like you've entered a string of characters that doesn't appear to be a meaningful word, phrase, or question."

Using the docker image 0.8.0 or 0.8.1, no matter the temperature, it always answers something like "It seems like you've entered a string of characters that doesn't appear to be a meaningful word, phrase, or question."

### Details
I tried with multiple models and the temperature seems to be ignored for all of them

### 🐛 Describe the bug

### Reproduction
Starting a Docker container with:
`docker run --gpus all \
    --entrypoint bash \
    -v ~/.cache/huggingface:/r

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: YI:34B在使用上无法停止。

**Link**: https://github.com/vllm-project/vllm/issues/3797
**State**: closed
**Created**: 2024-04-02T14:53:17+00:00
**Closed**: 2024-11-28T02:07:07+00:00
**Comments**: 4
**Labels**: bug, stale

### Description

### Your current environment

使用VLLM推理Yi:34b,注意不是chat，总是一直重复直到最大token。试了很多种办法，都没有很好的解决方法。有大佬能协助一下吗
![image](https://github.com/vllm-project/vllm/assets/50692992/b7894d6e-df35-42dd-9056-2ce14f6f4c45)
![image](https://github.com/vllm-project/vllm/assets/50692992/c3784dce-e614-42b9-aeaf-56b6ec72caf4)



### 🐛 Describe the bug

使用VLLM推理Yi:34b,注意不是chat，总是一直重复直到最大token。试了很多种办法，都没有很好的解决方法。有大佬能协助一下吗
![image](https://github.com/vllm-project/vllm/assets/50692992/b7894d6e-df35-42dd-9056-2ce14f6f4c45)
![image](https://github.com/vllm-project/vllm/assets/50692992/c3784dce-e614-42b9-aeaf-56b6ec72caf4)



---

## Issue #N/A: [Bug]: Tensor dimension mismatch when loading Qwen3-Reranker-4B with tensor parallel > 1

**Link**: https://github.com/vllm-project/vllm/issues/20670
**State**: closed
**Created**: 2025-07-09T09:05:10+00:00
**Closed**: 2025-07-12T03:52:44+00:00
**Comments**: 2
**Labels**: bug

### Description


### 🐛 Describe the bug

When trying to load the Qwen3-Reranker-4B model with tensor parallelism enabled (tensor_parallel_size=2), the model initialization fails due to a tensor dimension mismatch error. 

## Environment 
- vLLM version: 0.9.2
- Model: Qwen/Qwen3-Reranker-4B
- GPU configuration: 2 GPUs with tensor parallelism 
- CUDA version: 12.9 

## Steps to reproduce 
1. Run vLLM with the following configuration: 
```bash 
--model Qwen/Qwen3-Reranker-4B --task score --enforce_eager True --served_model_name Qwen/Qwen3-Reranker-4B-30k --hf_overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' --tensor_parallel_size 2 --gpu_memory_utilization 0.97 
``` 

## Expected behavior The model should load successfully with tensor parallelism across 2 GPUs. 
## Actual behavior The model fails to load with the following error: 
``` 
RuntimeError: The size of tensor a (1280) must match the size of tensor b (2560) at

[... truncated for brevity ...]

---

## Issue #N/A: Tool calls not triggered properly with vLLM 0.8.5 and Qwen2.5-Coder-32B-Instruct-GPTQ-Int4

**Link**: https://github.com/vllm-project/vllm/issues/17821
**State**: open
**Created**: 2025-05-08T00:29:00+00:00
**Comments**: 23
**Labels**: bug, tool-calling

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
INFO 05-07 17:26:12 [__init__.py:239] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 4.0.0
Libc version: glibc-2.35

Python version: 3.12.10 (main, Apr  9 2025, 08:55:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-6.8.0-35-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA H100 80GB HBM3
GPU 1: NVIDIA H100 80GB HBM3
GPU 2: NVIDIA H100 80GB HBM3
GPU 3: NVIDIA H100 80GB HBM3

Nvidia driver version: 550.54.15
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen ru

[... truncated for brevity ...]

---

## Issue #N/A: [Doc]: Latency vs Throughput Configurations

**Link**: https://github.com/vllm-project/vllm/issues/6272
**State**: closed
**Created**: 2024-07-09T21:36:46+00:00
**Closed**: 2024-12-02T02:08:36+00:00
**Comments**: 2
**Labels**: documentation, stale

### Description

### 📚 The doc issue

**Context:** During July 9, 2024, vLLM open office hours (FP8), there were several questions regarding how to **optimize** model deployment inference configurations targeting the two major regimes: **latency** and **throughput** (batch processing). Relevant articles around the same discussion, [Efficiently Scaling Transformer Inference](https://arxiv.org/pdf/2211.05102). Whereas there is an exploration of batch size, chip count and context length. Additionally we should explore the whole set of features (e.g optimized kernels, quantization strategies, pipeline/tensor/sequence parallelism)

### Suggest a potential alternative/fix

**Targets:** Create documentation making explicit what configurations are suitable for each regime, and listing some of its constraints and tradeoffs. The creation of this documentation should add new benchmarking and experimental scripts for reproducing such results. Simultaneously this issue will list the set of compatible flags, thus he

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: unsupported operand type(s) for -: 'int' and 'tuple' when compute max_images in qwen2.5vl

**Link**: https://github.com/vllm-project/vllm/issues/16266
**State**: closed
**Created**: 2025-04-08T12:54:52+00:00
**Closed**: 2025-04-08T16:10:44+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Debian GNU/Linux 11 (bullseye) (x86_64)
GCC version: (Debian 10.2.1-6) 10.2.1 20210110
Clang version: Could not collect
CMake version: version 3.18.4
Libc version: glibc-2.31

Python version: 3.11.11 (main, Feb 14 2025, 14:40:43) [GCC 10.2.1 20210110] (64-bit runtime)
Python platform: Linux-5.4.143.bsk.8-amd64-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA L20
GPU 1: NVIDIA L20
GPU 2: NVIDIA L20
GPU 3: NVIDIA L20

Nvidia driver version: 535.161.08
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.9.7
/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.9.7
/usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8.9.7
/usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.8.9.7
/usr/lib/x86_64-linux-gnu/libcudnn_cn

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: `logprobs` is not compatible with the OpenAI spec

**Link**: https://github.com/vllm-project/vllm/issues/4795
**State**: closed
**Created**: 2024-05-13T22:11:16+00:00
**Closed**: 2024-05-29T23:13:24+00:00
**Comments**: 1
**Labels**: bug, help wanted, good first issue

### Description

### Your current environment

I'm using Runpod Serverless vLLM (https://github.com/runpod-workers/worker-vllm) so I can't run this command. However, I confirmed that the issue is in the codebase in `main`:

https://github.com/vllm-project/vllm/blob/0fca3cdcf265cd375bca684d951702b6b7adf65a/vllm/entrypoints/openai/protocol.py


### 🐛 Describe the bug

The behavior of `logprobs=True` does not match OpenAI's.

I identified two issues:

**(1) vLLM throws an error when `logprobs=True` and `top_logprobs` is missing.**

OpenAI works fine:

```py
completion = openai_client.chat.completions.create(
  model="gpt-4-turbo-preview",
  messages=[
    {"role": "user", "content": "Hi!"}
  ],
  logprobs=True,
)
```

```
ChatCompletion(id='chatcmpl-9OY4XFK8suJ7ed0yw5vglbTsOZUt1', choices=[Choice(finish_reason='stop', index=0, logprobs=ChoiceLogprobs(content=[ChatCompletionTokenLogprob(token='Hello', bytes=[72, 101, 108, 108, 111], logprob=-0.0008963357, top_logprobs=[]), ChatComplet

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: TTFT Performance Regression in vLLM v0.7.0 Compared to v0.6.1.post2

**Link**: https://github.com/vllm-project/vllm/issues/14845
**State**: closed
**Created**: 2025-03-14T22:12:49+00:00
**Closed**: 2025-07-13T02:15:01+00:00
**Comments**: 3
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py` on vLLM 0.7.0</summary>

```text
INFO 03-14 21:24:53 __init__.py:183] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: CBL-Mariner/Linux (x86_64)
GCC version: (GCC) 11.2.0
Clang version: Could not collect
CMake version: version 3.21.4
Libc version: glibc-2.35

Python version: 3.10.14 (main, Jul 14 2024, 22:24:12) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-1070-azure-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 11.8.89
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA A100 80GB PCIe
GPU 1: NVIDIA A100 80GB PCIe
GPU 2: NVIDIA A100 80GB PCIe
GPU 3: NVIDIA A100 80GB PCIe

Nvidia driver version: 550.90.07
cuDNN version: Probably one of the following:
/usr/lib/libcudnn.so.8.9.5
/usr/lib/libcudnn

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: installation succeeded but No module named 'vllm._C'

**Link**: https://github.com/vllm-project/vllm/issues/15286
**State**: closed
**Created**: 2025-03-21T10:02:51+00:00
**Closed**: 2025-03-21T11:31:26+00:00
**Comments**: 2
**Labels**: installation

### Description

### Your current environment

My operation system is Windows, and i use anaconda3 to manage my virtual environment and site-packages. I create a new virtual environment with python 3.10.

I ran the command  ```pip install vllm``` to install vllm in my system
The installation process is smooth, and it seems that i successfully installed vllm.
However, when i tried to run ```vllm serve "Qwen/Qwen2.5-VL-7B-Instruct"```,  it occured such bug:
```
INFO 03-21 17:22:05 [__init__.py:256] Automatically detected platform cuda.
Traceback (most recent call last):
  File "C:\Users\Admin\anaconda3\envs\vllm\lib\runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "C:\Users\Admin\anaconda3\envs\vllm\lib\runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "C:\Users\Admin\anaconda3\envs\vllm\Scripts\vllm.exe\__main__.py", line 4, in <module>
  File "C:\Users\Admin\anaconda3\envs\vllm\lib\site-packages\vllm\__init__.py", line 11, in <module>
 

[... truncated for brevity ...]

---

## Issue #N/A: Can't get the same results 

**Link**: https://github.com/vllm-project/vllm/issues/2324
**State**: closed
**Created**: 2024-01-03T03:26:52+00:00
**Closed**: 2024-11-30T02:03:13+00:00
**Comments**: 6
**Labels**: stale

### Description

I try to adapte CogVLM to VLLM, but I got different result when using the same input and the same network weight

**VLLM**

```bash
hiddens_states
tensor([[[ 0.0019, -0.0034,  0.0021,  ..., -0.0099,  0.0027, -0.0037],
         [-0.0073, -0.0082, -0.0618,  ...,  0.0200, -0.0043, -0.0006],
         [-0.0889,  0.0262,  0.0144,  ..., -0.1797,  0.0549,  0.1406],
         ...,
         [-0.0205, -0.0162,  0.0045,  ...,  0.0172,  0.0013,  0.0128],
         [-0.0049, -0.0032, -0.0046,  ...,  0.0201,  0.0166,  0.0149],
         [ 0.0040,  0.0019,  0.0052,  ..., -0.0051,  0.0153,  0.0014]]],
       device='cuda:0', dtype=torch.bfloat16)
input_layernorm
Parameter containing:
tensor([0.0266, 0.0114, 0.0044,  ..., 0.0113, 0.0120, 0.0052], device='cuda:0',
       dtype=torch.bfloat16, requires_grad=True)
hidden_states
tensor([[[-0.0889,  0.0262,  0.0144,  ..., -0.1797,  0.0549,  0.1406],
         [ 0.0332,  0.0322,  0.1157,  ...,  0.1309, -0.0688,  0.0781],
         [-0.1050,  0

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: architecture of models not correctly recognized

**Link**: https://github.com/vllm-project/vllm/issues/16905
**State**: open
**Created**: 2025-04-21T06:33:09+00:00
**Comments**: 1
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 4.0.0
Libc version: glibc-2.35

Python version: 3.12.9 | packaged by conda-forge | (main, Mar  4 2025, 22:48:41) [GCC 13.3.0] (64-bit runtime)
Python platform: Linux-5.15.0-134-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA H100 PCIe
Nvidia driver version: 535.230.02
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address sizes:               

[... truncated for brevity ...]

---

## Issue #N/A: OOM while loading THUDM/chatglm-6b-int4

**Link**: https://github.com/vllm-project/vllm/issues/2338
**State**: closed
**Created**: 2024-01-04T02:13:23+00:00
**Closed**: 2024-03-06T09:58:12+00:00
**Comments**: 4

### Description

I am trying this code to load THUDM/chatglm-6b-int4 on a single GPU:
`llm = LLM(model=model_path, trust_remote_code=True)`

However it raises an OOM exception:

> Traceback (most recent call last):
  File "demo_vllm.py", line 15, in <module>
    llm = LLM(model="chatglm-6b-int4",
  File "miniconda3/envs/vllm/lib/python3.9/site-packages/vllm/entrypoints/llm.py", line 105, in __init__
    self.llm_engine = LLMEngine.from_engine_args(engine_args)
  File "miniconda3/envs/vllm/lib/python3.9/site-packages/vllm/engine/llm_engine.py", line 250, in from_engine_args
    engine = cls(*engine_configs,
  File "miniconda3/envs/vllm/lib/python3.9/site-packages/vllm/engine/llm_engine.py", line 110, in __init__
    self._init_workers(distributed_init_method)
  File "miniconda3/envs/vllm/lib/python3.9/site-packages/vllm/engine/llm_engine.py", line 146, in _init_workers
    self._run_workers(
  File "miniconda3/envs/vllm/lib/python3.9/site-packages/vllm/engine/llm_engine.py", line 755, in

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Why `use_beam_search` is eliminated in `vllm.SamplingParams` from v0.6.3?

**Link**: https://github.com/vllm-project/vllm/issues/10605
**State**: closed
**Created**: 2024-11-24T08:40:38+00:00
**Closed**: 2024-11-25T12:37:50+00:00
**Comments**: 2
**Labels**: usage

### Description

### Your current environment

I am asking a general API question regarding `vllm`, therefore, env info is not needed.

### How would you like to use vllm

I want to ask why `use_beam_search` is eliminated in `vllm.SamplingParams` from v0.6.3 (https://docs.vllm.ai/en/v0.6.3/dev/sampling_params.html)?

How can we control the usage of beam search from v0.6.3 onwards?

To the best of my knowledge, `use_beam_search` is supported in all versions from [v0.4.0.post1](https://docs.vllm.ai/en/v0.4.0.post1/dev/sampling_params.html) to [v0.6.2](https://docs.vllm.ai/en/v0.6.2/dev/sampling_params.html).

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: Connection closed by peer.

**Link**: https://github.com/vllm-project/vllm/issues/7772
**State**: closed
**Created**: 2024-08-22T06:08:52+00:00
**Closed**: 2024-12-22T02:04:23+00:00
**Comments**: 2
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
```
PyTorch version: 2.3.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Alibaba Group Enterprise Linux Server 7.2 (Paladin) (x86_64)
GCC version: (GCC) 9.2.1 20200522 (Alibaba 9.2.1-3 2.17)
Clang version: Could not collect
CMake version: version 3.29.3
Libc version: glibc-2.30

Python version: 3.8.18 (default, Sep 11 2023, 13:40:15)  [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-4.9.151-015.ali3000.alios7.x86_64-x86_64-with-glibc2.17
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA L40S
GPU 1: NVIDIA L40S
GPU 2: NVIDIA L40S
GPU 3: NVIDIA L40S

Nvidia driver version: 535.129.03
cuDNN version: Probably one of the following:
/usr/local/cuda/targets/x86_64-linux/

[... truncated for brevity ...]

---

## Issue #N/A: [Doc]: Invalid JSON examples in Engine Args Document

**Link**: https://github.com/vllm-project/vllm/issues/11965
**State**: closed
**Created**: 2025-01-12T05:16:06+00:00
**Closed**: 2025-01-14T17:03:06+00:00
**Comments**: 1
**Labels**: documentation, help wanted, good first issue

### Description

### 📚 The doc issue

On page https://docs.vllm.ai/en/latest/serving/engine_args.html#engine-args

Regarding the flag `--override-pooler-config`. The documentation provides the following example:

> Override or set the pooling method for pooling models. e.g. {“pooling_type”: “mean”, “normalize”: false}.’

However this example does not work if copy-pasted into a UTF-8 aware text editor as it is not a valid JSON document. (The quotation marks are not ascii quotation marks, they are left-quote and right-quote.) This is an insidious error as it is nearly invisible to the naked eye.

In addition to `--override-pooler-config`, this issue affects `--override-neuron-config`, `--rope-scaling`, and `--mm-processor-kwargs`.

### Suggest a potential alternative/fix

Change

> Override or set the pooling method for pooling models. e.g. {“pooling_type”: “mean”, “normalize”: false}.’

to

> Override or set the pooling method for pooling models. e.g. `{"pooling_type": "mean", "normalize":

[... truncated for brevity ...]

---

## Issue #N/A: [Misc]: How to set num-scheduler-steps

**Link**: https://github.com/vllm-project/vllm/issues/9158
**State**: closed
**Created**: 2024-10-08T13:23:11+00:00
**Closed**: 2025-02-07T01:59:39+00:00
**Comments**: 6
**Labels**: misc, stale

### Description

### Anything you want to discuss about vllm.

Recently **num-scheduler-steps** was introduced to "set the maximum number of forward steps per scheduler call". Is there any documentation on what this exactly means?
Also some guidance would on how to set this value would be much appreciated. For example, if I host a 70B model on 2x A100 with 80GB, does this narrow down the range of values I should consider?

Thanks to all the amazing vllm contributers for making this great peace of software! 🏎

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: Question: why could different ray.Workers produce same random sampling result when processing the same prob tensor?

**Link**: https://github.com/vllm-project/vllm/issues/1333
**State**: closed
**Created**: 2023-10-12T13:11:18+00:00
**Closed**: 2024-03-13T11:30:21+00:00
**Comments**: 1

### Description

I am referring to the `_random_sample` method from `vllm/model_executor/layers/sampler.py`

---

