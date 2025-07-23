# body_middle50pct_labels - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 9
- Closed Issues: 21

### Label Distribution

- bug: 8 issues
- stale: 5 issues
- performance: 4 issues
- RFC: 4 issues
- misc: 4 issues
- unstale: 4 issues
- new-model: 3 issues
- documentation: 3 issues
- good first issue: 3 issues
- ray: 3 issues

---

## Issue #N/A: [Performance]: test speculative decode accuracy

**Link**: https://github.com/vllm-project/vllm/issues/9609
**State**: closed
**Created**: 2024-10-23T07:40:46+00:00
**Closed**: 2024-10-25T09:18:03+00:00
**Comments**: 1
**Labels**: performance

### Description

### Proposal to improve performance

I use lm-evaluation-harness to test vllm accuracy
1.when don't enable spec decode,I got some result below
num_concurrent=1
![image](https://github.com/user-attachments/assets/dfa6ef55-216e-4460-9ef4-d387e0ce460e)

num_concurrent=8
![image](https://github.com/user-attachments/assets/505d051f-f119-4275-a5d4-5683b74be398)

num_concurrent=16
![image](https://github.com/user-attachments/assets/87e7c9c6-f2de-43de-8a20-96f82c4c9c7c)

num_concurrent=32
![image](https://github.com/user-attachments/assets/312e2703-cfc8-42c7-9751-22a0b1aba21d)


2.when enable spec decode,I got some result below
num_concurrent=1
![image](https://github.com/user-attachments/assets/6681a17f-3bc7-4d52-b0e5-5451a40dfcf4)

num_concurrent=8
![image](https://github.com/user-attachments/assets/4a1878a8-2da7-475e-9ecd-8400a6fc0620)

num_concurrent=16
![image](https://github.com/user-attachments/assets/fc9ca925-a57c-4c6f-9a07-1fa056e67d66)

num_concurrent=32
![i

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: UVA vs UVM for CPU offloading on v0.8.4+

**Link**: https://github.com/vllm-project/vllm/issues/17062
**State**: open
**Created**: 2025-04-23T15:58:29+00:00
**Comments**: 1
**Labels**: performance

### Description

### Proposal to improve performance

Referencing the recent implementation on https://github.com/vllm-project/vllm/pull/15354 (v0.8.4+) for CPU offloading

@youkaichao, is there any specific reason to pick UVA (`cudaHostAlloc`) over UVM `cudaMallocManaged()`? 

1. UVM goes further than UVA to manage data automatically, often using page-faulting hardware to migrate pages on demand. On systems like the GH200, this has potentially additional benefits such as hardware orchestrated frequency based migration. 
2. A key benefit of Unified Memory is simplifying the heterogeneous computing memory model by eliminating the need for deep copies when accessing structured data in GPU kernels. [Source](https://developer.nvidia.com/blog/unified-memory-in-cuda-6/#unified_memory_or_unified_virtual_addressing)
3. On several discussion threads, the larger access sizes of CPU offloading makes UVM seems to be the better approach compared to UVA [Source](https://forums.developer.nvidia.com/t/page-fault-profi

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]:Why do the prefill and decoding need to be executed twice for the same task?

**Link**: https://github.com/vllm-project/vllm/issues/12266
**State**: closed
**Created**: 2025-01-21T13:19:51+00:00
**Closed**: 2025-01-22T05:44:54+00:00
**Comments**: 3
**Labels**: performance

### Description

### Proposal to improve performance





### Report of performance regression

_No response_

### Misc discussion on performance

Hello, when I start the serving service using vllm serve and conduct tests using the benchmark_serving.py script, I captured the kernel pipeline of the CUDA backend through the nsight system. I found out why the prefill and decoding stages of the same task are executed twice?

![Image](https://github.com/user-attachments/assets/19c57ea0-bb61-49a5-ad3e-e2e4a678b845)

At the same time, my commands are as follows:
* serving:
```
vllm serve data/llama-3-8b-instruct \
        --swap-space 16 \
        --disable-log-requests \
        --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.9 \
        --dtype bfloat16
        --enforce-eager
```
* client:
```
python3 vllm/benchmarks/benchmark_serving.py \
        --backend vllm \
        --model data/llama-3-8b-instruct \
        --profile \
        --dataset-name random \
        --random-input-len 2048 \
 

[... truncated for brevity ...]

---

## Issue #N/A: [New Model]: pfnet/plamo-2-8b

**Link**: https://github.com/vllm-project/vllm/issues/14214
**State**: closed
**Created**: 2025-03-04T14:59:42+00:00
**Closed**: 2025-07-11T02:16:10+00:00
**Comments**: 3
**Labels**: new-model, stale

### Description

### The model to consider.

Please add support for PFN's plamo-2-8b https://huggingface.co/pfnet/plamo-2-8b

### The closest model vllm already supports.

_No response_

### What's your difficulty of supporting the model you want?

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [New Model]: Llama4 Support

**Link**: https://github.com/vllm-project/vllm/issues/16106
**State**: closed
**Created**: 2025-04-05T21:39:19+00:00
**Closed**: 2025-04-06T04:32:40+00:00
**Comments**: 3
**Labels**: new-model

### Description

### 🚀 The feature, motivation and pitch

Meta released 2 Variants:

Llama 4 Scout:
A high-performing small model with 17B activated parameters across 16 experts. Extremely fast, natively multimodal, supports a 10M+ token context window, and runs on a single GPU.

Llama 4 Maverick:
A top-tier multimodal model outperforming GPT-4o and Gemini 2.0 Flash, with performance on par with DeepSeek V3 at half the active parameters. ELO 1417 on LMArena and runs on a single host.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [New Model]: LLaVA-NeXT-Video support

**Link**: https://github.com/vllm-project/vllm/issues/5124
**State**: closed
**Created**: 2024-05-30T03:22:17+00:00
**Closed**: 2024-09-11T05:21:37+00:00
**Comments**: 4
**Labels**: new-model

### Description

### The model to consider.

The llava-next-video project has already been released, and the test results are quite good. Are there any plans to support this project?
`https://github.com/LLaVA-VL/LLaVA-NeXT/blob/inference/docs/LLaVA-NeXT-Video.md`
Currently, Hugging Face does not support this model.

### The closest model vllm already supports.

_No response_

### What's your difficulty of supporting the model you want?

_No response_

---

## Issue #N/A: [RFC]: Interface and Abstraction for Distributed Inference Environment

**Link**: https://github.com/vllm-project/vllm/issues/3587
**State**: closed
**Created**: 2024-03-23T23:41:40+00:00
**Closed**: 2024-06-14T01:00:32+00:00
**Comments**: 18
**Labels**: RFC, misc

### Description

This RFC describes a proposal for interfaces and abstractions for distributed inference environments. I plan to solicit discussions for a week (until March 31st) before I begin to actually refactor the code.

# Motivation

The current distributed inference environment in `vllm` is quite tangled, and we often see deadlocks and hangs (see https://github.com/vllm-project/vllm/issues/3455 , https://github.com/vllm-project/vllm/issues/2770 , https://github.com/vllm-project/vllm/issues/3559 , to name a few). The problem becomes prominent when we try to upgrade to pytorch 2.2.0 (see https://github.com/vllm-project/vllm/pull/3442 , https://github.com/vllm-project/vllm/pull/3442 ), because `pytorch 2.2.0` upgrades from `nccl==2.18.1` to `2.19.3` (see https://pypi.org/pypi/torch/2.1.2/json and https://pypi.org/pypi/torch/2.2.0/json to compare the dependency), and `nccl==2.19.3` breaks `vllm` due to increased memory cost during cudagraph capture (from 10MB per graph to 100MB per graph, adds u

[... truncated for brevity ...]

---

## Issue #N/A: [Misc]: Remove max_tokens field for chat completion requests when not supported anymore by the OpenAI client

**Link**: https://github.com/vllm-project/vllm/issues/9845
**State**: closed
**Created**: 2024-10-30T16:35:40+00:00
**Closed**: 2025-02-28T02:01:57+00:00
**Comments**: 2
**Labels**: misc, stale

### Description

With the introduction of the `o1` model series, OpenAI deprecated the `max_tokens` field in favor of the new `max_completion_tokens` field for the [chat completion API](https://platform.openai.com/docs/api-reference/chat/create).

This change is active since the [v1.45.0](https://github.com/openai/openai-python/compare/v1.44.1...v1.45.0) version of the OpenAI client.

https://github.com/vllm-project/vllm/pull/9837 added the support for the new `max_completion_tokens` in vLLM while deprecating the `max_tokens` field. However, both fields are supported and cohabit during the deprecation period.

When the OpenAI client definitely drops the `max_tokens` field, this change must also be reflected in the vLLM frontend.

This ticket is to keep track of this task. Relevant parts of the code to be updated are commented with `TODO(#9845)`.




### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom r

[... truncated for brevity ...]

---

## Issue #N/A: [Misc]: When using lossy optimization, how to explain that the loss caused by optimization is within the acceptable range?

**Link**: https://github.com/vllm-project/vllm/issues/14128
**State**: closed
**Created**: 2025-03-03T09:30:10+00:00
**Closed**: 2025-07-11T02:16:13+00:00
**Comments**: 8
**Labels**: misc, stale

### Description

### Anything you want to discuss about vllm.

I’ve noticed that with each version upgrade of vllm, there seems to be some degree of precision loss. How do you determine whether these losses are within an acceptable range?

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [RFC]: Encoder/decoder models & feature compatibility

**Link**: https://github.com/vllm-project/vllm/issues/7366
**State**: open
**Created**: 2024-08-09T15:03:54+00:00
**Comments**: 17
**Labels**: RFC, unstale

### Description

## Motivation <a href="#user-content-motivation" id="motivation">#</a>

There is significant interest in vLLM supporting encoder/decoder models. Issues #187  and #180 , for example, request encoder/decoder model support. As a result encoder/decoder support was recently introduced to vLLM via the following three PRs:

* #4837 
* #4888 
* #4942 

These three PRs make encoder/decoder model inference possible; however, they leave more to be desired in terms of (1) parity between vLLM's decoder-only & encoder/decoder request processing pipelines with respect to feature support, and (2) the number of encoder/decoder models which are supported.

The ask for the vLLM community is to contribute PRs which help bring vLLM encoder/decoder functionality to a similar level of maturity as that of vLLM's decoder-only functionality.

## Proposed changes <a href="#user-content-proposed-changes" id="proposed-changes">#</a>

The support matrix below summarizes which encoder/decoder models ha

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: TPU V1 Sampler planning

**Link**: https://github.com/vllm-project/vllm/issues/16268
**State**: open
**Created**: 2025-04-08T14:24:38+00:00
**Comments**: 6
**Labels**: RFC

### Description

### Motivation.

I'd like to gather some input on how to move forward with sampling support, and also provide a brief recap of the current state+planned support.

At a high level, the current design splits model forward and sampling into two separate graphs. 
As of now (`f2ebb6f54`) only the `temperature` and `min_p` have been intentionally enabled. 
As more techniques will be added, the sampling graph will grow in size (vertically, sequential ops) and performance may need monitoring, as we're simply evaluating more operations at runtime. 
To clarify, even when one option is not enabled, we still evaluate a no-op version that undergoes the same ops in the graph (eg top-p with p=1).

### Proposed Change.

Following https://github.com/vllm-project/vllm/pull/15489 a few concerns that have been raised regarding performance while enabling topk,  hence adding the **very first** op to the initial sampling graph, I'd like to re-evaluate the current approach.
Looking at the opposite side of the

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Deprecation of the `best_of` Sampling Parameter in vLLM V1

**Link**: https://github.com/vllm-project/vllm/issues/13361
**State**: closed
**Created**: 2025-02-16T17:57:03+00:00
**Closed**: 2025-03-05T20:22:45+00:00
**Comments**: 2
**Labels**: RFC

### Description

### Motivation.

### Overview
As we transition to vLLM V1, we plan to discontinue support for the `best_of` sampling parameter. This decision is driven by a combination of low usage, alignment with industry trends, and a desire for system simplicity and performance.

### Background: What is `best_of`?
The `best_of` parameter was originally part of the earlier OpenAI completion API. It enabled the generation of multiple completions—`n` different outputs—then selected the “best” completion based on the cumulative log probabilities of each result.

### Reasons for Deprecation

1. **Limited Usage and Industry Trends:**
   - **Low Adoption:** To the best of our knowledge, the `best_of` feature is used by very few users. Users have observed that output quality isn’t reliably correlated with their log probabilities in most cases.
   - **Evolving Standards:** Major AI providers such as OpenAI (in its current API), Claude, and Gemini have moved away from including the `best_of` option.

2. **Al

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: phi 3.5 vision model consuming high CPU RAM and the process getting killed

**Link**: https://github.com/vllm-project/vllm/issues/9190
**State**: open
**Created**: 2024-10-09T12:19:09+00:00
**Comments**: 37
**Labels**: performance, unstale

### Description

### Proposal to improve performance

I am trying to run phi3.5 vision instruct model with around 10k prompts. What I noticed with the increase in prompts my CPU RAM consumption keeps increasing and eventually the process gets killed. Its running fine for say small sample like 1000 prompts. My system configuration is 48 GB VRAM and 64GB CPU RAM. Noticed a similar pattern with PIXTRAL-12B-2409. Has anyone faced this issue?

I have tried the implementation by passing in batches of 1000 to llm.generate but still the CPU RAM keeps increasing
Below is the code implementation:
Ima using two images per prompt
from vllm import LLM, SamplingParams
llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        gpu_memory_utilization=0.7,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 4},
    )
sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
outputs = llm.generate(prompt_list, sampling_params=sampling_params)

[... truncated for brevity ...]

---

## Issue #N/A: HQQ quantization support

**Link**: https://github.com/vllm-project/vllm/issues/2871
**State**: closed
**Created**: 2024-02-14T15:50:49+00:00
**Closed**: 2025-01-14T13:32:35+00:00
**Comments**: 8
**Labels**: unstale

### Description

As we have a few models with Half-Quadratic Quantization (HQQ) out there, VLLM should also support them:

```sh
api_server.py: error: argument --quantization/-q: invalid choice: 'hqq' (choose from 'awq', 'gptq', 'squeezellm', None)
```

E.g.
* https://huggingface.co/mobiuslabsgmbh/Mixtral-8x7B-Instruct-v0.1-hf-attn-4bit-moe-2bit-HQQ

---

## Issue #N/A: [Usage]:  why speculate decoding is slower than normal decoding？

**Link**: https://github.com/vllm-project/vllm/issues/8439
**State**: open
**Created**: 2024-09-13T03:43:26+00:00
**Comments**: 14
**Labels**: usage, unstale

### Description

### Your current environment

The startup command is as follows: it initiates both a standard 7B model and an n-gram speculate model. Speed tests  discover that the speculate model performs more slowly."
```text
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 9000 --model Qwen2-7B-Instruct -tp 1 --gpu_memory_utilization 0.9

CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 9002 --model Qwen2-7B-Instruct -tp 1 --speculative_model [gram] --use-v2-block-manager --num_speculative_tokens 5 --ngram-prompt-lookup-max 4 --gpu_memory_utilization 0.9

result
7b:
first token:  0.04074668884277344s
decode time:  14.328832149505615s
output token:  1000
decode speed:  69.78935823702163 token/s

spec 7b
first token:  0.02350592613220215s
decode time:  15.324904918670654s
output token:  947
decode speed:  61.794836902788866 token/s
```


### How would you like to use vllm

I want to run inference of a

[... truncated for brevity ...]

---

## Issue #N/A: Guidance on how many requests can be processed at a time?

**Link**: https://github.com/vllm-project/vllm/issues/1555
**State**: closed
**Created**: 2023-11-03T19:27:46+00:00
**Closed**: 2024-12-01T02:15:58+00:00
**Comments**: 4
**Labels**: documentation, stale

### Description

Hello - 

I am trying to understand how many requests can be processed in parallel with the llm_engine, and what keeps requests WAITING. I see various variables like "max_num_batched_tokens" and "max_num_seqs", but more details or documentation describing how this process occurs would be helpful. Moreover, how can we tune our system to do process more requests in parallel (e.g. use more GPUs if available, use smaller models, use smaller context windows, etc.)

---

## Issue #N/A: [Doc]: No max_model_len parameter in the LLM class

**Link**: https://github.com/vllm-project/vllm/issues/13021
**State**: closed
**Created**: 2025-02-10T08:23:08+00:00
**Closed**: 2025-02-10T16:16:36+00:00
**Comments**: 2
**Labels**: documentation

### Description

### 📚 The doc issue

In this url: https://docs.vllm.ai/en/latest/serving/offline_inference.html
I see that there is no max_model_len parameter in the LLM class, but the documentation still says 
    llm = LLM(model="adept/fuyu-8b",
    max_model_len=2048,
    max_num_seqs=2)
Btw, I wonder how can I change the max_seq_len when I use offline_inference? 

### Suggest a potential alternative/fix

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: Add documents on how to add new models

**Link**: https://github.com/vllm-project/vllm/issues/65
**State**: closed
**Created**: 2023-05-04T09:05:56+00:00
**Closed**: 2023-06-06T03:01:28+00:00
**Comments**: 0
**Labels**: documentation

### Description

No description provided.

---

## Issue #N/A: [Bug]: vllm serve --config.yaml - Order of arguments matters?

**Link**: https://github.com/vllm-project/vllm/issues/8947
**State**: closed
**Created**: 2024-09-29T15:06:36+00:00
**Closed**: 2024-10-05T17:35:13+00:00
**Comments**: 11
**Labels**: bug, good first issue

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Rocky Linux release 8.10 (Green Obsidian) (x86_64)
GCC version: (GCC) 11.3.0
Clang version: Could not collect
CMake version: version 3.26.5
Libc version: glibc-2.28

Python version: 3.11.5 (main, Sep 11 2023, 13:54:46) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-4.18.0-553.16.1.el8_10.x86_64-x86_64-with-glibc2.28
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA H100
GPU 1: NVIDIA H100
GPU 2: NVIDIA H100
  MIG 1g.12gb     Device  0:
  MIG 1g.12gb     Device  1:
  MIG 1g.12gb     Device  2:
  MIG 1g.12gb     Device  3:
  MIG 1g.12gb     Device  4:
  MIG 1g.12gb     Device  5:
  MIG 1g.12gb     Device  6:
GPU 3: NVIDIA H100

Nvidia d

[... truncated for brevity ...]

---

## Issue #N/A: [Misc]: Implement CPU/GPU swapping in BlockManagerV2

**Link**: https://github.com/vllm-project/vllm/issues/3666
**State**: closed
**Created**: 2024-03-27T21:40:21+00:00
**Closed**: 2024-06-03T20:41:11+00:00
**Comments**: 5
**Labels**: good first issue, misc

### Description

Recently, we refactored the block manager subsystem to improve testability by separating concerns of each layer. See https://github.com/vllm-project/vllm/pull/3492 for more information.

The V2 implementation does not have support for CPU-GPU swapping. It can be added in the [CpuGpuBlockAllocator](https://github.com/vllm-project/vllm/blob/321dc1619ad60b6df74fa86ac6299bc83c223996/vllm/core/block/cpu_gpu_block_allocator.py). My first take on the design is that it should simply keep track of the requested swap requests and have the scheduler `get_and_clear` them after each scheduling step.

![image](https://github.com/vllm-project/vllm/assets/950914/55cf0db2-2614-463b-a053-eb3f182c01bb)


---

## Issue #N/A: [HELP WANTED] Fix Failing Spec Decoding Test

**Link**: https://github.com/vllm-project/vllm/issues/18166
**State**: closed
**Created**: 2025-05-14T20:14:33+00:00
**Closed**: 2025-05-19T02:49:47+00:00
**Comments**: 2
**Labels**: bug, good first issue

### Description

### Issue

We are seeing a test failure related to EAGLE on V0. We would appreciate anyone who can help addressing it. 

```bash
pytest -s -v tests/spec_decode/e2e/test_eagle_correctness.py::test_eagle_e2e_greedy_correctness_with_preemption
```

PR which disables the test: https://github.com/vllm-project/vllm/pull/18165

If anyone has capacity to help out with re-enabling this, we would greatly appreciate it!

---

## Issue #N/A: [Bug]: Vllm 0.8.2 + Ray 2.44 (Ray serve deployment) fallbacks to V0 Engine

**Link**: https://github.com/vllm-project/vllm/issues/15569
**State**: open
**Created**: 2025-03-26T19:29:21+00:00
**Comments**: 13
**Labels**: bug, ray

### Description

### Your current environment

<details>


```text
INFO 03-26 19:23:29 [__init__.py:239] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.11.11 (main, Dec 11 2024, 16:28:39) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-6.8.0-1020-gcp-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.5.40
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA A100-SXM4-40GB
Nvidia driver version: 550.90.07
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Addr

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vLLM still runs after Ray workers crash

**Link**: https://github.com/vllm-project/vllm/issues/16259
**State**: open
**Created**: 2025-04-08T11:23:38+00:00
**Comments**: 12
**Labels**: bug, ray

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
INFO 04-08 04:09:19 [__init__.py:239] Automatically detected platform cuda.
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

Python version: 3.12.9 (main, Feb  5 2025, 08:49:00) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-122-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA H100 NVL
GPU 1: NVIDIA H100 NVL
GPU 2: NVIDIA H100 NVL
GPU 3: NVIDIA H100 NVL

Nvidia driver version: 555.52.04
cuDNN version: Could not collect
HIP runtime version: N/

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Problems with vllm serve DeepSeek-R1 with 2 nodes and TP = 16（include vllm v0.8.4 v0.7.3 v0.7.2 V0 V1 engine）

**Link**: https://github.com/vllm-project/vllm/issues/16692
**State**: open
**Created**: 2025-04-16T02:43:26+00:00
**Comments**: 11
**Labels**: bug, ray

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>
 v0.8.4 using TP = 16 to serving deepseek-v3 in 2*H800*8 On Ray cluster, get EngineCore exception
```text
Your output of `python collect_env.py` here
```

</details>


### 🐛 Describe the bug

start command:
head node:
```bash
ray start --head --port=6379  && \
    vllm serve $MODELPATH \
    --max-num-seqs=256 \
    --max-model-len=32768 \
    --max-num-batched-tokens=32768 \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --distributed-executor-backend=ray \
    --trust-remote-code \
    --served-model-name deepseek-r1
```
slave node:
```bash
ray start --block --address=$HEADPODIP:6379
```

get error:
```bash
2025-04-16 10:27:16,259 INFO usage_lib.py:467 -- Usage stats collection is enabled by default without user confirmation because this terminal is detected to be non-interactive. To disable this, add `--disa

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Error when running Llama-4-Maverick-17B-128E-Instruct-FP8 on mi300x

**Link**: https://github.com/vllm-project/vllm/issues/16474
**State**: closed
**Created**: 2025-04-11T10:27:28+00:00
**Closed**: 2025-04-23T12:07:16+00:00
**Comments**: 8
**Labels**: bug, rocm

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 04-11 09:49:33 [__init__.py:239] Automatically detected platform rocm.
Collecting environment information...
PyTorch version: 2.7.0a0+git295f2ed
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.3.42133-1b9c17779

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.3.1 24491 1e0fda770a2079fbd71e4b70974d74f62fd3af10)
CMake version: version 3.31.6
Libc version: glibc-2.35

Python version: 3.12.10 (main, Apr  9 2025, 08:55:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-128-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: AMD Instinct MI300X (gfx942:sramecc+:xnack-)
Nvidia driver version: Could not collect
cuDNN v

[... truncated for brevity ...]

---

## Issue #N/A: [Bug][Rocm] Garbage Response from vLLM When Using Tensor Parallelism on AMD CPX/NPS4 Partitioned GPUs

**Link**: https://github.com/vllm-project/vllm/issues/20125
**State**: open
**Created**: 2025-06-26T13:18:47+00:00
**Comments**: 0
**Labels**: bug, rocm

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
I have attached output file.
```

[vllm_collect_env_output.txt](https://github.com/user-attachments/files/20926332/vllm_collect_env_output.txt)

</details>


### 🐛 Describe the bug

**Steps to reproduce:**
We referred to doc:  [Steps to Run a vLLM Workload on AMD partition](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/gpu-partitioning/mi300x/run-vllm.html).
- [ ] **Do CPS/NPS4 Partition**
`sudo amd-smi set --memory-partition NPS4`


- [ ] **Launch container**
`docker run -it --network=host --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device /dev/kfd --device /dev/dri rocm/vllm:latest /bin/bash`


 - [ ] **Set Env**
```
export HF_TOKEN=<token>
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```


- [ ] `vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 8`


- [ ] **Query the model**
```
curl http://local

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: ROCM with AWQ

**Link**: https://github.com/vllm-project/vllm/issues/11249
**State**: closed
**Created**: 2024-12-17T03:37:54+00:00
**Closed**: 2024-12-18T02:57:04+00:00
**Comments**: 8
**Labels**: bug, rocm

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.6.0.dev20241113+rocm6.2
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.2.41133-dd7f95766

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.2.0 24292 26466ce804ac523b398608f17388eb6d605a3f09)
CMake version: version 3.26.4
Libc version: glibc-2.31

Python version: 3.9.19 (main, May  6 2024, 19:43:03)  [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-6.8.0-50-generic-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: Radeon RX 7900 XTX (gfx1100)
Nvidia driver version: Could not collect
cuDNN version: Could not collect
HIP runtime version: 6.2.41133
MIOpen runtime version: 3.2.0
Is XNNPACK 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Benchmarks for audio models

**Link**: https://github.com/vllm-project/vllm/issues/16354
**State**: closed
**Created**: 2025-04-09T16:55:19+00:00
**Closed**: 2025-04-19T09:24:15+00:00
**Comments**: 2
**Labels**: help wanted, feature request, multi-modality

### Description

### 🚀 The feature, motivation and pitch

- Add audio datasets to `benchmarks/benchmark_dataset.py` to so we can run performance benchmarks on audio models as well.
- Add a benchmark similar to MMMU (#11196) but for audio models to evaluate their correctness.

### Alternatives

_No response_

### Additional context

cc @mgoin @ywang96 

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature]: Implement Priority Scheduling In V1 Engine

**Link**: https://github.com/vllm-project/vllm/issues/14002
**State**: closed
**Created**: 2025-02-28T01:33:35+00:00
**Closed**: 2025-06-23T03:18:09+00:00
**Comments**: 10
**Labels**: help wanted, feature request

### Description

### 🚀 The feature, motivation and pitch

In V0, we support request priority. I would like to see this in V1

cc @WoosukKwon 

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature]: bind python and c++ through tools other than pybind11

**Link**: https://github.com/vllm-project/vllm/issues/4694
**State**: closed
**Created**: 2024-05-08T23:18:56+00:00
**Closed**: 2024-10-27T22:53:29+00:00
**Comments**: 3
**Labels**: help wanted, feature request, stale

### Description

### 🚀 The feature, motivation and pitch

As vLLM goes into a fast release schedule (currently one release every two weeks), we will quickly hit the project-wide limit of pypi (around 5GB per project). One solution, as pointed out in https://github.com/pypi/support/issues/3792#issuecomment-2099941677 , is to build one wheel for all python versions (Python 3.8+).

I have figured out the procedure https://github.com/pypi/support/issues/3792#issuecomment-2101360740 , but pybind11 does not support this Python Limited API protocol.

One possible solution is to replace pybind11 with some other tools, so that the binding procedure can be used with Python Limited API.

Possible solutions:

- Nanobind (seems to support it starting from Python 3.12 only: https://github.com/wjakob/nanobind/pull/561 )
- register ops through pytorch directly https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html

### Alternatives

_No response_

### Additional context

_No response_

---

