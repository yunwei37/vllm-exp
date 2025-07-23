# recent_burst_30days - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 23
- Closed Issues: 7

### Label Distribution

- amd: 2 issues
- lora: 1 issues
- high priority: 1 issues

---

## Issue #N/A: [Bug] [ROCm] Segmentation fault when capture batches in cuda graph

**Link**: https://github.com/sgl-project/sglang/issues/7847
**State**: open
**Created**: 2025-07-08T07:17:10+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Encounter segmentation fault issue when capturing cuda graph in latest docker image(lmsysorg/sglang:v0.4.9-rocm630)

```
Loading safetensors checkpoint shards:  97% Completed | 158/163 [00:35<00:01,  4.41it/s]
Loading safetensors checkpoint shards:  99% Completed | 162/163 [00:35<00:00,  7.01it/s]
Loading safetensors checkpoint shards: 100

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] support dimensions param for embedding models

**Link**: https://github.com/sgl-project/sglang/issues/7474
**State**: open
**Created**: 2025-06-23T12:09:32+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

As documented in [openai website](https://platform.openai.com/docs/api-reference/embeddings/create#embeddings-create-dimensions), users can pass a param called dimensions to specify the dimension of output vector. 




### Related resources

_No response_

---

## Issue #N/A: [Bug] [CI regression] [AMD] TestNoOverlapScheduler

**Link**: https://github.com/sgl-project/sglang/issues/7703
**State**: open
**Created**: 2025-07-02T01:44:36+00:00
**Comments**: 0
**Labels**: amd

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

The CI **unit-test-backend-1-gpu-amd** failed when run`test/srt/test_no_overlap_scheduler.py`. It exits with a GPU memory access fault on node-2. 

**Error snippet**:

```text
...
batch. #new-seq: 1, #new-token: 32, #cached-token: 0, token usage: 0.00, #running-req: 8, #queue-req: 119
[2025-07-02 03:22:31] Prefill batch. #new-seq: 1, #new-

[... truncated for brevity ...]

---

## Issue #N/A: Performance regression: 4090 GPUs slower on v0.4.8-cu126 (was 27→20 tokens/sec, A100 unaffected)

**Link**: https://github.com/sgl-project/sglang/issues/7568
**State**: open
**Created**: 2025-06-26T13:32:58+00:00
**Comments**: 2

### Description

Hi! After upgrading from `lmsysorg/sglang:v0.4.6.post5-cu124` to `lmsysorg/sglang:v0.4.8-cu126`, I've noticed a **significant drop in generation speed** and GPU utilization on my setup with 2x4090 (48Gb each).  
- **v0.4.6:** 27 tokens/sec, GPU usage 100%
- **v0.4.8:** 20 tokens/sec, GPU usage ~75%

No other changes were made except switching the Docker image.

On an A100 80Gb, both versions work fine — no speed or GPU usage drop.

### Environment

- **GPUs:** 2x4090 48Gb
- **Docker Compose Config:** (see below)
- **Model:** Qwen3-32B
- **nvidia-smi:**
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 565.57.01              Driver Version: 565.57.01      CUDA Version: 12.7     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] If rid is repeated, there will be no output

**Link**: https://github.com/sgl-project/sglang/issues/8089
**State**: closed
**Created**: 2025-07-16T08:37:45+00:00
**Closed**: 2025-07-17T02:50:18+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Only Receive, no Finish
[2025-07-16 14:43:09] Receive: obj=EmbeddingReqInput(rid=['1846246021_91f15ad6-b9e3-43f4-a1a4-ef9bbb30f8d4', '1846246021_d55a53c3-3e6b-41e2-9e0f-af822fd7dfe3', '1846246021_97fab5d0-edd9-4742-9fc2-b745737fc031', '1846246021_643f81cc-1414-497f-af0e-bd1d0a0adcdd', '1846246021_04f97c20-9143-4798-8131-a786e89d0601', '184

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] TokenWeave optimizations

**Link**: https://github.com/sgl-project/sglang/issues/7652
**State**: closed
**Created**: 2025-06-30T08:04:47+00:00
**Closed**: 2025-06-30T08:16:16+00:00
**Comments**: 0

### Description

Hi,
Would it be possible to integrate these optimizations into sglang?

Code: https://github.com/microsoft/tokenweave/tree/main
Paper: https://arxiv.org/abs/2505.11329

---

## Issue #N/A: [Bug] [CI regression] TestEpMoEFP8

**Link**: https://github.com/sgl-project/sglang/issues/7586
**State**: open
**Created**: 2025-06-27T05:49:07+00:00
**Comments**: 6

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

CI unit-test-backend-2-gpu seems to be broken since the past few days (if not longer).

Looking at the log, it seems to be watchdog timeout at TestEpMoEFP8. The non-quantized version TestEpMoE seems to be working fine.

### Reproduction

Sample failure: https://github.com/sgl-project/sglang/actions/runs/15916204833/job/44895747274

### Env

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Cutlass kernels for LoRA

**Link**: https://github.com/sgl-project/sglang/issues/7910
**State**: open
**Created**: 2025-07-09T21:43:29+00:00
**Comments**: 0
**Labels**: lora

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Creating an issue to track the work for supporting a CUTLASS / CUTE kernel for LoRA to see if there is any perf gain comparing with the current Triton one.

Dependency: this task should happen after #7809 as the FlashInfer deprecation is expected to change / simplify the kernel interface.

(cc @Fridge003 @Ying1123 )

### Related resources

_No response_

---

## Issue #N/A: [Bug] Qwen3 FP8 models crash at startup without `SGL_ENABLE_JIT_DEEPGEMM=0`

**Link**: https://github.com/sgl-project/sglang/issues/7495
**State**: open
**Created**: 2025-06-24T09:10:01+00:00
**Comments**: 4

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

This problem is mentioned in #7482, but the issue is about a different problem. Running sglang on commit fa42e419629e0651a8caf332330942da920cdac8 built from source fails on FP8 versions of Qwen3 models.

```
$ python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B-FP8 --tp 1 --reasoning-parser qwen3
INFO 06-24 08:53:58 [__init__.py:24

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] [CI regression] [AMD] TestVisionChunkedPrefill

**Link**: https://github.com/sgl-project/sglang/issues/7701
**State**: open
**Created**: 2025-07-01T23:25:20+00:00
**Comments**: 2
**Labels**: amd

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

CI unit-test-backend-1-gpu-amd seems to be broken since the past few days.

```
	output with chunked prefill:
	The video features a person standing on a stage with a dark background. The individual is dressed in a black outfit and appears to be speaking or presenting. The stage
	output without chunked prefill:
	The video features a person 

[... truncated for brevity ...]

---

## Issue #N/A: Task 005: Router Interface and Factory

**Link**: https://github.com/sgl-project/sglang/issues/7538
**State**: open
**Created**: 2025-06-25T20:22:21+00:00
**Comments**: 1

### Description

# Task 005: Router Interface and Factory

## Summary
Define a Router trait (interface) that abstracts routing operations and create a RouterFactory to centralize router creation logic. This enables clean separation between the server and router implementations, replacing the current enum-based approach.

## Motivation
Current issues:
- Router is an enum with all logic embedded in match statements
- Server code directly depends on specific router implementations
- Difficult to extend with new router types
- No unified interface for routing operations
- Policy creation logic scattered
- No clear initialization pipeline

## Implementation Plan

### 1. Define Router Trait
```rust
// src/router/mod.rs
use crate::core::Worker;
use crate::openai_api_types::{ChatCompletionRequest, CompletionRequest};
use actix_web::{HttpRequest, HttpResponse};

#[async_trait]
pub trait Router: Send + Sync {
    /// Route a chat completion request
    async fn route_chat_completion(
        &self,
        req: 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]  support truncate_dim param of  emdding model

**Link**: https://github.com/sgl-project/sglang/issues/8055
**State**: open
**Created**: 2025-07-15T09:14:07+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

when i use SentenceTransformer i can  truncate the dim of emdding model output use the param "truncate_dim", i hope  sglang can support , thank you

### Related resources

_No response_

---

## Issue #N/A: [Bug] Some FP8 models fail to load

**Link**: https://github.com/sgl-project/sglang/issues/7482
**State**: open
**Created**: 2025-06-23T20:04:33+00:00
**Comments**: 16

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Sglang crashes when loading some FP8 models. I only got Qwen3 FP8 model work, but not others from RedHatAI:
WORKS:
https://huggingface.co/Qwen/Qwen3-32B-FP8

DO NOT WORK
https://huggingface.co/RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic
https://huggingface.co/RedHatAI/Qwen3-32B-FP8-dynamic
https://huggingface.co/RedHatAI/Meta-

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] docker image has lots of CVE issues

**Link**: https://github.com/sgl-project/sglang/issues/8109
**State**: open
**Created**: 2025-07-17T03:32:19+00:00
**Comments**: 4

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I use https://github.com/aquasecurity/trivy to scan, find lots of CVE issues.

Full log is as 

[log.txt](https://github.com/user-attachments/files/21277986/log.txt)

### Reproduction

trivy image lmsysorg/sglang:v0.4.8.post1-cu126 --scanners vuln

### Environment

N

---

## Issue #N/A: [Bug] install sglang by pip install sglang[all]>=0.4.9, work well when run llama3 model, but raise "most likely due to a circular import" error when check ColumnParallelLinear op..

**Link**: https://github.com/sgl-project/sglang/issues/7919
**State**: closed
**Created**: 2025-07-10T05:57:35+00:00
**Closed**: 2025-07-17T07:47:09+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

>>> from sglang.srt.layers.linear import ColumnParallelLinear
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/linear.py", line 30, in <module>
    from sglang.srt.layers.quantization.base_config import (
  File "/usr/local/lib/python3.10/dist-package

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Deploying DeepSeek V3 0324 has blocking decode phase requests by prefill phase ones

**Link**: https://github.com/sgl-project/sglang/issues/7571
**State**: open
**Created**: 2025-06-26T15:47:55+00:00
**Comments**: 3

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I am running the original fp8 DeepSeek V3 0324 model in production, on a Nvidia 8xB200 node, and unfortunately my throughput is not ideal because often times new scheduled prefill requests block other requests in decoding phase. 

I have tried to change the chunked prefill size from 4096 up to 100000, without much avail (large chunked pref

[... truncated for brevity ...]

---

## Issue #N/A: Assessment of the difficulty in porting CPU architecture for sglang

**Link**: https://github.com/sgl-project/sglang/issues/7582
**State**: open
**Created**: 2025-06-27T02:03:04+00:00
**Comments**: 1

### Description

The RISC-V ecosystem is maturing rapidly, with an increasing number of software undergoing migration to RISC-V. We developed a tool named RAX to assess the porting complexity of projects migrating to the RISC-V architecture. RAX evaluates complexity by incorporating Cyclomatic Complexity alongside multiple architecture-specific factors, such as the proportion of assembly code instructions and the frequency of inline function usage. The tool classifies the assessment results into three levels: Low, Middle, and High. For example, the simple project is libtool, the medium project is mesa, and the difficult project is gcc.
Your project sglang is very well-known. Our tool RAX evaluates that the complexity of your project is low. Could you please confirm the assessment accuracy?
More details detected by our tool RAX are shown: the cyclic complexity is 24117. If you want to learn more, please don’t hesitate to us. Thank you.


---

## Issue #N/A: [Feature] Add chat method support to the offline Engine class

**Link**: https://github.com/sgl-project/sglang/issues/8084
**State**: open
**Created**: 2025-07-16T07:37:45+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

`vLLM` now supports both the `generate` and `chat` methods for offline inference. The `generate` method offers more flexibility, while the `chat` method provides a more convenient interface for conversational use cases.

Currently, `SGLang` supports the `generate` method. Are there any plans to add support for the `chat` method as well? This feature would be helpful for users who want a streamlined conversational interface similar to that provided by vLLM.

### Related resources

[vLLM.LLM.chat](https://docs.vllm.ai/en/stable/api/vllm/index.html#vllm.LLM.chat)

---

## Issue #N/A: [Bug] sglang v0.4.8 use remote model required boto3 package

**Link**: https://github.com/sgl-project/sglang/issues/7650
**State**: open
**Created**: 2025-06-30T07:14:29+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

sglang supported  remote model from AWS S3, but when I run docker images: v0.4.8-cu126, raise error:
```[2025-06-29 23:45:28 TP7] Pulling model configs from remote...
[2025-06-29 23:45:28 TP7] Scheduler hit an exception: Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 2631, in

[... truncated for brevity ...]

---

## Issue #N/A: Error when starting minicpm4

**Link**: https://github.com/sgl-project/sglang/issues/7926
**State**: closed
**Created**: 2025-07-10T09:26:49+00:00
**Closed**: 2025-07-15T07:12:47+00:00
**Comments**: 0

### Description

I'm running on H20 and sglang is installed through these command: 
`git clone -b openbmb https://github.com/OpenBMB/sglang.git
cd sglang

pip install --upgrade pip
pip install -e "python[all]"`

when I execute this command I encountered the below error

`python -m sglang.launch_server --model-path OpenBMB/MiniCPM4-8B/ --trust-remote-code --port 30000 --chat-template chatml`

`  File "sglang/python/sglang/srt/models/minicpm.py", line 145, in __init__
    self.rotary_emb.cos_sin_cache = self.rotary_emb._compute_cos_sin_cache()
TypeError: Phi3LongRoPEScaledRotaryEmbedding._compute_cos_sin_cache() missing 3 required positional arguments: 'max_position_embeddings', 'rescale_factors', and 'mscale'`

Did anyone deal with this error before?

---

## Issue #N/A: Task 002: Introduce RoutingPolicy Trait

**Link**: https://github.com/sgl-project/sglang/issues/7535
**State**: open
**Created**: 2025-06-25T20:15:05+00:00
**Comments**: 1

### Description

# Task 002: Introduce RoutingPolicy Trait

## Summary
Create a unified RoutingPolicy trait that enables all routing algorithms (Random, RoundRobin, CacheAware, PowerOfTwo) to work seamlessly in both regular and PD routing modes, eliminating code duplication.

## Problem Statement
The current routing implementation has several issues:
- Routing policies are duplicated between regular and PD routers
- PowerOfTwo policy only exists in PD mode, but could benefit regular routing
- CacheAware logic is copy-pasted with slight variations
- Adding new routing policies requires modifying router internals
- No clear interface or contract for routing algorithms

## Proposed Solution

### 1. RoutingPolicy Trait
Define a trait that all routing policies must implement:

```rust
// src/routing/policies/mod.rs
#[async_trait]
pub trait RoutingPolicy: Send + Sync {
    /// Select a single worker for regular routing
    async fn select_single(
        &self,
        workers: &[Arc<dyn Worker>],
        re

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] [ROCm] GSM8K accuracy issue when using DeekSeek R1 + DP

**Link**: https://github.com/sgl-project/sglang/issues/7692
**State**: closed
**Created**: 2025-07-01T10:05:04+00:00
**Closed**: 2025-07-09T16:52:13+00:00
**Comments**: 2

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Got dp accuracy issue when testing gsm8k benchmark.

GSM8K benchmark (enable DP at lmsysorg/sglang:v0.4.8-rocm630)
Accuracy: 0.755
Invalid: 0.008
Latency: 130.073 s
Output throughput: 1437.044 token/s

GSM8K benchmark (enable DP at lmsysorg/sglang:v0.4.8.post1-rocm630-srt)
Accuracy: 0.002
Invalid: 0.970
Latency: 150.454 s
Output throughput

[... truncated for brevity ...]

---

## Issue #N/A: For PD, Variables in multi thread environment NOT BEING PROTECTED

**Link**: https://github.com/sgl-project/sglang/issues/7894
**State**: closed
**Created**: 2025-07-09T07:31:11+00:00
**Closed**: 2025-07-09T07:31:37+00:00
**Comments**: 0

### Description

file location: disaggregation/mooncake/conn.py
hi, i have successfully reproduce the blog. But when i carefully read the code. I found that in mooncake/conn.py many variables which may be read/write by muti threads have not been protected by threading.Lock(). LIke the variable "transfer_infos".
I just want to confirm whether this would become a problem which can bring unexpected behavior and would be fixed in the future.
Thank u!

---

## Issue #N/A: [Feature] Enable extracting hidden states from intermediate layers

**Link**: https://github.com/sgl-project/sglang/issues/8069
**State**: open
**Created**: 2025-07-15T21:06:59+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

This [PR #3364](https://github.com/sgl-project/sglang/pull/3364) implemented `return_hidden_states` argument which makes the results contain the last layer hidden states in `output["meta_info"]["hidden_states"]`. In certain domains, extracting the hidden states from intermediate layers of the LLM can also be very useful, see [Layer by Layer: Uncovering Hidden Representations in Language Models](https://arxiv.org/abs/2502.02013).

The only alternative to doing this right now is Hugging Face, but implementing this in SGLang's engine could provide a large performance boost to this type of workload.

### Related resources

```
import unittest, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LAYER_ID

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Cannot batch generate or n>1 in sampling_params when input_embeds enabled

**Link**: https://github.com/sgl-project/sglang/issues/7807
**State**: closed
**Created**: 2025-07-06T13:42:01+00:00
**Closed**: 2025-07-08T21:00:43+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I tried to get multi generation outputs via a single post to the server, but I find that if input_embeds is used, the generation will fail and the error message is weired.

In the [document](https://docs.sglang.ai/backend/sampling_params.html), it is stated that `input_embeds` can be a `List[List[List[float]]]` item, but it turns out that 

[... truncated for brevity ...]

---

## Issue #N/A: [About Fused MoE impl] Fused MoE support for EP

**Link**: https://github.com/sgl-project/sglang/issues/7674
**State**: open
**Created**: 2025-07-01T03:47:37+00:00
**Comments**: 0

### Description

Hi dear authors, by reading the source code and running some tests, we find that Sglang implements Fused MoE kernels for non-EP scenarios only.  When EP is enabled, the class EPMoE and DeepEPMoE seem not like Fused MoE implementation. Is there any performance concern on this?
It seems that vLLM and TensorRT-LLM have implemented Fused MoE kernels for both EP and non-EP. 

---

## Issue #N/A: Gemma3n Usage

**Link**: https://github.com/sgl-project/sglang/issues/7574
**State**: open
**Created**: 2025-06-26T21:14:01+00:00
**Comments**: 3
**Labels**: high priority

### Description

~~Due to some compatible issues, we need to manually install the latest version of transformers and timm by:
`pip install -U transformers timm`.~~

The latest SGLang version 0.4.8.post1 could not work with gemma3n, and was fixed in latest main.
To solve the issue, please **install from source** by: (please remove `uv` if you don't use)

```
git clone https://github.com/sgl-project/sglang.git
cd sglang
uv pip install -e "python[all]"
```

Launch the server with:
`python -m sglang.launch_server --model-path google/gemma-3n-E4B-it --attention-backend fa3`

If you encounter any issue when running the gemma3n, welcome to comment under this issue.

Known issues:
1. `TypeError: unsupported operand type(s) for %: 'list' and 'int'` : Please follow above instruction

---

## Issue #N/A: [Bug] Kimi K2 function call failed if set "strict": true in tool calls definition.

**Link**: https://github.com/sgl-project/sglang/issues/8087
**State**: open
**Created**: 2025-07-16T08:21:35+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Kimi K2 function call failed if set "strict": true in tool calls definition.

```
[2025-07-16 14:47:06] [2025-07-15 23:47:06] INFO:     59.82.59.90:0 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
[2025-07-16 14:47:06] [2025-07-15 23:47:06] Error in request: 
[2025-07-16 14:47:06] Traceback (most recent call last):
[2025-

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] NVTX Tracing Error with DeepSeekV3 on Output_LEN=2 (nsys 2025.3.1.90)

**Link**: https://github.com/sgl-project/sglang/issues/8017
**State**: open
**Created**: 2025-07-14T08:17:06+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When using NVIDIA Tools Extension (NVTX) markers to profile DeepSeekV3 with nsys 2025.3.1.90, the generated trace exhibits incorrect hierarchical nesting levels between NVTX tags when generating multiple output tokens (OUTPUT_LEN=2). However, when output = 1, the nsys file is fine.

### Reproduction


1. **Modify DeepSeekV3 Code:**
在Deepse

[... truncated for brevity ...]

---

## Issue #N/A: [question of eagle] why create a new token_to_kv_pool in draft model worker？

**Link**: https://github.com/sgl-project/sglang/issues/7659
**State**: open
**Created**: 2025-06-30T15:30:11+00:00
**Comments**: 0

### Description

in model_runner.py:
if is_draft_worker is true,  we resue req_to_token_pool and token_to_kv_pool_allocator with target model workder, but why create a new token_to_kv_pool? 

---

