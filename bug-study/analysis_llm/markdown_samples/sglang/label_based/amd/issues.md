# amd - issues

**Total Issues**: 29
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 8
- Closed Issues: 21

### Label Distribution

- amd: 29 issues
- inactive: 14 issues
- help wanted: 4 issues
- documentation: 3 issues
- high priority: 2 issues
- deepseek: 2 issues
- await-response: 2 issues
- RLHF: 1 issues
- feature: 1 issues
- good first issue: 1 issues

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

## Issue #N/A: [Feature] Provide pre-built wheel files for AMD GPUs

**Link**: https://github.com/sgl-project/sglang/issues/6060
**State**: open
**Created**: 2025-05-06T16:44:39+00:00
**Comments**: 0
**Labels**: amd

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Hi,

The installation of SGLang for CUDA is easier than AMD GPUs as there are pre-built wheel files as seen at https://docs.sglang.ai/start/install.html .
While docker images are available for AMD, docker is pretty heavy, and doesn't fit in some infrastructures.

It would be great if pre-built wheel files were provided for ROCm, similarly to what Pytorch does.

Thank you for considering this,
Best regards,
Epliz

### Related resources

_No response_

---

## Issue #N/A: [Bug] OutOfResources: out of resource: shared memory, Required: 196608, Hardware limit: 65536 when run deepseek-r1 on sglang 0.4.6 with AMD Mi308

**Link**: https://github.com/sgl-project/sglang/issues/6001
**State**: open
**Created**: 2025-05-04T01:20:02+00:00
**Comments**: 2
**Labels**: amd

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Run the deepseek-r1 with AMD MI308, sglang 0.6.4

The error log is as following:

python3 -m sglang.launch_server --model /model/deepseek-r1 --trust-remote-code --tp 8 --chunked-prefill-size 130172 --enable-torch-compile --torch-compile-max-bs 64 --cuda-graph-max-bs 16  --mem-fraction-static 0.7
INFO 05-04 01:04:03 __init__.py:179] Automat

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Segmentation Fault on AMD MI300X

**Link**: https://github.com/sgl-project/sglang/issues/5987
**State**: closed
**Created**: 2025-05-02T20:08:36+00:00
**Closed**: 2025-06-12T05:58:47+00:00
**Comments**: 4
**Labels**: amd

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I am unable to run online benchmarks on the new AMD SGLang Docker image due to a segmentation fault error.

- Hardware: MI300X
- Docker image: `rocm/sgl-dev:upstream_20250324`

I am attaching the error log because it exceeds the character limit.

[segfault_error.log](https://github.com/user-attachments/files/20017838/segfault_error.log)

#

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] RuntimeError: HIP error: invalid device function in Mi100

**Link**: https://github.com/sgl-project/sglang/issues/5775
**State**: open
**Created**: 2025-04-27T07:43:22+00:00
**Comments**: 0
**Labels**: amd

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Environment: The docker lmsysorg/sglang:v0.4.5.post3-rocm630
Just run python3 -m sglang.launch_server --model-path mistralhf-gptq --host 0.0.0.0 --port 30000 to launch sglang to serve a mistral gptq model.

When I launch sglang or vllm to serve a mistral model on Mi100 cards, the folloing error was shown:

WARNING: Published ports are disc

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Unsupported conversion from 'f8E4M3FN' to 'f16' When Inferencing DeepSeek-v3 with enable_ep_moe on AMD MI300x

**Link**: https://github.com/sgl-project/sglang/issues/5705
**State**: open
**Created**: 2025-04-24T08:01:36+00:00
**Comments**: 2
**Labels**: amd

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

offline batch inference, using 8x AMD MI300x, when enable_ep_moe, error happens when starts to prefill.
```python 
/sgl-workspace/sglang/python/sglang/srt/layers/moe/ep_moe/kernels.py:606:42: error: Unsupported conversion from 'f8E4M3FN' to 'f16'
            accumulator += tl.dot(a_tile, b_tile.T) * a_scale * b_scale[None, :]
             

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] After updating from 0.4.5.post2 to 0.4.5.post3, the following error is reported: Child process unexpectedly failed with an exit code 256. pid=32885.

**Link**: https://github.com/sgl-project/sglang/issues/5637
**State**: open
**Created**: 2025-04-22T13:37:01+00:00
**Comments**: 1
**Labels**: amd

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

[2025-04-22 21:28:53 TP0] awq quantization is not fully optimized yet. The speed can be slower than non-quantized models.
[2025-04-22 21:28:53 TP0] Attention backend not set. Use triton backend by default.
[2025-04-22 21:28:53 TP0] Init torch distributed begin.
[W422 21:28:53.246570644 HIPAllocatorConfig.h:29] Warning: expandable_segments 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] AMD ROCm system: Inference error

**Link**: https://github.com/sgl-project/sglang/issues/5414
**State**: open
**Created**: 2025-04-15T08:46:38+00:00
**Comments**: 1
**Labels**: amd

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When starting the service of a multi-GPU model on the ROCm platform and requesting inference, the obtained response is completely wrong.

prompt="List 3 countries and their capitals."：

response：ChatCompletion(id='5379cef8260f4cc6a9272581759b199b', choices=[Choice(finish_reason='length', index=0, logprobs=None, message=ChatCompletionMessag

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Support for Llama 4 on ROCm?

**Link**: https://github.com/sgl-project/sglang/issues/5362
**State**: closed
**Created**: 2025-04-14T02:33:17+00:00
**Closed**: 2025-07-01T22:25:55+00:00
**Comments**: 3
**Labels**: amd

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hi team,

I’m wondering if sglang currently supports running Llama 4 models on AMD ROCm (e.g., 8xMI300X). I tried using your latest ROCm Sglang SRT Docker image. While the server can start, it produces incorrect results when using the standard endpoint with the Llama 4 Maverick model. For the FP8 Maverick version, it fails to start at all.

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] SGLang on ROCm - NameError: name 'torch_memory_saver' is not defined

**Link**: https://github.com/sgl-project/sglang/issues/5093
**State**: closed
**Created**: 2025-04-05T23:11:55+00:00
**Closed**: 2025-06-08T00:21:35+00:00
**Comments**: 7
**Labels**: high priority, inactive, amd, RLHF

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug
[RCOm Docker - `lmsysorg/sglang:v0.4.4.post3-rocm630-srt`]
The issue arises from here:
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/torch_memory_saver_adapter.py#L48

1. In line 6, if the code fails to import torch_memory_saver, it just bypasses instead of triggering any error. Thus, if the code calls line46 class and u

[... truncated for brevity ...]

---

## Issue #N/A: Question sgl_kernel on amd paltforms

**Link**: https://github.com/sgl-project/sglang/issues/3965
**State**: closed
**Created**: 2025-02-28T16:13:40+00:00
**Closed**: 2025-05-04T00:21:06+00:00
**Comments**: 12
**Labels**: inactive, amd

### Description

Hey,

I get the following warning when running on mi300:

```
sgl-kernel is not available on Non-NV platforms. Fallback to other kernel libraries.
```

Even though I compiled sgl_kernel and that is what the doc says https://docs.sglang.ai/start/install.html#method-2-from-source.

In the code, both sglang/python/sglang/srt/layers/activation.py and sglang/python/sglang/srt/layers/layernorm.py are gated behind
```
is_cuda_available()
```

Doe this applies only to layernorm.py and activations.py ? then the message may not be explicit enough, leading me to believe I didnt even built sgl kernel.

---

## Issue #N/A: [Feature] GPU inference on AMD Ryzen AI (370HX-890M) iGPU + NPU

**Link**: https://github.com/sgl-project/sglang/issues/3823
**State**: closed
**Created**: 2025-02-24T15:49:18+00:00
**Closed**: 2025-04-26T00:17:55+00:00
**Comments**: 2
**Labels**: inactive, feature, amd

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Ryzen AI devices have been out since mid 2024 yet there's no end user friendly local inference engine that can use the iGPU or the NPU for inference. Some people seem to be able to make it working using hacks but it's still a hit or miss and you need to build your own custom room and hip packages to it to kind of work. 

### Related resources

_No response_

---

## Issue #N/A: [Bug] fix amd runner offline

**Link**: https://github.com/sgl-project/sglang/issues/3755
**State**: closed
**Created**: 2025-02-21T09:58:30+00:00
**Closed**: 2025-03-13T21:10:31+00:00
**Comments**: 1
**Labels**: high priority, amd

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

![Image](https://github.com/user-attachments/assets/1fa39f19-18ce-4020-a793-b5f1a3647c4b)

![Image](https://github.com/user-attachments/assets/2be670db-2e32-4f9c-8837-106f8aa5d200)

@saienduri May you help fix it? Thanks!

### Reproduction

N/A

### Environment

N/A

---

## Issue #N/A: [Feature] is flex attention compatible?

**Link**: https://github.com/sgl-project/sglang/issues/3479
**State**: closed
**Created**: 2025-02-11T07:30:03+00:00
**Closed**: 2025-04-13T00:43:12+00:00
**Comments**: 2
**Labels**: help wanted, inactive, amd

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

context:
In the future, we plan on extending this support to allow for quantized versions of attention or things like [RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/) as well.

https://pytorch.org/blog/flexattention/

### Related resources

_No response_

---

## Issue #N/A: [Bug] 4x8 Mi210 Deepseek V3 runtime error

**Link**: https://github.com/sgl-project/sglang/issues/3400
**State**: closed
**Created**: 2025-02-08T11:17:41+00:00
**Closed**: 2025-04-10T00:18:03+00:00
**Comments**: 2
**Labels**: inactive, amd, deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I tried to start 4 nodes with 32 Mi210 GPUs to launch Deepseek R1 using sglang and I've already converted the weights to bf16.

``` bash
[2025-02-08 18:30:47] INFO:     Started server process [445197]
[2025-02-08 18:30:47] INFO:     Waiting for application startup.
[2025-02-08 18:30:47] INFO:     Application startup complete.
[2025-02-08 1

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] AttributeError: module 'vllm._custom_ops' has no attribute 'silu_and_mul'

**Link**: https://github.com/sgl-project/sglang/issues/3392
**State**: closed
**Created**: 2025-02-08T06:32:39+00:00
**Closed**: 2025-05-03T00:18:16+00:00
**Comments**: 5
**Labels**: inactive, amd, deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hell folks,

 I'm attempting to deploy DeepSeek-R1 with SGLang on an AMD MI300X, but I'm encountering compatibility issues. 
Could someone please help me troubleshoot these issues?


### Reproduction

1. build and install **triton 3.0.0** from source
2. build and install **vllm v0.7.2** from source
3. build and install **sglang** (rev 0a6f

[... truncated for brevity ...]

---

## Issue #N/A: [Docs] Add docs for running SGLang on AMD

**Link**: https://github.com/sgl-project/sglang/issues/3245
**State**: closed
**Created**: 2025-02-01T00:23:16+00:00
**Closed**: 2025-05-21T15:40:21+00:00
**Comments**: 4
**Labels**: documentation, good first issue, help wanted, amd

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

That has long been waiting, we should add a docs on how to run SGLang on AMD devices.

https://github.com/sgl-project/sglang/issues/3219
https://github.com/sgl-project/sglang/issues/3243
https://github.com/sgl-project/sglang/issues/3200
https://github.com/sgl-project/sglang/pull/3208
https://github.com/sgl-project/sglang/issues/3198

Here is something related. To me, I think we should add a docs on how to:
 
1. configure environment in AMD GPU;
2. how to install sglang;
3. how to run a llama model;
4. how to run deepseek V3 models.

### Related resources

_No response_

---

## Issue #N/A: [Feature] Instructions for running Sglang on AMD RX 7900 XTX (gfx1100) ROCm 6.2.4

**Link**: https://github.com/sgl-project/sglang/issues/3243
**State**: closed
**Created**: 2025-01-31T20:01:43+00:00
**Closed**: 2025-05-14T00:19:11+00:00
**Comments**: 10
**Labels**: documentation, inactive, amd

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Hello, 

If anyone is interested, here's how I run SGlang on the AMD RX 7900 XTX (gfx1100) with ROCm 6.2.4.  Currently, the attention backend is based on Triton. It seems that flashInfer support is under development. Hope it helps.

Create a Dockerfile, which is based on the vLLM ROCm dockerfile:

```
# default base image
ARG REMOTE_VLLM="0"
ARG USE_CYTHON="0"
ARG BUILD_RPD="1"
ARG COMMON_WORKDIR=/app
ARG BASE_IMAGE=rocm/vllm-dev:base

FROM ${BASE_IMAGE} AS base

ARG ARG_PYTORCH_ROCM_ARCH
ENV PYTORCH_ROCM_ARCH=${ARG_PYTORCH_ROCM_ARCH:-${PYTORCH_ROCM_ARCH}}

# Install some basic utilities
RUN apt-get update -q -y && apt-get install -q -y \
    sqlite3 libsqlite3-dev libfmt-dev libmsgpack-dev libsuitesparse-dev
# Rem

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] [ROCm] Running DeepSeek V3 on MI300X, getting "Config not found, Performance might be sub-optimal" error

**Link**: https://github.com/sgl-project/sglang/issues/3219
**State**: closed
**Created**: 2025-01-30T19:34:09+00:00
**Closed**: 2025-04-12T00:17:45+00:00
**Comments**: 13
**Labels**: help wanted, inactive, amd

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I am running DeepSeek v3 on a node with 8xMI300X GPUs on ROCm 6.3.1. I am able to run it using an image built from `Dockerfile.rocm` in `docker`, however I have noticed this warning show up:
```
Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at <multiple config files>
```
In the containe

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Tried to run DeepSeek V3 by amd instructions

**Link**: https://github.com/sgl-project/sglang/issues/3200
**State**: closed
**Created**: 2025-01-28T22:33:58+00:00
**Closed**: 2025-04-03T00:17:38+00:00
**Comments**: 4
**Labels**: documentation, help wanted, inactive, amd

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I tried to use [AMD instruction](https://www.amd.com/en/developer/resources/technical-articles/amd-instinct-gpus-power-deepseek-v3-revolutionizing-ai-development-with-sglang.html) but i have an error.

### Reproduction

After running in a container
```
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --port 30000 --tp 8

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] ERROR: No matching distribution found for vllm==0.6.3.post2.dev1; extra == "srt-hip"

**Link**: https://github.com/sgl-project/sglang/issues/3189
**State**: closed
**Created**: 2025-01-28T01:43:00+00:00
**Closed**: 2025-05-03T00:18:15+00:00
**Comments**: 10
**Labels**: inactive, amd

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Trying to install sglang for AMD, but hitting this issue.  I was following this: https://docs.sglang.ai/start/install.html#method-2-from-source

Happens with latest sglang or the version 0.4.2 specified in the instructions.

```
base) root@6e2c9e6215c7:/# conda activate sglang
(sglang) root@6e2c9e6215c7:/# conda install python=3.10 -y
Chan

[... truncated for brevity ...]

---

## Issue #N/A: Warning while running Deepseek-V3

**Link**: https://github.com/sgl-project/sglang/issues/2921
**State**: closed
**Created**: 2025-01-16T11:59:39+00:00
**Closed**: 2025-02-06T13:24:17+00:00
**Comments**: 2
**Labels**: amd

### Description

Hi, we are trying to run DeepSeek-V3 using SGLang, on both A100 and MI300x GPUs.

We receive these warnings on both systems-

<img width="1917" alt="Image" src="https://github.com/user-attachments/assets/086f5324-30a5-439a-b9e1-1ec22a0045a9" />

Are there any specific flags that need to be enabled for deepseek?

Background:
We have followed the steps in the documentation and are unable to increase throughput beyond 10 tokens/second on either system.. tried everything.
We are running like so-
`python3 -m sglang.launch_server --model-path ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V3/snapshots/4c1f24cc10a2a1894304c7ab52edd9710c047571/ --port 8000 --tp 8 --context-length 122880 --trust-remote-code`



---

## Issue #N/A: [Bug] The performance of v0.4.1 on AMD GPU is lower than v0.4.0

**Link**: https://github.com/sgl-project/sglang/issues/2675
**State**: closed
**Created**: 2024-12-31T03:56:50+00:00
**Closed**: 2025-03-02T00:18:46+00:00
**Comments**: 2
**Labels**: inactive, amd

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

We found that the performance test results on the latest sglang v0.4.1 version were lower than v0.4.0. The following are the test results
![v0 4 0](https://github.com/user-attachments/assets/0aa783d9-2f4a-4c7a-8670-9b6203428ba1)
![v0 4 1](https://github.com/user-attachments/assets/3410e457-34ff-4992-85fd-fea88f8cc027)

By com

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] add AMD CIs

**Link**: https://github.com/sgl-project/sglang/issues/2621
**State**: closed
**Created**: 2024-12-27T18:18:37+00:00
**Closed**: 2025-02-28T00:17:01+00:00
**Comments**: 1
**Labels**: inactive, amd

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

As discussed offline, we should also support AMD CIs. @HaiShaw cc @merrymercy 

### Related resources

_No response_

---

## Issue #N/A: [Bug] Deepseek-v2-lite AMD MI300 run failed

**Link**: https://github.com/sgl-project/sglang/issues/2384
**State**: closed
**Created**: 2024-12-07T07:24:15+00:00
**Closed**: 2024-12-31T02:41:23+00:00
**Comments**: 8
**Labels**: amd

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

#### Deepseek-v2 ROCM Env triton compiler error
Bug report:
```bash
WARNING 12-07 02:43:18 rocm.py:17] `fork` method is not supported by ROCm. VLLM_WORKER_MULTIPROC_METHOD is overridden to `spawn` instead.
[2024-12-07 02:43:23] server_args=ServerArgs(model_path='/data/deepseek-v2-lite/', tokenizer_path='/data/deepseek-v2-lite

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] amdgpu，tp-size=2，Detected errors during sampling! NaN in the logits.

**Link**: https://github.com/sgl-project/sglang/issues/1953
**State**: closed
**Created**: 2024-11-08T04:44:58+00:00
**Closed**: 2025-01-29T00:16:25+00:00
**Comments**: 6
**Labels**: await-response, inactive, amd

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

python3 -m sglang.launch_server --model-path  /root/.xinference/cache/qwen2_5-instruct-gptq-7b-Int8/ --port 30000 --mem-fraction-static  0.8 --kv-cache-dtype int8 --attention-backend triton --sampling-backend pytorch --tp-size 2
WARNING 11-08 04:42:43 rocm.py:13] `fork` method is not supported by ROCm. VLLM_WORKER_MULTIPROC_METHOD is over

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] tp-size=2，model launch error

**Link**: https://github.com/sgl-project/sglang/issues/1945
**State**: closed
**Created**: 2024-11-07T06:14:03+00:00
**Closed**: 2025-01-29T00:16:26+00:00
**Comments**: 5
**Labels**: await-response, inactive, amd

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

tp-size=2, model launch is frozen.

### Reproduction

 python3 -m sglang.launch_server --model-path  /root/.xinference/cache/qwen2_5-instruct-gptq-7b-Int8/ --port 30000 --mem-fraction-static  0.8 --tp-size 2 --kv-cache-dtype int8 --attention-backend triton --sampling-backend pytorch --enable-torch-compile

### Environment

amd gpu RTX 7900

[... truncated for brevity ...]

---

## Issue #N/A: TP8 scheduling overhead is very high for small model, Llama 3 8B on AMD

**Link**: https://github.com/sgl-project/sglang/issues/1857
**State**: closed
**Created**: 2024-10-31T20:38:46+00:00
**Closed**: 2025-01-10T19:56:20+00:00
**Comments**: 18
**Labels**: amd

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

When I benchmark TP1, the throughput is great.

Backend:                                 sglang
Traffic request rate:                    inf
Successful requests:                     4000
Benchmark duration (s):                  22.18
Total input tokens:                      257409
Total generated tokens:                  257960
Tot

[... truncated for brevity ...]

---

