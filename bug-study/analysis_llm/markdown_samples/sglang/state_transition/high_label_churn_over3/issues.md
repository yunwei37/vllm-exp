# high_label_churn_over3 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 10
- Closed Issues: 20

### Label Distribution

- high priority: 25 issues
- help wanted: 16 issues
- good first issue: 15 issues
- enhancement: 9 issues
- inactive: 9 issues
- performance: 8 issues
- quant: 5 issues
- deepseek: 5 issues
- flashinfer: 5 issues
- collaboration: 5 issues

---

## Issue #N/A: [Feature] DeepSeek V3 optimization

**Link**: https://github.com/sgl-project/sglang/issues/2591
**State**: closed
**Created**: 2024-12-26T08:52:39+00:00
**Closed**: 2025-03-25T04:10:46+00:00
**Comments**: 52
**Labels**: enhancement, high priority, performance, quant

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Adoption

[SGLang adoption for DeepSeek V3 and R1](https://github.com/sgl-project/sglang/discussions/3322)

### Usage

User Guide for Existing System (Installation & Launch)

https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3

Please use the latest version [v0.4.2.post4](https://pypi.org/project/sglang/0.4.2.post4/). Please prefer to use docker image. `docker pull lmsysorg/sglang:latest`

For running on AMD MI300X, use this as a reference. [Running DeepSeek-R1 on a single NDv5 MI300X VM](https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/running-deepseek-r1-on-a-single-ndv5-mi300x-vm/4372726)

### Features

- [x] Support CUDA Graph @HandH1998 @ispobock 
- [x] Support Torch compile @is

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] support torch compile cache for DeepSeek V3/R1

**Link**: https://github.com/sgl-project/sglang/issues/3614
**State**: closed
**Created**: 2025-02-16T16:18:21+00:00
**Closed**: 2025-02-21T18:18:09+00:00
**Comments**: 6
**Labels**: good first issue, help wanted, high priority, deepseek

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

The time taken for each startup is currently too long when torch compile is enabled. It needs optimization.

### Related resources

_No response_

---

## Issue #N/A: [Track] DeepSeek V3/R1 nextn progress

**Link**: https://github.com/sgl-project/sglang/issues/3472
**State**: closed
**Created**: 2025-02-10T14:46:03+00:00
**Closed**: 2025-03-25T04:13:25+00:00
**Comments**: 8
**Labels**: enhancement, high priority, flashinfer, deepseek

### Description

## Triton Backend

@ispobock @pankajroark 

- [x] [refactor triton backend 1](https://github.com/sgl-project/sglang/pull/3292), [2](https://github.com/sgl-project/sglang/pull/3309)

- [x] [support custom mask](https://github.com/sgl-project/sglang/pull/3317)

- [x] [support EAGLE 2](https://github.com/sgl-project/sglang/pull/3466)

- [x] [compatible with CUDA Graph](https://github.com/sgl-project/sglang/pull/3500)

- [x] [support nextn I (single MTP head)](https://github.com/sgl-project/sglang/pull/3582)

- [x] support next II (multi MTP heads) (WIP @pankajroark )

## FlashInfer Backend

@zhyncs @yzh119 

- [x] compatible with disable MLA

- [x] support FlashInfer nightly MLA ragged prefill and CUDA Core MLA decoding

- [x] support FlashInfer v0.2.0.post3 MLA ragged, paged prefill and decoding (@zhyncs @yzh119 )

- [x] nextn parts can be shared with Triton Backend

## EAGLE 2

@zhyncs @Ying1123 

- [x] implement sampling kernel in [sgl-kernel](https://github.com/sgl-project/sglang/tree

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] FlashInfer new version integration

**Link**: https://github.com/sgl-project/sglang/issues/2620
**State**: closed
**Created**: 2024-12-27T18:14:29+00:00
**Closed**: 2025-03-11T00:17:39+00:00
**Comments**: 3
**Labels**: enhancement, high priority, inactive, flashinfer

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

### Related resources

_No response_

---

## Issue #N/A: [Feature] Rewrite the SRT Backend docs

**Link**: https://github.com/sgl-project/sglang/issues/2660
**State**: closed
**Created**: 2024-12-30T07:49:17+00:00
**Closed**: 2025-05-24T21:27:16+00:00
**Comments**: 3
**Labels**: documentation, good first issue, help wanted, RLHF

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

This doc has been outdated for a long time:

https://sgl-project.github.io/backend/backend.html#backend-sglang-runtime-srt

1. Only keep an explanation for server arguments and give the link to sampling parameters.
2. Add essential explanation for server arguments. Remember to add these kinds of arguments. https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models
3. A group of parameters have ##, ### is not allowed.
4. Use Models From ModelScope and Run Llama 3.1 405B move to reference, and potentially adds docs for deepseek.
5. change main readme.md.


### Related resources

No such.

---

## Issue #N/A: [Feature] Integrate CUTLASS FP8 GEMM into sgl-kernel

**Link**: https://github.com/sgl-project/sglang/issues/2472
**State**: closed
**Created**: 2024-12-12T20:08:31+00:00
**Closed**: 2025-02-12T00:16:40+00:00
**Comments**: 4
**Labels**: high priority, inactive, performance, quant

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

ref 
https://github.com/NVIDIA/cutlass/pull/1932/files

### Related resources

_No response_

---

## Issue #N/A: [Tracker] FA3 performance on sm80

**Link**: https://github.com/sgl-project/sglang/issues/5938
**State**: open
**Created**: 2025-05-01T02:14:42+00:00
**Comments**: 2
**Labels**: good first issue, help wanted, high priority, collaboration

### Description

```bash
git clone https://github.com/sgl-project/sglang
cd sglang
pip3 install -e "python[all]"
```

```bash
--attention-backend fa3
```

---

## Issue #N/A: [Feature] beat torch compile

**Link**: https://github.com/sgl-project/sglang/issues/4748
**State**: closed
**Created**: 2025-03-25T06:18:28+00:00
**Closed**: 2025-05-26T16:55:12+00:00
**Comments**: 7
**Labels**: good first issue, help wanted, high priority, collaboration, performance

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

Last year and in the first few months of this year, a significant part of my work focused on removing vLLM dependency. Many reliable teammates joined in this process, and we successfully removed the vLLM dependency on the NVIDIA platform for SGLang. Next, I will co-lead progress on beat torch compile. Past experience shows that torch compile is effective - we just need to write some simple torch ops and let torch compile handle the rest. However, in actual production serving, it is not as smooth as expected - for example, slow startup even with cache enabled, compatibility issues when upgrading torch versions leading to previous features breaking in new versions. We need to profile, benchmark, rewrite th

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] support and turn on chunked prefill by default for VLM

**Link**: https://github.com/sgl-project/sglang/issues/5250
**State**: closed
**Created**: 2025-04-10T18:45:57+00:00
**Closed**: 2025-05-26T16:56:01+00:00
**Comments**: 2
**Labels**: good first issue, help wanted, high priority, MLLM

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct

### Related resources

_No response_

---

## Issue #N/A: [Feature] Support more multi-modal input for VLM

**Link**: https://github.com/sgl-project/sglang/issues/5964
**State**: open
**Created**: 2025-05-02T02:28:40+00:00
**Comments**: 4
**Labels**: good first issue, help wanted, feature, MLLM

### Description

### Motivation

The current endpoint only supports image data input, limiting its flexibility for diverse VLM use cases. We need additional input formats, particularly for RL applications:
(Could be split into multiple PRs)

- [x] Pre-computed Image Embeddings
- [ ] Pixel Values
- [ ] Pixel Value Range Parameters (min_pixel/max_pixel) for qwen-vl

Welcome to propose more.

#### Benefits

1. Enhanced flexibility for RL workflows
2. Reduced preprocessing overhead
3. Better integration with existing pipelines

---

## Issue #N/A: [Tracker] SGLang v0.4.5.post1 performance on H200

**Link**: https://github.com/sgl-project/sglang/issues/5514
**State**: closed
**Created**: 2025-04-18T02:46:46+00:00
**Closed**: 2025-04-29T19:47:52+00:00
**Comments**: 9
**Labels**: high priority, collaboration, performance, deepseek

### Description

**Update**:
**see the latest benchmark results in another post https://github.com/sgl-project/sglang/pull/5611#issuecomment-2819965621** 


```bash
# launch server
# First, warm up for DeepGEMM
# SGLang uses FA3 backend by default since v0.4.5.post1
# Use dp 8 for offline use case
SGL_ENABLE_JIT_DEEPGEMM=1 python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --enable-dp-attention --dp-size 8

# Random 1k, 2k
python3 -m sglang.bench_serving --backend sglang-oai --num-prompts 50 --request-rate 10 --dataset-name random --random-input-len 1000 --random-output-len 2000 --random-range-ratio 1

# Random 5k, 1k
python3 -m sglang.bench_serving --backend sglang-oai --num-prompts 50 --request-rate 10 --dataset-name random --random-input-len 5000 --random-output-len 1000 --random-range-ratio 1

# Random 10k, 500
python3 -m sglang.bench_serving --backend sglang-oai --num-prompts 50 --request-rate 10 --dataset-name random --random-input-len 10000 --random-output

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Can router support prometheus metrics

**Link**: https://github.com/sgl-project/sglang/issues/3393
**State**: closed
**Created**: 2025-02-08T06:42:46+00:00
**Closed**: 2025-04-28T00:19:29+00:00
**Comments**: 3
**Labels**: enhancement, inactive, feature, router

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

K8s is often used to deploy applications online. After the router module is introduced, related service indicator monitoring is also required. Therefore, similar to https://github.com/sgl-project/sglang/pull/1853 provided by the server, does it support the collection of monitoring indicators of the router?

### Related resources

_No response_

---

## Issue #N/A: [Feature] support MiniMax

**Link**: https://github.com/sgl-project/sglang/issues/2898
**State**: open
**Created**: 2025-01-15T06:36:10+00:00
**Comments**: 5
**Labels**: good first issue, help wanted, high priority, new-model

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

ref https://github.com/MiniMax-AI/MiniMax-01

### Related resources

_No response_

---

## Issue #N/A: [Bug] fix gemma-2-2b-it-FP8 accuracy

**Link**: https://github.com/sgl-project/sglang/issues/4324
**State**: closed
**Created**: 2025-03-12T01:27:58+00:00
**Closed**: 2025-05-21T09:30:43+00:00
**Comments**: 8
**Labels**: bug, good first issue, help wanted, high priority, quant

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

The accuracy of `neuralmagic/gemma-2-2b-it-FP8` drops from 0.62 to 0.52 in the main branch. It was detected by our nightly CI run. We need to fix this.

```
neuralmagic/gemma-2-2b-it-FP8 | 0.512 | 0.6
```
https://github.com/sgl-project/sglang/actions/runs/13800885290

### Reproduction

N/A

### Environment

N/A

---

## Issue #N/A: [Feature] enable SGLang custom all reduce by default

**Link**: https://github.com/sgl-project/sglang/issues/4436
**State**: closed
**Created**: 2025-03-14T19:46:52+00:00
**Closed**: 2025-03-29T02:50:50+00:00
**Comments**: 5
**Labels**: good first issue, help wanted, high priority, performance

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

We need community users to help test these cases. After confirming that there are no issues, we will default to using the custom all reduce implemented in SGLang. You can reply with your test results below this issue. Thanks!

**GPU Hardware Options**:
- H100/H200/H20/H800/A100

**Model Configurations with Tensor Parallelism (TP) Settings**:
- Llama 8B with TP 1/2/4/8
- Llama 70B with TP 4/8
- Qwen 7B with TP 1/2/4/8
- Qwen 32B with TP 4/8
- DeepSeek V3 with TP 8/16

**Environment Variables**:
```
export USE_VLLM_CUSTOM_ALLREDUCE=0
export USE_VLLM_CUSTOM_ALLREDUCE=1
```

**Benchmarking Commands**:
```bash
python3 -m sglang.bench_one_batch --model-path model --batch-size --input 128 --output 8
python3 -m sglang.benc

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Testing new Llama-3_3-Nemotron-Super-49B-v1 by Nvidia: "Model architectures ['DeciLMForCausalLM'] are not supported for now."

**Link**: https://github.com/sgl-project/sglang/issues/4689
**State**: open
**Created**: 2025-03-23T05:40:20+00:00
**Comments**: 14
**Labels**: enhancement, good first issue, help wanted, new-model

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I tried to run on SGLang Llama-3_3-Nemotron-Super-49B-v1 recently announced by Nvidia.

It seems not to be yet supported by SGLang since `DeciLMForCausalLM`is not yet accepted by SGLang. See below.

Can you add corresponding support?

```
Scheduler hit an exception: Traceback (most recent call last):
  File "/usr/local/lib/python3.12/site-

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] add kernel level benchmark

**Link**: https://github.com/sgl-project/sglang/issues/2402
**State**: closed
**Created**: 2024-12-08T11:06:34+00:00
**Closed**: 2025-05-21T09:31:00+00:00
**Comments**: 2
**Labels**: enhancement, good first issue, help wanted, high priority, flashinfer

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

use triton benchmark utils https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html#triton.testing.do_bench to benchmark kernels (flashinfer, triton, vllm, tensorrt llm, cudnn etc)

### Related resources

_No response_

---

## Issue #N/A: [Feature] support mistral small vlm

**Link**: https://github.com/sgl-project/sglang/issues/4518
**State**: closed
**Created**: 2025-03-17T18:43:18+00:00
**Closed**: 2025-05-21T15:27:30+00:00
**Comments**: 16
**Labels**: enhancement, good first issue, help wanted, high priority

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

https://mistral.ai/fr/news/mistral-small-3-1

### Related resources

_No response_

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

## Issue #N/A: [Bug] fix dsv3 awq issue

**Link**: https://github.com/sgl-project/sglang/issues/4462
**State**: closed
**Created**: 2025-03-16T05:27:20+00:00
**Closed**: 2025-04-07T02:17:41+00:00
**Comments**: 4
**Labels**: bug, high priority, performance, quant

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

as titled

### Reproduction

N/A

### Environment

N/A

---

## Issue #N/A: [Feature] Accuracy test of VLM

**Link**: https://github.com/sgl-project/sglang/issues/3142
**State**: open
**Created**: 2025-01-26T06:25:40+00:00
**Comments**: 7
**Labels**: good first issue, help wanted, high priority, MLLM

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

In sglang, LLMs have accuracy tests with Hugging Face models:

https://github.com/sgl-project/sglang/blob/main/test/srt/models/test_generation_models.py

https://github.com/sgl-project/sglang/blob/main/test/srt/test_nightly_math_eval.py

We need similar one for VLM also.

### Related resources

_No response_

---

## Issue #N/A: [Bug] fix nightly test

**Link**: https://github.com/sgl-project/sglang/issues/3396
**State**: closed
**Created**: 2025-02-08T08:58:41+00:00
**Closed**: 2025-04-10T00:17:59+00:00
**Comments**: 1
**Labels**: bug, high priority, inactive, flashinfer

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

After upgrading FlashInfer, there are issues with the nightly tests.

### Reproduction

N/A

### Environment

N/A

---

## Issue #N/A: [Bug] llama 3 405b fb fp8 issue

**Link**: https://github.com/sgl-project/sglang/issues/7124
**State**: open
**Created**: 2025-06-12T08:58:27+00:00
**Comments**: 2
**Labels**: bug, good first issue, help wanted, high priority

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

as titled

### Reproduction

N/A

### Environment

N/A

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

## Issue #N/A: [Kernel] cuDNN attention backend

**Link**: https://github.com/sgl-project/sglang/issues/2272
**State**: open
**Created**: 2024-11-30T06:36:16+00:00
**Comments**: 3
**Labels**: enhancement, good first issue, help wanted, high priority, inactive

### Description

cuDNN provides very fast attention implementation and it is well maintained by NVIDIA. We would like to add a new attention backend based on cudnn.  

## Steps
1. Learn this cudnn paged attention python api. https://github.com/NVIDIA/cudnn-frontend/blob/v1.8.0/samples/python/52_scaled_dot_product_attention_with_paged_caches.ipynb
2. Add a new attention backend "cudnn" here https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/attention
3. We should be able to use it with `python3 -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct --attention-backend cudnn`

---

## Issue #N/A: [Feature] integrate FlashInfer Blackwell kernels

**Link**: https://github.com/sgl-project/sglang/issues/5855
**State**: open
**Created**: 2025-04-28T19:12:30+00:00
**Comments**: 4
**Labels**: high priority, flashinfer, performance, blackwell

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

### Related resources

_No response_

---

## Issue #N/A: [Feature] support DeepSeek R1 FP4

**Link**: https://github.com/sgl-project/sglang/issues/5055
**State**: closed
**Created**: 2025-04-04T01:11:06+00:00
**Closed**: 2025-06-04T00:19:47+00:00
**Comments**: 4
**Labels**: high priority, inactive, performance, quant, deepseek

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled cc @Edwardf0t1 @kushanam @elfiegg 

Optimization is also important on Blackwell

### Related resources

_No response_

---

## Issue #N/A: [Feature] RFC for adding CPU support for SGLang

**Link**: https://github.com/sgl-project/sglang/issues/2807
**State**: open
**Created**: 2025-01-09T07:58:45+00:00
**Comments**: 13
**Labels**: enhancement, high priority, intel, cpu

### Description

### Motivation

Hi, SGLang folks! This is Mingfei from intel pytorch team, our team helps optimize PyTorch performance on CPU. I am also the PyTorch module maintainer for cpu performance. We would like to contribute to SGLang for CPU enabling and performance optimization.

### Targets
Our primary target is to optimize SGLang performance on Intel Xeon Scalable Processors (x86 server CPUs).
* Optimization will be focusing on Xeon with [Intel® Advanced Matrix Extensions](https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html) support, including Sapphire Rapids(4th gen), Emerald Rapids(5th gen), Granite Rapids(6th gen).
* Native implementations or fallbacks will be provided for CPUs with other ISA to make it functional.
* Providing good performance per dollar.

### Limitations

* Kernels written in **avx512** and **amx-bf16**, requires **GCC11** or above.
* **BFloat16/Float16** will be enabled at the same time on CPU, but we only 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] integrate MTP with some new features

**Link**: https://github.com/sgl-project/sglang/issues/7077
**State**: open
**Created**: 2025-06-11T03:33:03+00:00
**Comments**: 3
**Labels**: high priority, collaboration, deepseek, speculative-decoding

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

- [x] compatibility with dp attention #6081 
- [x] compatibility with eplb
- [x] compatibility with `enable-dp-lm-head`
- [x] compatibility with pd disaggregation @Atream #7242 
- [x] compatibility with two-batch-overlap @Qiaolin-Yu #7225 
- [x] compatibility with deepep #7206 
...

### Related resources

https://github.com/sgl-project/sglang/issues/6017
https://lmsys.org/blog/2025-05-05-large-scale-ep/#large-scale-expert-parallelism

---

## Issue #N/A: [Feature] (Willing to PR) Avoid KV cache occupying GPU memory when not used

**Link**: https://github.com/sgl-project/sglang/issues/2542
**State**: closed
**Created**: 2024-12-22T09:07:26+00:00
**Closed**: 2025-03-16T14:34:36+00:00
**Comments**: 43
**Labels**: high priority, collaboration, inactive, feature

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Hi thank you for the library! The use case is that, when doing online PPO, I hope to use SGLang to generate llm completions, and then use RL to do gradient descent on those completions.

The problem is, to do this on a single GPU, the timeline is "SGLang generate - Torch backward - repeat it". Thus, when torch doing backprop, I hope SGLang can free its KV cache memory consumption, otherwise torch will not have enough memory.

Thanks for any suggestions!

### Related resources

_No response_

---

