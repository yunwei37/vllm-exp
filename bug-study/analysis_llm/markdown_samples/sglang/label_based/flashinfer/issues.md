# flashinfer - issues

**Total Issues**: 15
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 14

### Label Distribution

- flashinfer: 15 issues
- high priority: 8 issues
- bug: 5 issues
- performance: 3 issues
- inactive: 3 issues
- enhancement: 3 issues
- deepseek: 2 issues
- blackwell: 1 issues
- collaboration: 1 issues
- MLLM: 1 issues

---

## Issue #N/A: [Bug] FMHA using flashinfer cutlass on Blackwell has low accuracy result

**Link**: https://github.com/sgl-project/sglang/issues/6906
**State**: closed
**Created**: 2025-06-05T22:15:07+00:00
**Closed**: 2025-06-06T19:57:51+00:00
**Comments**: 0
**Labels**: bug, high priority, flashinfer

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When setting BatchPrefillWithRaggedKVCacheWrapper backend to "cutlass" in flashinfer backend, the test result for Llama-3.1-8B-Instruct is low:
```
Accuracy: 0.018
Invalid: 0.110
Latency: 45.625 s
Output throughput: 12136.862 token/s
```
Triton backend result with the same test:
```
Accuracy: 0.788
Invalid: 0.001
Latency: 16.626 s
Output t

[... truncated for brevity ...]

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

## Issue #N/A: [Feature] attention backend default choice

**Link**: https://github.com/sgl-project/sglang/issues/5064
**State**: closed
**Created**: 2025-04-04T08:13:51+00:00
**Closed**: 2025-05-21T09:29:52+00:00
**Comments**: 2
**Labels**: high priority, collaboration, flashinfer, performance, MLLM, deepseek

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

The standards we choose prioritize **performance first**, ease of use second (such as interface and installation), while also considering compatibility (such as older arch). Therefore, if in the future, the performance of different backends changes, we will still choose **the best performing one**.

1. NVIDIA

```
sm75 -> Triton
sm80, sm86, sm89 -> FlashInfer
sm90 -> FA3 (Llama, Qwen, Gemma), FlashInfer (Others)
sm100 -> FlashInfer

MLA
sm90 -> FA3 (DeepSeek)
sm100 -> FlashInfer (DeepSeek)

Other options
FlashMLA, cuDNN etc
```

SGLang will install the JIT version of FlashInfer on PyPI for a better user installation experience. Alternatively, the whl size limit of FlashInfer can be increased on PyPI. cc @yzh119 

F

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Remove stream sync in fast decode plan of flashinfer mla backend

**Link**: https://github.com/sgl-project/sglang/issues/4905
**State**: closed
**Created**: 2025-03-29T23:34:44+00:00
**Closed**: 2025-04-30T03:06:05+00:00
**Comments**: 1
**Labels**: bug, flashinfer

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

https://github.com/flashinfer-ai/flashinfer/pull/969 claims that the flashinfer mla backend can be sped up after removal of 
```python
  with self.device as device:
      stream = torch.cuda.current_stream(device).cuda_stream
```
in `fast_mla_decode_plan` of `flashinfer_mla_backend.py`

We need to test its performance after removal.

### R

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Behavior difference in `use_ragged` between versions 0.4.2.post3 and 0.4.4 for Gemma2-9b-it model

**Link**: https://github.com/sgl-project/sglang/issues/4406
**State**: closed
**Created**: 2025-03-14T01:42:11+00:00
**Closed**: 2025-05-14T00:19:06+00:00
**Comments**: 2
**Labels**: inactive, flashinfer

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hi, I recently updated the `sglang` package to version 0.4.4 and noticed that the google/gemma-2-9b-it model is not working as expected. After comparing it with the older version (0.4.2.post3), I found a difference in the `use_ragged` flag in the following code snippet in flashinfer_backend.py:

```python
if self.is_multimodal:
    use_rag

[... truncated for brevity ...]

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

## Issue #N/A: [Track] long context performance sglang vs vllm

**Link**: https://github.com/sgl-project/sglang/issues/3471
**State**: closed
**Created**: 2025-02-10T14:11:02+00:00
**Closed**: 2025-05-26T16:54:51+00:00
**Comments**: 4
**Labels**: high priority, flashinfer, performance

### Description

Currently, the two most popular practical scenarios for LLM are chatbot-like scenario or code completion scenario. SGLang has shown good performance on the ShareGPT dataset in the past. With the increasing popularity of open source models like Qwen2.5-Coder-7B-Instruct with a context of 128k, some potential users, such as hot startups, are interested in customizing SGLang for their own use cases, especially when dealing with long contexts in code scenario. The following is a simple performance benchmark aimed at providing insights into the current capabilities of open source LLM engine rather than comparing them directly. This will help guide future optimization efforts effectively. The following content will be regularly updated.

Performance: SGLang (chunked prefill 32k) > vLLM default > SGLang default (chunked prefill 8k) > vLLM enable chunked prefill (2k)
Hardware: H200
Version: SGLang v0.4.2.post4, vLLM 0.7.2

```bash
python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen

[... truncated for brevity ...]

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

## Issue #N/A: [Bug] Loading model meta-llama/Llama-3.3-70B-Instruct with flashinfer-0.2.0(+) raises an error

**Link**: https://github.com/sgl-project/sglang/issues/2577
**State**: closed
**Created**: 2024-12-25T20:44:40+00:00
**Closed**: 2024-12-28T06:08:56+00:00
**Comments**: 6
**Labels**: bug, flashinfer

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

There is an issue I have checked with @zhaochenyang20 

flash infer 0.2.0 + torch 2.4.0 + cuda 12.5 -> llama 3.3 70B will cause error:

  File "/home/***/miniconda3/envs/sglang/lib/python3.10/site-packages/flashinfer/prefill.py", line 2330, in forward_return_lse
    return self.run_return_lse(q, k, v)
  File "/home/***/mini

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

## Issue #N/A: [Bug] T4 not work

**Link**: https://github.com/sgl-project/sglang/issues/1058
**State**: closed
**Created**: 2024-08-12T14:46:18+00:00
**Closed**: 2024-08-28T13:05:35+00:00
**Comments**: 8
**Labels**: bug, flashinfer

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.

### Describe the bug

T4 not work w/o FlashInfer ref https://github.com/flashinfer-ai/flashinfer/issues/421

```
CUDA Error: no kernel image is available for execution on the device (209) /tmp/build-via-sdist-iemil769/flashinfer-0.1.4+cu121torch2.4/include/flashinfer/attention/handler.cuh: line 169 at function cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, num_threads, smem_size)
CUDA Er

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] requires "python-multipart" to be installed with docker image

**Link**: https://github.com/sgl-project/sglang/issues/949
**State**: closed
**Created**: 2024-08-06T08:03:23+00:00
**Closed**: 2024-08-16T08:09:06+00:00
**Comments**: 18
**Labels**: await-response, flashinfer

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

I tested sglang in Kubernetes with the minimum configuration below:
```
  containers:
  - args:
    - --model-path
    - /workspace/models/models--facebook--opt-125m
    - --served-model-name
    - opt-125m
    - --host
    - 0.0.0.0
    - --port
    - "8080"
    command:
    - python3
    - -m
    - sglang.launch_server
    image: lmsysorg/sglang:v0.2.9-cu121
```

However, it emits error like:
```
Form data requires "python-multipart" to be installed.
You can install "python-multipart" with:

pip install python-multipart

Traceback (most re

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] FlashInfer support for <=sm_75

**Link**: https://github.com/sgl-project/sglang/issues/931
**State**: closed
**Created**: 2024-08-05T09:14:49+00:00
**Closed**: 2024-09-22T14:25:24+00:00
**Comments**: 6
**Labels**: flashinfer

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

Can't use sglang with flashinfer if you have sm_75 or lower. Not even recompiling. Better put up this information so people don't waste time trying to make it work.

### Reproduction

simply trying to use it without `--disable-flashinfer --disable-flashinfer-sampling` causes a crash

### Environment

```Shell
Python: 3.11.9 (main, Apr 19 2024, 16:48:06) [GCC 11.2.0]
CUDA available: True
GPU 0: Tesla T4
CUDA_HOME: /usr/local/cuda-12.6
NVCC: Cuda compilation tools, release 12.6, V12.6.20
CUDA Driver Version: 535.54.03
PyTorch: 2.3.1+cu121
sglang: 0.2.10
flashinfer: 0.1

[... truncated for brevity ...]

---

## Issue #N/A: There doesn't seem to be a wheel available for python312.

**Link**: https://github.com/sgl-project/sglang/issues/849
**State**: closed
**Created**: 2024-07-31T09:35:12+00:00
**Closed**: 2024-08-01T23:30:03+00:00
**Comments**: 7
**Labels**: flashinfer

### Description

(llm_venv_sglang) xlab@xlab:/mnt/SGLang$ uv pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/
  × No solution found when resolving dependencies:
  ╰─▶ Because only the following versions of flashinfer are available:
          flashinfer==0.0.4+cu121torch2.3
          flashinfer==0.0.5+cu121torch2.3
          flashinfer==0.0.6+cu121torch2.3
          flashinfer==0.0.7+cu121torch2.3
          flashinfer==0.0.8+cu121torch2.3
          flashinfer==0.0.9+cu121torch2.3
          flashinfer==0.1.0+cu121torch2.3
          flashinfer==0.1.1+cu121torch2.3
          flashinfer==0.1.2+cu121torch2.3
      and flashinfer==0.0.4+cu121torch2.3 has no wheels with a matching Python ABI tag, we can conclude that
      flashinfer<0.0.5+cu121torch2.3 cannot be used.
      And because flashinfer==0.0.5+cu121torch2.3 has no wheels with a matching Python ABI tag, we can conclude
      that flashinfer<0.0.6+cu121torch2.3 cannot be used.
      And because flashinfer==0.0.6+cu12

[... truncated for brevity ...]

---

