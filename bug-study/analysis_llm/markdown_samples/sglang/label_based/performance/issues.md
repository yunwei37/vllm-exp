# performance - issues

**Total Issues**: 20
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 19

### Label Distribution

- performance: 20 issues
- high priority: 16 issues
- inactive: 6 issues
- deepseek: 4 issues
- quant: 4 issues
- good first issue: 4 issues
- flashinfer: 3 issues
- collaboration: 3 issues
- help wanted: 2 issues
- lora: 2 issues

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

## Issue #N/A: [Feature] VLM performance optimization

**Link**: https://github.com/sgl-project/sglang/issues/4805
**State**: closed
**Created**: 2025-03-27T05:17:30+00:00
**Closed**: 2025-05-27T00:18:51+00:00
**Comments**: 2
**Labels**: high priority, inactive, performance

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled @mickqian @yizhang2077 

### Related resources

_No response_

---

## Issue #N/A: [Feature] speedup DeepGEMM JIT compilation

**Link**: https://github.com/sgl-project/sglang/issues/4773
**State**: closed
**Created**: 2025-03-25T23:34:59+00:00
**Closed**: 2025-05-21T09:30:26+00:00
**Comments**: 0
**Labels**: high priority, performance

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

https://github.com/sgl-project/sglang/pull/4199
https://github.com/sgl-project/sglang/pull/4640

Before fixing #4640, I didn't realize how slow the DeepGEMM JIT compilation was.

```bash
python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319
```

Every time the test is run, the server is newly started. We can see that the first time, due to DeepGEMM JIT compilation, gsm8k only runs at 606 token/s.

```
➜  sglang git:(main) python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319
100%|████████████████████████████████████████████████████| 1319/1319 [03:37<00:00,  6.06it/s]
Accuracy: 0.954
Invalid: 0.000
Latency: 223.933 s
Output throughp

[... truncated for brevity ...]

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

## Issue #N/A: [Feature] Optimizing DeepSeek with the DeepSeek Infra OSS component

**Link**: https://github.com/sgl-project/sglang/issues/3758
**State**: closed
**Created**: 2025-02-21T11:52:28+00:00
**Closed**: 2025-03-10T18:28:27+00:00
**Comments**: 4
**Labels**: high priority, performance, deepseek

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

ref https://github.com/deepseek-ai/open-infra-index

- [ ] https://github.com/deepseek-ai/DeepEP
- [ ] https://github.com/deepseek-ai/DeepGEMM

### Related resources

_No response_

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

## Issue #N/A: [Feature] optimize group gemm

**Link**: https://github.com/sgl-project/sglang/issues/3323
**State**: closed
**Created**: 2025-02-05T22:56:43+00:00
**Closed**: 2025-02-20T08:26:59+00:00
**Comments**: 1
**Labels**: high priority, performance, lora

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Rewrite the  Grouped GEMM used by LoRA with cuBLAS 12.5 in sgl-kernel for improved speed.

https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/
https://github.com/zhihu/ZhiLight/blob/main/src/nn/linear/gemm_grouped.cpp

### Related resources

_No response_

---

## Issue #N/A: [Feature] adapt fused sigmoid gate for MoE model

**Link**: https://github.com/sgl-project/sglang/issues/2739
**State**: closed
**Created**: 2025-01-05T16:55:21+00:00
**Closed**: 2025-05-25T23:52:20+00:00
**Comments**: 20
**Labels**: good first issue, performance

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

ref https://github.com/NVIDIA/TensorRT-LLM/blob/be1788106245496872d18e702978e59b6bfd50e0/cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu#L232

### Related resources

_No response_

---

## Issue #N/A: [Feature] optimize moe_align_block_size_kernel

**Link**: https://github.com/sgl-project/sglang/issues/2732
**State**: closed
**Created**: 2025-01-05T05:56:21+00:00
**Closed**: 2025-03-25T04:11:57+00:00
**Comments**: 7
**Labels**: good first issue, high priority, wip, performance

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

The original version performs poorly and needs optimization. I suggest rewriting a new implementation.

https://github.com/sgl-project/sglang/blob/main/sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu

### Related resources

_No response_

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

## Issue #N/A: [Feature] lora serving performance 

**Link**: https://github.com/sgl-project/sglang/issues/2372
**State**: closed
**Created**: 2024-12-06T08:22:03+00:00
**Closed**: 2025-04-30T00:18:53+00:00
**Comments**: 3
**Labels**: inactive, performance, lora

### Description

lora reasoning speed is very slow, I ran a gemma's lora, found that qkv proj takes 0.0003s, but without lora only 0.0001s, so the result is a token decode time difference of 20ms+

however, vllm lora serving is faster

---

## Issue #N/A: AWQ performance tracking

**Link**: https://github.com/sgl-project/sglang/issues/1505
**State**: closed
**Created**: 2024-09-24T14:33:27+00:00
**Closed**: 2024-11-24T01:20:38+00:00
**Comments**: 2
**Labels**: inactive, performance

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

# Current Situation

## SGLang

```bash
# v0.3.1.post3
pip install --upgrade pip
pip install "sglang[all]"

pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

```
python3 -m sglang.launch_server --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --disable-radix

python3 bench_serving.py --backend sglang --num-prompts 5000
```

```
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Successful requests:                     5000
Benchmark duration (s):                  161.16
Total input tokens:                      1130466
Total generated tokens:                  971613


[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Performance issue on MoE with torch.compile

**Link**: https://github.com/sgl-project/sglang/issues/1446
**State**: closed
**Created**: 2024-09-17T10:08:42+00:00
**Closed**: 2024-09-23T16:54:18+00:00
**Comments**: 1
**Labels**: performance

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

We have implemented the MoE with native PyTorch and used torch.compile to accelerate it (https://github.com/sgl-project/sglang/commit/5574cc8b93d0b0f5b5ba697bc146fa672d3e4945). However, the performance is poor even with batch size = 1.
cc: @merrymercy 

### Reproduction

```bash
# clone code and install dependencies
git cl

[... truncated for brevity ...]

---

## Issue #N/A: TTFT latency for long context (16K) is very high around 15 seconds for llama3.1 70b model. (same or worse than vLLM)

**Link**: https://github.com/sgl-project/sglang/issues/922
**State**: closed
**Created**: 2024-08-04T23:14:23+00:00
**Closed**: 2024-10-09T01:10:58+00:00
**Comments**: 12
**Labels**: high priority, inactive, performance

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

I am experimenting with SGLang and vLLM for long context(16K) RAG application which requires real time responses.
I am using single Nvidia A6000 48GB GPU and llaam3.1 70b awq 4 bit model.

Currently I am seeing Time for first token latency is around 15 seconds which is very high.
Experimented with parameters like --chunked-prefill-size , --mem-frac etc

can you please suggest what are the parameters I need to mainly focus on to get the optimal TTFT for long context ?

### Reproduction

na

### Environment

```Shell
na
```


---

