# never_closed_180days - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 30
- Closed Issues: 0

### Label Distribution

- good first issue: 26 issues
- help wanted: 15 issues
- high priority: 6 issues
- new-model: 4 issues
- enhancement: 4 issues
- inactive: 2 issues
- documentation: 2 issues
- RLHF: 2 issues
- lora: 2 issues
- bug: 2 issues

---

## Issue #N/A: [Feature] Jamba 1.5 Support PLS

**Link**: https://github.com/sgl-project/sglang/issues/1190
**State**: open
**Created**: 2024-08-23T09:49:47+00:00
**Comments**: 2
**Labels**: good first issue, new-model

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

First SOTA ssm based model, vllm currently supports it but there is some parallel work in vllm to optimise it aswell
- https://github.com/vllm-project/vllm/pull/7428
- https://github.com/vllm-project/vllm/pull/7651

https://huggingface.co/collections/ai21labs/jamba-15-66c44befa474a917fcf55251

### Related resources

vllm implementation
https://github.com/vllm-project/vllm/pull/4115

---

## Issue #N/A: [CI] Add accuracy test for multimodal models

**Link**: https://github.com/sgl-project/sglang/issues/2277
**State**: open
**Created**: 2024-11-30T07:35:49+00:00
**Comments**: 3
**Labels**: good first issue, help wanted

### Description

We want to add accuracy test for multimodal models, such as llama 3.2 and llava onevision.


## Steps
1. Learn the current multimodal model tests. https://github.com/sgl-project/sglang/blob/main/test/srt/test_vision_openai_server.py
2. Learn the current accuracy test for text models. https://github.com/sgl-project/sglang/blob/main/test/srt/test_eval_accuracy_large.py
3. Adapt the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) framework for accuracy tests.

---

## Issue #N/A: [Feature] Save cache from requests and load

**Link**: https://github.com/sgl-project/sglang/issues/1932
**State**: open
**Created**: 2024-11-06T02:56:51+00:00
**Comments**: 2
**Labels**: good first issue

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

This might be difficult to implement, but I am facing the following issue:
When running Qwen2-VL on bigger images, the preprocessor takes a long time to convert the images to tokens.

It would be awesome if we could have a way (OpenAI API with extra parameters) to tell the backend to store the cache of a request and load it by ID for another request, which would make it possible to not reprocess every image (and prompt in general) on each call.

If my problem could be solved in an easier way I would be thankful for any input :)

### Related resources

_No response_

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

## Issue #N/A: [Feature] support ngram

**Link**: https://github.com/sgl-project/sglang/issues/2681
**State**: open
**Created**: 2024-12-31T07:03:24+00:00
**Comments**: 4
**Labels**: enhancement, good first issue, help wanted

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

ref https://github.com/apoorvumang/prompt-lookup-decoding

### Related resources

_No response_

---

## Issue #N/A: [Feature] Add docs for pass in token ids directly

**Link**: https://github.com/sgl-project/sglang/issues/2661
**State**: open
**Created**: 2024-12-30T07:51:00+00:00
**Comments**: 10
**Labels**: documentation, good first issue, RLHF

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

In most of RLHF frameworks, the prompts are pre-tokenized when data processing, so they can directly pass in token ids to the sglang engine rather than the prompts. So we should add docs on how to do this and how to get tokens directly.

### Related resources

No such.

---

## Issue #N/A: [Feature] Support TRI-ML/prismatic-vlms

**Link**: https://github.com/sgl-project/sglang/issues/1129
**State**: open
**Created**: 2024-08-16T18:15:10+00:00
**Comments**: 2
**Labels**: good first issue, feature, new-model

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

I'm trying to speed up inference for new VLM models on huggingface: https://huggingface.co/TRI-ML/prismatic-vlms/tree/main. I'm wondering if there are additional documentation on how to adapt new models? 

### Related resources

The model I'm trying to adapt is detailed here: https://arxiv.org/pdf/2402.07865. 

---

## Issue #N/A: [Feature] Request to 8-bit Quantization of Attention with SageAttention

**Link**: https://github.com/sgl-project/sglang/issues/1763
**State**: open
**Created**: 2024-10-23T09:30:36+00:00
**Comments**: 5
**Labels**: good first issue

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

As https://github.com/thu-ml/SageAttention mentioned, the quantized 8-bit attention will improvement the speed of inference about 2x and more with the same accuracy, so shall we give it a try or do some verification?

### Related resources

github: https://github.com/thu-ml/SageAttention

---

## Issue #N/A: [Feature] support llm_bench

**Link**: https://github.com/sgl-project/sglang/issues/2400
**State**: open
**Created**: 2024-12-08T11:02:20+00:00
**Comments**: 3
**Labels**: good first issue, help wanted, backlog

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

use `locust` as a benchmark option
ref https://github.com/fw-ai/benchmark/tree/main/llm_bench

### Related resources

_No response_

---

## Issue #N/A: [Bug] Decode Throughput Inconsistency Between bench_serving and Engine Logs

**Link**: https://github.com/sgl-project/sglang/issues/3050
**State**: open
**Created**: 2025-01-22T11:32:09+00:00
**Comments**: 16
**Labels**: good first issue, help wanted

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hi, I encountered an inconsistency in decode throughput reporting. When benchmarking with the bench_serving script, the reported TPOT is **much lower** than the decode throughput logged by the engine. This gap is significant for **small models or high concurrency settings**.

### Reproduction

#### start the server
```
python -m sglang.lau

[... truncated for brevity ...]

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

## Issue #N/A: [CI] Print nightly evaluation results to GITHUB_STEP_SUMMARY

**Link**: https://github.com/sgl-project/sglang/issues/2275
**State**: open
**Created**: 2024-11-30T07:14:29+00:00
**Comments**: 2
**Labels**: good first issue, help wanted

### Description

We would like to to have a nicer summary page on the nightly evaluation results. The summary should include the scores and thresholds of all models in a markdown table.

An example run: https://github.com/sgl-project/sglang/actions/runs/12060420843

## Steps
1. Learn `GITHUB_STEP_SUMMARY` at https://github.blog/news-insights/product-news/supercharging-github-actions-with-job-summaries/
2. Print a summary of the [nightly eval](https://github.com/sgl-project/sglang/blob/main/.github/workflows/nightly-eval.yml) to `GITHUB_STEP_SUMMARY`

See also an example https://github.com/sgl-project/sglang/pull/2274 

---

## Issue #N/A: [Feature] Integration of TurboMind AWQ and GPTQ

**Link**: https://github.com/sgl-project/sglang/issues/2788
**State**: open
**Created**: 2025-01-08T08:37:01+00:00
**Comments**: 1
**Labels**: good first issue, help wanted

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

The AWQ and GPTQ of TurboMind should be among the best-performing open-source implementations currently available. We plan to integrate them into SGLang, and once the integration is complete, we can consider removing SGLang's dependency on vLLM's AWQ and GPTQ kernel.

During development, we can initially install the wheel https://github.com/InternLM/turbomind/releases/tag/v0.0.1 manually for verification and later add the TurboMind repo as a dependency in [sgl-kernel](https://github.com/sgl-project/sglang/tree/main/sgl-kernel).

ref
https://github.com/InternLM/turbomind

### Related resources

_No response_

---

## Issue #N/A: [Feature] Serving VLM VILA

**Link**: https://github.com/sgl-project/sglang/issues/2345
**State**: open
**Created**: 2024-12-04T07:44:26+00:00
**Comments**: 1
**Labels**: good first issue, help wanted

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Hello,

I want to deploy the VILA model for serving VILA1.5-3B-AWQ (https://github.com/NVlabs/VILA). Could you please guide me on how to get started? Are there any specific instructions or tools I should follow for setting up the serving environment?

### Related resources

_No response_

---

## Issue #N/A: [Feature]: Benchmarking H200

**Link**: https://github.com/sgl-project/sglang/issues/2450
**State**: open
**Created**: 2024-12-11T14:11:42+00:00
**Comments**: 6
**Labels**: good first issue, high priority

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

#  Research Questions

- Explore the tradeoffs of increasing the **number of chips** with more memory, H200, versus increasing the parallel inference **world size** when using less HBM GPUs, H100 (see [[Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)](https://arxiv.org/abs/2211.05102)). Reduce as much as possible **price/generation** at **scale.**
- How can we leverage H200 **extra HBM** for efficient KV cache management?  Test long context window.
- Measure the implications of faster GPU **memory bandwidth** while executing **parallel inference**.

# Models of Interest

- **Llama 3.3 70B**
- **Llama 3.1 405B**
- **DeepSeek Models:** Testing latest sglang `0.4` [data pa

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Regex stop condition

**Link**: https://github.com/sgl-project/sglang/issues/2007
**State**: open
**Created**: 2024-11-11T23:54:02+00:00
**Comments**: 1
**Labels**: good first issue

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Hi! This would be awesome, as it would address following problems:
- most custom stopping conditions can be expressed as a regex
- handling custom stopping in a streaming response does not work as quickly as a backend based stopping condition would

(at least in my testing, a NodeJS client (with breaking the AsyncGenerator for await loop) can not stop the SGLang streaming generation the same way it works with the official OpenAI API)

I hope this is easy to implement

### Related resources

_No response_

---

## Issue #N/A: [Feature] Lora Development Roadmap

**Link**: https://github.com/sgl-project/sglang/issues/2929
**State**: open
**Created**: 2025-01-16T21:30:56+00:00
**Comments**: 0
**Labels**: help wanted, lora

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Features 

- [x] triton kernel & benchmark #3161 @Fridge003 
- [x] accuracy alignment #2671 #3413 @Fridge003 
- [x] test cases enhancement #3414 #3652 #4492 #4925 @aoshen524 @jcbjcbjc
- [x] support multi-rank adaptors #4492 @jcbjcbjc
- [x] support tensor parallel #2931 #4274 @aoshen524 
- [ ] compatibility with radix attention #2880 @Sunt-ing @jcbjcbjc
- [x] compatibility with cuda graph #3282 #4115 @Qiaolin-Yu  @Beichen-Ma 
- [x] support phi4mm #6544 @lifuhuang 
- [ ] support lora for embedding layer #3438 @Beichen-Ma 
- [x] load/unload #7412 #7446 @lifuhuang  @Fridge003 
- [ ] optimizing speed #2372 #3323 #6961 @jcbjcbjc @Fridge003 @lifuhuang 
- [ ] unified paging (support lora with different ranks) #3647 @Sunt-ing @jcbjcbjc

[... truncated for brevity ...]

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

## Issue #N/A: [Feature] Cascade attention kernels 

**Link**: https://github.com/sgl-project/sglang/issues/1715
**State**: open
**Created**: 2024-10-19T16:30:29+00:00
**Comments**: 5
**Labels**: good first issue, high priority

### Description

We would like to integrate the [cascade attention kernel](https://flashinfer.ai/2024/02/02/cascade-inference.html) from flashinfer.

Code pointers:
- Attention backend in sglang: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashinfer_backend.py
- Usage of cascade: https://docs.flashinfer.ai/api/python/cascade.html


---

## Issue #N/A: [Bug] Why can't I use multi-lora adapter and radix attention together?

**Link**: https://github.com/sgl-project/sglang/issues/2880
**State**: open
**Created**: 2025-01-14T07:03:52+00:00
**Comments**: 5
**Labels**: bug, lora

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

Why can't I use multi-lora adapter and radix attention together?
If I have multi-lora adapters, why not just insert the ID of the LoRA adapter before the first token?

When using a multi-lora adapter, it is extremely slow because radix attention cannot be used.

### Reproduction

https://github.com/sgl-project/sglang/blob/v0.4.1.post5/p

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] disk cache io error when simultaneously loading lots of  sglang offline engine

**Link**: https://github.com/sgl-project/sglang/issues/2090
**State**: open
**Created**: 2024-11-19T08:46:49+00:00
**Comments**: 13
**Labels**: good first issue, help wanted

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

when I use slurm to launch 32 or 192 jobs for offline batch inference, which simultaneously load sgl.engine. I met the following error although I set disable_disk_cache=True. If I only run one job for this, it will not meet this error.


The error is as follows:

```Python
Traceback (most recent call last):
  File "/home/x

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Log input text instead of input_ids when using openai chat apis

**Link**: https://github.com/sgl-project/sglang/issues/1608
**State**: open
**Created**: 2024-10-08T11:16:46+00:00
**Comments**: 4
**Labels**: good first issue

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug


I checked the docker logs and tried to find the request text in the logs, but the logs showed text=None, but input_ids was returned. I want it to display the request text directly. What parameters should I add when starting it?

docker Logs：

 in=GenerateReqInput(text=None, input_ids=[151644, 8948, 198, 2610, 525, 264, 10950, 17847, 

[... truncated for brevity ...]

---

## Issue #N/A: [Kernel] Optimize triton decoding kernels for long context

**Link**: https://github.com/sgl-project/sglang/issues/2271
**State**: open
**Created**: 2024-11-30T06:04:27+00:00
**Comments**: 5
**Labels**: good first issue, help wanted, high priority, inactive

### Description

We noticed the current triton decoding kernel is very slow on long context. This is due to a missing flash decoding like optimization.

## Reproduce
We test the decoding speed with a context length of 200 and 2,000.

triton backend: The decoding speed drops from 147.64 token/s to 126.41 token/s
```
$ python3 -m sglang.bench_offline_throughput --model meta-llama/Llama-3.1-8B-Instruct --dataset-name random --num-prompt 1 --random-input 128 --random-output 2048 --random-range 1 --attention-backend triton

[2024-11-30 05:10:04 TP0] Decode batch. #running-req: 1, #token: 234, token usage: 0.00, gen throughput (token/s): 147.64, #queue-req: 0
... 
[2024-11-30 05:10:18 TP0] Decode batch. #running-req: 1, #token: 2154, token usage: 0.00, gen throughput (token/s): 126.41, #queue-req: 0
```

flashinfer backend: The decoding speed only drops from 144.17 token/s to 143.35 token/s
```
$ python3 -m sglang.bench_offline_throughput --model meta-llama/Llama-3.1-8B-Instruct --dataset-nam

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Do we have any plan for supporting Phi3V?

**Link**: https://github.com/sgl-project/sglang/issues/1108
**State**: open
**Created**: 2024-08-15T05:03:47+00:00
**Comments**: 2
**Labels**: good first issue, new-model

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Do we have any plan for supporting Phi3V?

### Related resources

_No response_

---

## Issue #N/A: [Feature] Support for rerank models

**Link**: https://github.com/sgl-project/sglang/issues/2109
**State**: open
**Created**: 2024-11-21T06:45:30+00:00
**Comments**: 7
**Labels**: good first issue

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

vLLM has completion but no rerank, infinity has no completion but rerank, therefore sglang should have rerank

### Related resources

_No response_

---

## Issue #N/A: [Feature] Support bitsandbytes in QWen2 VL

**Link**: https://github.com/sgl-project/sglang/issues/2729
**State**: open
**Created**: 2025-01-04T08:12:29+00:00
**Comments**: 2
**Labels**: good first issue, help wanted

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Support bitsandbytes in QWen2 VL

### Related resources

_No response_

---

## Issue #N/A: [Feature] Add arguments mapping between SGLang / vllm / trt-llm

**Link**: https://github.com/sgl-project/sglang/issues/2657
**State**: open
**Created**: 2024-12-30T07:23:00+00:00
**Comments**: 2
**Labels**: documentation, good first issue, help wanted, RLHF

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

This is what I need to do for integrating SGLang into OpenRLHF. OpenRLHF already supports vllm. We need to add sglang. I need to map the server and sampling parameters from vllm to sglang. I think this is a good issue for us to let our users switch smoothly between mainstream engines.

**I attached how I am doing right now. But it may be wrong.**

### Related resources

**The args Mapping from vllm to sglang**

These are the server parameters of vllm:

```python
pretrain,
noset_visible_devices=noset_visible_devices,
trust_remote_code=True,
tensor_parallel_size=tensor_parallel_size,
dtype="bfloat16",
seed=seed + i,
enable_prefix_caching=enable_prefix_caching,
enforce_eager=enforce_eager,
max_model_len

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] EOFError

**Link**: https://github.com/sgl-project/sglang/issues/2294
**State**: open
**Created**: 2024-12-01T09:15:30+00:00
**Comments**: 16
**Labels**: bug

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

```
[2024-12-01 09:13:28 TP0] Init torch distributed begin.
[2024-12-01 09:13:28 TP2] Init torch distributed begin.
[2024-12-01 09:13:29 TP7] Init torch distributed begin.
[2024-12-01 09:13:29 TP1] Init torch distributed begin.
[2024-12-01 09:13:29 TP6] Init torch distributed begin.
[2024-12-01 09:13:30 TP4] Init torch distributed be

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] support torchao for qwen2 models

**Link**: https://github.com/sgl-project/sglang/issues/2219
**State**: open
**Created**: 2024-11-27T09:19:01+00:00
**Comments**: 11
**Labels**: good first issue, help wanted

### Description

I used one A30 card, and used Qwen2-7B-Instruct, the speed with quantization seems no different

python3 -m sglang.bench_latency --model ../Qwen2-7B-Instruct --batch-size 1 --input-len 200 --output-len 100
Benchmark ...
Prefill. latency: 0.03508 s, throughput:   5700.84 token/s
Decode.  latency: 0.01952 s, throughput:     51.23 token/s
Decode.  latency: 0.01947 s, throughput:     51.37 token/s
Decode.  latency: 0.01939 s, throughput:     51.58 token/s
Decode.  latency: 0.01933 s, throughput:     51.74 token/s
Decode.  latency: 0.01928 s, throughput:     51.87 token/s
Decode.  median latency: 0.01924 s, median throughput:     51.98 token/s
Total. latency:  1.942 s, throughput:    154.52 token/s

python3 -m sglang.bench_latency --model ../Qwen2-7B-Instruct --batch-size 1 --input-len 200 --output-len 100 --enable-torch-compile
Benchmark ...
Prefill. latency: 0.03655 s, throughput:   5471.84 token/s
Decode.  latency: 0.01852 s, throughput:     54.00 token/s
Decode.  latenc

[... truncated for brevity ...]

---

## Issue #N/A: [willing to PR] Add Lookahead speculative decoding

**Link**: https://github.com/sgl-project/sglang/issues/2772
**State**: open
**Created**: 2025-01-07T08:38:49+00:00
**Comments**: 1
**Labels**: enhancement, good first issue

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

n-gram based speculative is very effective in retrieval augmented generation(RAG). The cost of generating draft tokens is relatively low compared to eagle and has a great potential for accelerating token generation in RAG. Ant group has proposed the Trie-based retrieval and verification mechanism. I want to adopt it to SGLang.

### Related resources

[Lookahead: An Inference Acceleration Framework for Large Language Model with Lossless Generation Accuracy](https://arxiv.org/abs/2312.12728)

---

