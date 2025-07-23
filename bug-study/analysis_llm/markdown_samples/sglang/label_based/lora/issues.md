# lora - issues

**Total Issues**: 24
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 9
- Closed Issues: 15

### Label Distribution

- lora: 24 issues
- bug: 7 issues
- inactive: 5 issues
- feature: 3 issues
- enhancement: 2 issues
- performance: 2 issues
- low-priority: 1 issues
- high priority: 1 issues
- help wanted: 1 issues

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

## Issue #N/A: [Refactor] Deprecate FlashInfer lora backend

**Link**: https://github.com/sgl-project/sglang/issues/7809
**State**: open
**Created**: 2025-07-06T19:05:41+00:00
**Comments**: 0
**Labels**: lora

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

The current FlashInfer lora backend has been in a deprecation mode for a while with various bugs and feature gaps. Based on offline discussion with @Fridge003 and @Ying1123 , we decided to deprecate FlashInfer backend in favor of the Triton backend.

This would largely simplify our codebase and potentially bring perf gains as we eliminate the overheads to accommodate the special requirements from Flashinfer (e.g. today we have to reshape qkv in a way that's quite specific to FlashInfer in LoRAAdapter for compatibility but then convert it back in Triton backend).

We should keep the abstraction for supporting multiple backends as in the long term, we should consider prioritize introducing cutlass/cuda backend. 

###

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Dynamic LoRA load does not handle modules with greater rank correctly

**Link**: https://github.com/sgl-project/sglang/issues/7808
**State**: closed
**Created**: 2025-07-06T18:59:18+00:00
**Closed**: 2025-07-14T09:28:24+00:00
**Comments**: 1
**Labels**: lora

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

Adding an issue to track a bug for the new dynamic lora support:

Currently the new dynamic lora support creates gpu buffer based on existing adapter max lora ranks, when a new adapter is loaded that has larger lora rank than the initial set, it might not correctly handle it (to be verified).

We need to add logic to reset existing buffers

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Division by zero error in the current Triton LoRA backend when lora_path is None

**Link**: https://github.com/sgl-project/sglang/issues/7765
**State**: closed
**Created**: 2025-07-04T06:15:16+00:00
**Closed**: 2025-07-06T08:25:54+00:00
**Comments**: 0
**Labels**: lora

### Description

### Describe the bug

Creating an issue to track a potential bug based on offline discussion with @Fridge003 

In the current LoRA implementation, we handle lora_path = None requests by calling LoRA backend kernel (_sgemm_lora_a_kernel) with lora rank being zero. However, the kernel does not seem to be designed to handle such case.

If I add `tl.device_print` to the triton kernel, I can observe division by zero:

![Image](https://github.com/user-attachments/assets/710f4abe-de0b-4852-b18c-a60645da0009)

Interestingly, during the limited test cases I tried, the division-by-zero line (`pid // num_pid_n`) returned `-1`, and it did not result in memory access error as I expected it would be, the program output also appears to be reasonable. 

### Reproduction

```
python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct  --disable-radix-cache --lora-paths lora=algoprog/fact-generation-llama-3.1-8b-instruct-lora --max-loras-per-batch 1

curl -s http://localhost:30000/gen

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Add server arg enable-lora to allow starting up with empty lora-paths

**Link**: https://github.com/sgl-project/sglang/issues/7463
**State**: open
**Created**: 2025-06-23T07:42:22+00:00
**Comments**: 3
**Labels**: lora

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Currently SGL implicitly uses --lora-paths to decide whether engine/server should be started with LoRA support enabled.

As we are going to support dynamic lora loading/unloading soon (#7446), the current implicit constraint is no longer reasonable as it should be perfectly legal for users to start a LoRA-enabled server without having to provide any lora paths, but instead load/unload adapters later as needed. 

### Related resources

_No response_

---

## Issue #N/A: [Feature] Graceful handling of non-existing lora_path in inference request

**Link**: https://github.com/sgl-project/sglang/issues/7447
**State**: closed
**Created**: 2025-06-22T21:36:01+00:00
**Closed**: 2025-07-03T03:59:17+00:00
**Comments**: 1
**Labels**: lora

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Creating an issue to track this TODO for myself (or anyone else who wants to help):

Currently when users call SGLang with a non-existing lora_path, SGLang server/engine would crash due to failed assertions in `prepare_lora_batch`. This is unideal as it imposes unnecessary burden for server owner to validate request params before they are passed to the SGLang backend.

Ideally, SGLang should have gracefully handled the exception and respond 4xx errors without crashing the server.

### Related resources

_No response_

---

## Issue #N/A: [Bug] LoRA buffer eviction does not correctly handle adapters with different target weights

**Link**: https://github.com/sgl-project/sglang/issues/7426
**State**: open
**Created**: 2025-06-21T18:12:07+00:00
**Comments**: 3
**Labels**: lora

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

In the [load_lora_weight_to_buffer ](https://github.com/sgl-project/sglang/blob/9edf6608c9d299e126cc65634ee368d2fc52b0ad/python/sglang/srt/lora/mem_pool.py#L164C9-L168C19) function, we zero out `A_buffer` when `uid == None` ([code reference](https://github.com/sgl-project/sglang/blob/9edf6608c9d299e126cc65634ee368d2fc52b0ad/python/sglang/s

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] test_lora.py bug

**Link**: https://github.com/sgl-project/sglang/issues/7062
**State**: closed
**Created**: 2025-06-10T18:10:20+00:00
**Closed**: 2025-06-28T04:28:35+00:00
**Comments**: 2
**Labels**: bug, lora

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

There is a prompt "AI is a field of computer science focused on" in `test/srt/models/lora/test_lora.py ` that can easily break CI, which might be caused some internal bug of lora.

We remove this prompt temporarily in #7061. It should be added back after this bug is fixed.

### Reproduction

Uncomment line 49 of `test_lora.py`

![Image](ht

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Customized mapping for LoRA weight names

**Link**: https://github.com/sgl-project/sglang/issues/6608
**State**: open
**Created**: 2025-05-26T04:08:39+00:00
**Comments**: 0
**Labels**: low-priority, lora

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

The current LoRA impl in SGL maps LoRA weight to modules by (layer index, op_type) tuple, where op_type operation looks like `qkv_proj`, `o_proj`, `gate_up`, etc. This works fine for most standard cases, however, there are some limitations:
1. For models where there are more than one attention stacks (e.g., VLM), there could be multiple modules with the same (layer index, op_type), e.g., one from vision tower, the other from the language model. Currently SGL cannot handle such cases correctly and would usually fail during loading due to incorrect mapping.
2. Users cannot enable/disable application of LoRA at module-level, e.g., if user only wants to apply LoRA at language model but not vision (common); or when user

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] QLoRA adapters: not working with sglang : random predictions from text with LoRA, model load error with QLoRA

**Link**: https://github.com/sgl-project/sglang/issues/6501
**State**: open
**Created**: 2025-05-21T13:00:00+00:00
**Comments**: 0
**Labels**: lora

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I am using Mistral v0.3 along with QLoRA adapters for a specific task. It works perfectly with vLLM.

But with sglang it creates random continuous prediction from the text till max tokens.

### Reproduction

model_path = "/AITraining/home/llms/mistral-7b-instruct-v0.3"

server_process, port = launch_server_cmd(f"""
python -m sglang.launch_

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Lora manager cannot correctly process a batch if lora_path=[None, lora1]

**Link**: https://github.com/sgl-project/sglang/issues/5928
**State**: closed
**Created**: 2025-04-30T18:06:25+00:00
**Closed**: 2025-05-01T02:42:43+00:00
**Comments**: 1
**Labels**: bug, lora

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

 If lora_path=[None, lora1], it will throw an error
```
File "/sgl-workspace/sglang/python/sglang/srt/lora/lora_manager.py", line 220, in prepare_lora_batch
    lora = self.loras[lora_path]
KeyError: None
```

The issue is from this commit (https://github.com/sgl-project/sglang/commit/ef9a378a209d970e0b5c48ae3eac6f2660d43faf#diff-830eb84f0

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] HF_Runner can't produce correct results after applying lora

**Link**: https://github.com/sgl-project/sglang/issues/5897
**State**: closed
**Created**: 2025-04-30T00:10:58+00:00
**Closed**: 2025-04-30T03:17:43+00:00
**Comments**: 2
**Labels**: bug, lora

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

First batch input lora_path as [a, a]. Second batch input lora_path as [None, None]. The second batch will be processed as if you had input lora_path as [a, a].

### Reproduction

```
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] LoRA adapter can't process successfully when LoRA Path is None

**Link**: https://github.com/sgl-project/sglang/issues/4739
**State**: closed
**Created**: 2025-03-25T01:15:36+00:00
**Closed**: 2025-03-28T04:03:09+00:00
**Comments**: 1
**Labels**: bug, lora

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

Currently, LoRA in SGLang cannot handle `none` lora_path properly.

### Reproduction

Some cases below will have issues.
1. First batch input `lora_path` as `[a, a]`. Second batch input `lora_path` as `[None, None]`. The second batch will be processed as if you had input lora_path as `[a, a]`.
2. If you input `lora_path` as [a, None], the 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support unified paging in multi-lora serving

**Link**: https://github.com/sgl-project/sglang/issues/3647
**State**: open
**Created**: 2025-02-17T19:14:47+00:00
**Comments**: 5
**Labels**: enhancement, inactive, feature, lora

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Currently, SGL doesn't support the unified paging feature proposed by S-LoRA. However, this feature is important for memory management in multi-LoRA serving.

### Related resources

_No response_

---

## Issue #N/A: [Feature] Support Lora for VocabParallelEmbedding layer

**Link**: https://github.com/sgl-project/sglang/issues/3438
**State**: closed
**Created**: 2025-02-09T19:13:40+00:00
**Closed**: 2025-06-30T00:21:14+00:00
**Comments**: 2
**Labels**: inactive, feature, lora

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Support lora for `VocabParallelEmbedding`. Not a trivial task.

### Related resources

_No response_

---

## Issue #N/A: [Feature] Test case enhancement for Lora features

**Link**: https://github.com/sgl-project/sglang/issues/3414
**State**: closed
**Created**: 2025-02-09T00:41:02+00:00
**Closed**: 2025-04-17T21:40:34+00:00
**Comments**: 0
**Labels**: enhancement, lora

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Currently there is only two test files for Lora features:
`test/srt/models/test_lora.py` and 
`test/srt/models/test_lora_backend.py`
These two tests are only tested on llama models, thus not comprehensive.

Lora needs a series of well-organized tests, which can be similar to 
`test/srt/models/test_generation_models.py`


### Check List
check list copied from #3652:
- [x] Add backend test support for single adaptor, single prompt inference.
- [ ] Add backend test support for single adaptor, batch prompts serving.
- [ ] Add backend test support for multi-adaptor, same rank.
- [ ] Add backend test support for multi-adaptor, different rank.
- [ ] Add backend test support for adaptor with Embedding and Lm_head layer wei

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

## Issue #N/A: [Feature] Support compatibility between Cuda Graph and Lora

**Link**: https://github.com/sgl-project/sglang/issues/3282
**State**: closed
**Created**: 2025-02-04T06:27:19+00:00
**Closed**: 2025-04-29T06:30:45+00:00
**Comments**: 2
**Labels**: inactive, feature, lora

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Currently Lora and Cuda Graph cannot be used at the same time. 

Theoretically they should be compatible.

### Related resources

_No response_

---

## Issue #N/A: [Bug] tensor_model_parallel_all_reduce' is not defined

**Link**: https://github.com/sgl-project/sglang/issues/2931
**State**: closed
**Created**: 2025-01-17T02:02:26+00:00
**Closed**: 2025-03-19T05:22:34+00:00
**Comments**: 11
**Labels**: bug, lora

### Description

### Describe the bug

I attempted to serve the Phi-4 Lora Fine-tuning model by setting tensor parallel size 2 using the sglang framework, but the following error occurred.

> [Error Log]

```
[2025-01-17 01:51:55 TP0] LoRA manager ready.
[2025-01-17 01:51:57 TP1] Load weight end. type=Phi3ForCausalLM, dtype=torch.float16, avail mem=15.70 GB
[2025-01-17 01:52:00 TP1] LoRA manager ready.
[2025-01-17 01:52:00 TP0] Memory pool end. avail mem=39.54 GB
[2025-01-17 01:52:02 TP1] Memory pool end. avail mem=13.43 GB
[2025-01-17 01:52:02 TP1] max_total_num_tokens=16384, max_prefill_tokens=16384, max_running_requests=2049, context_len=16384
[2025-01-17 01:52:02 TP0] max_total_num_tokens=16384, max_prefill_tokens=16384, max_running_requests=2049, context_len=16384
[2025-01-17 01:52:02] INFO:     Started server process [649817]
[2025-01-17 01:52:02] INFO:     Waiting for application startup.
[2025-01-17 01:52:02] INFO:     Application startup complete.
[2025-01-17 01:52:02] INFO:     Uvicorn runnin

[... truncated for brevity ...]

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

## Issue #N/A: [Feature] Dynamic Lora Support in SGLang

**Link**: https://github.com/sgl-project/sglang/issues/2686
**State**: closed
**Created**: 2024-12-31T12:20:29+00:00
**Closed**: 2025-06-28T04:36:06+00:00
**Comments**: 15
**Labels**: lora

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

In SGLang, I would to dynamically apply domain-specific Lora adapters to smaller/local models. Normally, I use SGLang for inference. Recently, I've switched to Vllm which already has the ability to unload/load adaptors: https://docs.vllm.ai/en/latest/usage/lora.html
If this feature is already exists in SGlang, can you add an example in the documentation?


### Related resources

https://docs.vllm.ai/en/latest/usage/lora.html

---

## Issue #N/A: [Bug] HuggingFace and SGLang inference don't match

**Link**: https://github.com/sgl-project/sglang/issues/2671
**State**: closed
**Created**: 2024-12-30T22:54:09+00:00
**Closed**: 2025-05-03T00:18:08+00:00
**Comments**: 9
**Labels**: bug, inactive, lora

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

The accuracy of the model is degraded due to inconsistent outputs from SGLang. While HF and vLLM produce consistent results such as "A" or "B," SGLang occasionally outputs responses like "I can't process that request." or "A." / "B." This inconsistency impacts overall accuracy.

### Reproduction

What command or script did yo

[... truncated for brevity ...]

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

