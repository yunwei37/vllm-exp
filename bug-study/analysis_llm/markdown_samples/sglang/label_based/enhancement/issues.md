# enhancement - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 9
- Closed Issues: 21

### Label Distribution

- enhancement: 30 issues
- inactive: 13 issues
- high priority: 10 issues
- good first issue: 6 issues
- help wanted: 5 issues
- feature: 4 issues
- collaboration: 2 issues
- flashinfer: 2 issues
- router: 2 issues
- grammar-backend: 1 issues

---

## Issue #N/A: [Feature] Per-request random seed

**Link**: https://github.com/sgl-project/sglang/issues/1335
**State**: closed
**Created**: 2024-09-05T13:16:11+00:00
**Closed**: 2024-12-14T00:17:29+00:00
**Comments**: 8
**Labels**: enhancement, inactive

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

I believe there is an option for fixing the random seed for the backend, but I think there isn't a feature for per-request random seeds.

### Related resources

_No response_

---

## Issue #N/A: lora speed

**Link**: https://github.com/sgl-project/sglang/issues/2559
**State**: closed
**Created**: 2024-12-23T14:28:48+00:00
**Closed**: 2025-02-22T00:16:12+00:00
**Comments**: 2
**Labels**: enhancement, inactive

### Description

I measured the speed of starting multiple loras using sglang and vllm. Why is vllm faster than sglang? What acceleration method is sglang? I haven’t enabled it yet?
Graphics card 4090
sglang sever：
python -m sglang.launch_server --model-path /mnt/models/source/model/qwen2_5-7b-instruct/Qwen2___5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tp-size 1 \
  --mem-fraction-static 0.9 \
  --served-model-name "Qwen2.5-7B-Instruct" \
  --chunked-prefill-size 4096 \
  --disable-cuda-graph \
  --disable-radix-cache \
  --show-time-cost \
  --enable-torch-compile \
  --schedule-conservativeness 0.03 \
  --schedule-policy fcfs \
  --lora-paths lora0=“” lora_batch="" \
  --max-loras-per-batch 32 \
  --dtype bfloat16

vllm sever
python -m vllm.entrypoints.openai.api_server --model /mnt/models/source/model/qwen2_5-7b-instruct/Qwen2___5-7B-Instruct \
   --port 8899 \
   --served-model-name Qwen2.5-7B-Instruct \
   --enable-lora \
   --lora-moduleslora0=“” lora_batch=""

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Set outlines and xgrammar as addtional dependency

**Link**: https://github.com/sgl-project/sglang/issues/2549
**State**: closed
**Created**: 2024-12-23T02:35:28+00:00
**Closed**: 2025-02-22T00:16:13+00:00
**Comments**: 4
**Labels**: enhancement, inactive, grammar-backend

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

I am trying to integrate SGLang and vllm into OpenRLHF. For the grammar backend, could we set it as additional requirements, i.e. import it when we use it? Like:

```python

def __init__():
    if use_constrained_decoding:
        if grammar_backend == "xgrammar":
            import xgrammar
            xgrammar.function()
        if grammar_backend == "outlines":
            import outlines
            outlines.function()
```

This to avoid the version conflicts with vllm.

### Related resources

No such.

---

## Issue #N/A: LLM integration with normal programming patterns or, a high level sglang interface

**Link**: https://github.com/sgl-project/sglang/issues/39
**State**: closed
**Created**: 2024-01-18T14:14:29+00:00
**Closed**: 2024-07-25T06:31:58+00:00
**Comments**: 3
**Labels**: enhancement, inactive

### Description

I posted a similar issue in outlines, but here goes:  we're building something complex and I think it would be helpful to have a marvin-like library that supports normal programming patterns with LLM's but also gives control over generation. This  would provide high level pythonic abstractions like typed functions dynamically compiling grammars for return pydantic structs that would also allow you to drop down to customize generation either within or around these functions. This could    be like high level mypy typed boundaries around sglang programs.

[Marvin](https://github.com/PrefectHQ/marvin) and [funcchain](https://github.com/shroominic/funcchain) do the high level (sort of), but you give up control. Marvin relies on json and/or function calling and is constrained to OAI models, funcchain uses dynamically compiled  Lllamacpp grammar   as well. 

Analogy would be Pytorch:triton::funcchain/equivalent:sglang

Aside from the funcchain-like feature, for my use case I'd love to s

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support QLoRA weights

**Link**: https://github.com/sgl-project/sglang/issues/1826
**State**: closed
**Created**: 2024-10-28T07:39:48+00:00
**Closed**: 2025-01-01T00:18:43+00:00
**Comments**: 3
**Labels**: enhancement, inactive

### Description

Does sgl support qlora? Could you provide some instructions on how to use it?

---

## Issue #N/A: Beam Search Support 

**Link**: https://github.com/sgl-project/sglang/issues/353
**State**: closed
**Created**: 2024-04-08T06:30:44+00:00
**Closed**: 2024-07-25T06:33:08+00:00
**Comments**: 1
**Labels**: enhancement, inactive

### Description

There was support for a num_beams parameter before but now the sampling parameters has no such parameter. Can support for beam search be restored?

---

## Issue #N/A: [Feature] support more user-friendly MTP

**Link**: https://github.com/sgl-project/sglang/issues/5595
**State**: closed
**Created**: 2025-04-21T08:03:50+00:00
**Closed**: 2025-04-29T23:33:16+00:00
**Comments**: 3
**Labels**: enhancement, high priority

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

As we discussed offline, we need to support more user-friendly MTP. cc @merrymercy 

- [ ] best configuration for the default @zhyncs 
- [ ] user doesn't need to specify draft model separately @ispobock 

### Related resources

_No response_

---

## Issue #N/A: [Bug] sglang doesn't stop the generation when the request is canceled

**Link**: https://github.com/sgl-project/sglang/issues/3520
**State**: open
**Created**: 2025-02-12T09:17:16+00:00
**Comments**: 6
**Labels**: enhancement

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I have been experimenting with sglang. Amazing work there !

I have found that sgland continues to generate even if the request has been canceled by the client.

It would be great if it wasn't the case. 

### Reproduction

Start the server, send a request, cancel it.

### Environment

I was using 2 * H200 to serve R1 600B

---

## Issue #N/A: [Roadmap] Prefill and Decoding Disaggregation

**Link**: https://github.com/sgl-project/sglang/issues/4655
**State**: open
**Created**: 2025-03-21T19:26:55+00:00
**Comments**: 30
**Labels**: enhancement, high priority, collaboration

### Description

### Design: 

[SGLang PD Disaggregation (Open Source)](https://docs.google.com/document/d/1rQXJwKd5b9b1aOzLh98mnyMhBMhlxXA5ATZTHoQrwvc/edit?tab=t.0#heading=h.i3s2t1j0e1ik)

### Progress
- [x] Release initial code @ByronHsu  #4654
  - prefill and decode event loop, queue, and transfer interface
  - **transfer engine is faked** 
  - easy python load balancer
- [x] Mooncake integration @ShangmingCai   https://github.com/sgl-project/sglang/pulls?q=is%3Apr+mooncake+is%3Aopen
- [x] NIXL Integration @trevor-m #5477
- [x] PD + overlap schedule @ByronHsu 
- [x] PD + DP attention @ch-wan @ByronHsu 
- [x] PD + fault tolerance https://github.com/sgl-project/sglang/pull/6504 https://github.com/sgl-project/sglang/pull/6263
- [x] PD + spec decode https://github.com/sgl-project/sglang/pull/6507
- [x] PD + logprob https://github.com/sgl-project/sglang/pull/6558
- [x] PD + Structured Output https://github.com/sgl-project/sglang/pull/6560

- [x] PD + retract @Ying1123 https://github.com/sgl-project/sglan

[... truncated for brevity ...]

---

## Issue #N/A: Further Speed up FA3 Backend

**Link**: https://github.com/sgl-project/sglang/issues/5810
**State**: open
**Created**: 2025-04-28T04:57:56+00:00
**Comments**: 13
**Labels**: enhancement, good first issue, help wanted

### Description

We explored and discussed some ideas and we want to write it down for tracking, also welcome community developer to try out those unfinished

- [x] (Good first issue) Skip `len` operation, get it directly from forward batch: https://github.com/sgl-project/sglang/pull/5969 @lifuhuang 
- [ ] GQA head packing: https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_attn_interface.py#L658 Change it to True and run benchmark.
- [x] Split-KV. aka Flash Decoding: We already enabled it, it is indeed faster in lower batch and long context scenario. Benchmark will be attached.
- [ ] PDL: https://github.com/Dao-AILab/flash-attention/commit/000090d02f0398e9087a8823fc1f5242becfac99
- [x] (Won't do) Prepare Scheduler Metadata: https://github.com/Dao-AILab/flash-attention/commit/fa60e7cc97300b4b26721983df580a7da7a8ebea (From Tri Dao's note, it can only speed up 2us, we can keep an eye on this, not recommending adopting this)
- [ ] For Llama Models, we observed that Spec Decoding with Top 

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

## Issue #N/A: [Feature] (Willing to PR) Proposal: Drop-in fast replacement of `PreTrainedModel.generate`

**Link**: https://github.com/sgl-project/sglang/issues/2569
**State**: closed
**Created**: 2024-12-24T06:18:24+00:00
**Closed**: 2025-03-30T00:19:36+00:00
**Comments**: 9
**Labels**: enhancement, high priority, collaboration, inactive, feature, RLHF

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Hi thanks for the lib! Currently, a lot of code uses `model.generate()`, such as TRL's PPOTrainer, etc. If we can make a drop-in replacement of it using SGLang, then everyone can very easily speed up their code related to generation. For example, TRL's PPOTrainer, OpenRLHF's train_ppo.py (not the train_ppo_ray.py which is more for distributed training). IMHO there are many places this can be useful - many online RL algorithm can benefit from this.

As for when to update SGLang weight from HF weight, most naive solution may be, we update weights *every* time the generate is called. This may not be a big problem, because we can configure the PPO batch size to be so huge that the model.generate is only called

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Specify dtype at begin_forward for FlashInfer > 0.1.6

**Link**: https://github.com/sgl-project/sglang/issues/2313
**State**: closed
**Created**: 2024-12-02T10:45:21+00:00
**Closed**: 2024-12-08T12:07:32+00:00
**Comments**: 2
**Labels**: bug, enhancement

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled
fix https://github.com/sgl-project/sglang/pull/2295#issuecomment-2509684766

### Related resources

_No response_

---

## Issue #N/A: [Feature] several features for veRL integration

**Link**: https://github.com/sgl-project/sglang/issues/2736
**State**: closed
**Created**: 2025-01-05T15:59:49+00:00
**Closed**: 2025-03-08T00:14:04+00:00
**Comments**: 4
**Labels**: enhancement, inactive

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

TL;DR: Introducing several features that would be beneficial for integrating SGLang into veRL and may also be beneficial for other Post-Training frameworks.
### Provide an inference script that is started by torchrun (support SPMD)
Currently, the offline inference script is launched by `sgl.Engine`. Internally, it spawns multiple [`Scheduler`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L98).
With `torchrun`, the `Scheduler` is launched by `torchrun` and the tp_rank can be obtained from the environ.
In veRL, the Data Parallel dimension is managed by our `WorkerGroup` and the dp_rank of each Scheduler should be None.
More specifically, if the current `WorkerGro

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

## Issue #N/A: OpenAI speculative execution

**Link**: https://github.com/sgl-project/sglang/issues/44
**State**: closed
**Created**: 2024-01-18T18:09:31+00:00
**Closed**: 2024-01-25T10:10:02+00:00
**Comments**: 3
**Labels**: enhancement, high priority

### Description

The current frontend using OpenAI will invoke multiple calls for the example below:
```
@sgl.function
def example(s):
  s += "Construct a character."
  s += "Name: " + gen("name") + " Birthday: " + gen("birthday") + " Job: " + gen("job")
```
We can optimize this to send less number of calls to save money:
1. Gen longer in the first gen call, and skip the later if the first gen did the right thing.
2. Allow using OpenAI's n=10 keyword argument to sample multiple completions when forked. We can also provide the interface `example.run(n=10)`.

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

## Issue #N/A: [Feature] Support Beam Search

**Link**: https://github.com/sgl-project/sglang/issues/3032
**State**: closed
**Created**: 2025-01-21T11:46:57+00:00
**Closed**: 2025-03-23T00:19:16+00:00
**Comments**: 1
**Labels**: enhancement, inactive

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

cc @HandH1998 @sleepcoo @ispobock 

Beam search is a common method in LLM generation, supported by some LLM engines, e.g,. vLLM, Transformers. 

This issue proposes our implementation to support beam search in SGLang and discusses its rationality, similar to an RFC.

vLLM's beam search implementation was performant, but in a recent release, beam search support was dropped from the core (https://github.com/vllm-project/vllm/issues/6226) and became much slower. Our implementation aims to achieve minimal modifications and minimal overhead. We found that in vLLM's high-level implementation (https://github.com/vllm-project/vllm/blob/2fc6944c5e69d5d0ce15d09a855452c795d75c3c/vllm/entrypoints/llm.py#L507), each decoding it

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support AMD GPU via PyTorch for ROCm

**Link**: https://github.com/sgl-project/sglang/issues/1419
**State**: closed
**Created**: 2024-09-14T05:55:31+00:00
**Closed**: 2024-09-19T11:01:59+00:00
**Comments**: 1
**Labels**: enhancement

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Enable SGLang on AMD GPUs !

### Related resources

_No response_

---

## Issue #N/A: [Feature] SGLang Router design discussion

**Link**: https://github.com/sgl-project/sglang/issues/2389
**State**: closed
**Created**: 2024-12-07T11:53:13+00:00
**Closed**: 2025-04-06T00:19:37+00:00
**Comments**: 7
**Labels**: enhancement, inactive

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

@ispobock  and I had a brief discussion about the current design and implementation of the SGLang Router.

I think the main concerns currently are as follows, before large-scale deployment.

- The current Router is stateful, which means I cannot deploy the Router like scaling a stateless service.
May we consider storing the state of the Router in services like Redis, DB, or etcd here?

- The current Router is at the cluster level. Although there are replicas, when the master fails, a replica can be used.
Imagine a real deployment scenario, such as one used by actual customers, where the deployment requires simultaneous use of AWS, GCP, and Oracle. The data centers are distributed across the Western US, Cent

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Add Math in our CI

**Link**: https://github.com/sgl-project/sglang/issues/2504
**State**: closed
**Created**: 2024-12-17T22:37:36+00:00
**Closed**: 2024-12-30T06:52:10+00:00
**Comments**: 4
**Labels**: enhancement, good first issue

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

One of my friends told me that SGLang Engine's performance on Math is abnormally a bit lower. We will find this out, benchmarking SGLang and other engines' performance on Math (use GPT-4 to evaluate). And, ultimately, we will add a CI test for Math which runs daily.

### Related resources

No such.

---

## Issue #N/A: Task 000: Centralized Configuration Module

**Link**: https://github.com/sgl-project/sglang/issues/7533
**State**: closed
**Created**: 2025-06-25T20:09:47+00:00
**Closed**: 2025-06-27T22:42:03+00:00
**Comments**: 0
**Labels**: enhancement, router

### Description

# Task 000: Centralized Configuration Module

## Summary
Create a comprehensive configuration module that centralizes all validation logic, provides type-safe configuration structures, and eliminates scattered validation code throughout the router.

## Problem Statement
Currently, configuration validation is scattered across multiple locations:
- URL validation happens in Python code
- Mode compatibility checks occur during server startup
- Policy parameter validation is embedded in individual routers
- No centralized error handling for configuration issues
- Duplicate validation logic in different components

This leads to:
- Inconsistent validation rules
- Runtime errors that could be caught at startup
- Difficult maintenance when adding new configuration options
- Poor error messages that don't guide users to fixes

## Proposed Solution

### 1. Configuration Type System
Create strongly-typed configuration structures with built-in validation:

```rust
// src/config/types.rs
#[derive(

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

## Issue #N/A: [Feature] FusedMoE H200 tuning

**Link**: https://github.com/sgl-project/sglang/issues/2471
**State**: closed
**Created**: 2024-12-12T20:01:22+00:00
**Closed**: 2024-12-31T16:15:10+00:00
**Comments**: 0
**Labels**: enhancement

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

ref https://github.com/sgl-project/sglang/issues/2450
https://github.com/sgl-project/sglang/blob/main/benchmark/kernels/fused_moe_triton/README.md
DeepSeek V2, Mixtral 8x7B 8x22B, Qwen MoE etc

BTW Thanks @antferdom 

### Related resources

_No response_

---

## Issue #N/A: [Tracker] Blackwell support

**Link**: https://github.com/sgl-project/sglang/issues/5338
**State**: open
**Created**: 2025-04-13T04:35:37+00:00
**Comments**: 29
**Labels**: enhancement, blackwell

### Description

## Usage

```bash
docker pull lmsysorg/sglang:blackwell

# use latest main
cd /sgl-workspace/sglang && git pull
```

## Models

### DeepSeek V3 ✅
```bash
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code
```

### Llama 4 ✅
```bash
python3 -m sglang.launch_server --model meta-llama/Llama-4-Scout-17B-16E-Instruct --tp 8 --context-length 131072
```

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

## Issue #N/A: [Feature] Tool Call Roadmap

**Link**: https://github.com/sgl-project/sglang/issues/6589
**State**: open
**Created**: 2025-05-25T10:29:03+00:00
**Comments**: 6
**Labels**: enhancement, high priority, feature, function-calling

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

## Motivation

Add a list of issues need to resolve in tool call.

## Track for Tool Call Issues

### High Piority

Issues related to accuracy, consistency, and performance.

- [x] [Multiple Tool Call Support for MistralDetector and Qwen25Detector](https://github.com/sgl-project/sglang/issues/6589#issuecomment-2907987558)
#6597 

- [ ] [JSON Double Dumping Behavior](https://github.com/sgl-project/sglang/issues/6589#issuecomment-2907988051)

- [x] [`ToolCallItem.tool_index` not following OpenAI API](https://github.com/sgl-project/sglang/issues/6589#issuecomment-2907988438) 
#6715 
#6655 
#6678 

----

### Medium Priority

Issues that are not immediate, such as features still WIP, or needs refactor, or edge cases.

- [ ] [Tests for 

[... truncated for brevity ...]

---

