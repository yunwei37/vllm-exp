# release - issues

**Total Issues**: 20
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 20

### Label Distribution

- release: 20 issues
- stale: 1 issues

---

## Issue #N/A: v0.7.2 Release Tracker

**Link**: https://github.com/vllm-project/vllm/issues/12700
**State**: closed
**Created**: 2025-02-03T17:14:45+00:00
**Closed**: 2025-02-06T18:55:16+00:00
**Comments**: 12
**Labels**: release

### Description


We will make a new release as soon as these PRs are merged.

- [x] https://github.com/vllm-project/vllm/pull/12696
- [x] https://github.com/vllm-project/vllm/pull/12604
- [x] https://github.com/vllm-project/vllm/pull/12729
- [x] https://github.com/vllm-project/vllm/pull/12676
- [x] https://github.com/vllm-project/vllm/pull/12732
- [x] https://github.com/vllm-project/vllm/pull/12796

---

## Issue #N/A: Release v0.7.3

**Link**: https://github.com/vllm-project/vllm/issues/12465
**State**: closed
**Created**: 2025-01-27T05:29:49+00:00
**Closed**: 2025-02-20T06:45:24+00:00
**Comments**: 6
**Labels**: release

### Description

Update (02/03/2025):
* This has been renamed to v0.7.3 as we are releasing v0.7.2 for MLA bug fixes, transformers backend, and Qwen2.5VL

Update (01/31/2025):
* This has been renamed to v0.7.2 as we are releasing v0.7.1 for Deepseek enhancements. 


Blockers
- [ ] Support for Qwen-1M: https://github.com/vllm-project/vllm/pull/11844
- [ ] Support for Baichuan-M1: https://github.com/vllm-project/vllm/pull/12251

---

## Issue #N/A: Release v0.7.0

**Link**: https://github.com/vllm-project/vllm/issues/12365
**State**: closed
**Created**: 2025-01-23T18:17:20+00:00
**Closed**: 2025-02-11T18:29:43+00:00
**Comments**: 15
**Labels**: release

### Description


* Alpha release for vLLM V1 architecture, ETA 1/23-1/24
* Pending V1 items
  - [ ] Performance numbers @ywang96 @robertgshaw2-redhat @WoosukKwon 
  - [ ]  Documentation @WoosukKwon 
  - [ ]  Blog post @WoosukKwon 
* Other Pending PRs
  - [x] https://github.com/vllm-project/vllm/pull/12361
  - [x] https://github.com/vllm-project/vllm/pull/12243
  - [x] ~https://github.com/vllm-project/vllm/pull/12377~ #12380
  - [x] https://github.com/vllm-project/vllm/pull/12375
  - [x] https://github.com/vllm-project/vllm/pull/12405



---

## Issue #N/A: [Release]: v0.7.0 Release Tracker

**Link**: https://github.com/vllm-project/vllm/issues/11218
**State**: closed
**Created**: 2024-12-16T00:55:05+00:00
**Closed**: 2025-04-27T02:11:35+00:00
**Comments**: 8
**Labels**: release, stale

### Description

We plan a release after V1 is ready. 


~Planned release Dec 16-17.~

We do not block release on feature request, minor issues. We do block release on major feature development, important user impacting issues, and compatibility fixes. 

Add pending PRs directly here ⬇️ 

- [x] https://github.com/vllm-project/vllm/pull/10511
- [x] #11210 



---

## Issue #N/A: v0.6.1.post1 Release Tracker

**Link**: https://github.com/vllm-project/vllm/issues/8426
**State**: closed
**Created**: 2024-09-12T17:46:21+00:00
**Closed**: 2024-09-13T05:02:37+00:00
**Comments**: 6
**Labels**: release

### Description

### Anything you want to discuss about vllm.

- [x] #8390
- [x] #8375 
- [x] #8399 
- [x] #8417 
- [x] #8415 
- [x] #8376
- [x] #8425



---

## Issue #N/A: Release v0.6.0

**Link**: https://github.com/vllm-project/vllm/issues/8144
**State**: closed
**Created**: 2024-09-04T05:47:03+00:00
**Closed**: 2024-09-05T04:27:00+00:00
**Comments**: 6
**Labels**: release

### Description

### Anything you want to discuss about vllm.

Target Sept 04-06. 

This release will will bring to a close for majority of enhancements in #6801. So I'll mostly merge PRs that are performance sensitive. But feel free to comment the PR that you think should go in. As a reminder we do aim to release every 2 weeks. 

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: Release v0.5.5

**Link**: https://github.com/vllm-project/vllm/issues/7481
**State**: closed
**Created**: 2024-08-13T21:01:28+00:00
**Closed**: 2024-08-23T19:50:56+00:00
**Comments**: 14
**Labels**: release

### Description

We will make a release later this week or early next week (Aug 16-Aug19) to address Gemma logits soft-caps bug, openai server metrics bug, and include more performance enhancements. 

Please add blockers if needed. 

---

## Issue #N/A: v0.5.2, v0.5.3, v0.6.0 Release Tracker

**Link**: https://github.com/vllm-project/vllm/issues/6434
**State**: closed
**Created**: 2024-07-15T03:19:27+00:00
**Closed**: 2024-08-05T21:39:49+00:00
**Comments**: 11
**Labels**: release

### Description

### Anything you want to discuss about vllm.

We will make a triplet of releases in the following 3 weeks. 
- [x] v0.5.2 on Monday July 15th. 
- [x] v0.5.3 by Tuesday July 23rd.
- [ ] v0.6.0 after Monday July 29th.

Blockers
- [x] #6463
- [x] #6517
- [x] #6698
- [ ] Test vLLM works with 405B that's `num_kv_heads=8` instead of 16. 

~The reason for such pace is that we want to remove beam search (#6226), which unlocks a suite of scheduler refactoring to enhance performance (async scheduling to overlap scheduling and forward pass for example). We want to release v0.5.2 ASAP to issue warnings and uncover new signals. Then we will decide the removal in v0.6.0. Normally we will deprecate slowly by stretching it by one month or two. However, (1) RFC has been opened for a while (2) it is unfortunately on the critical path of refactoring and performance enhancements.~

Please also feel free to add release blockers. But do keep in mind that I will not slow the release for v0.5.* 

[... truncated for brevity ...]

---

## Issue #N/A: v0.5.1 Release Tracker

**Link**: https://github.com/vllm-project/vllm/issues/5806
**State**: closed
**Created**: 2024-06-25T00:18:07+00:00
**Closed**: 2024-07-06T04:55:52+00:00
**Comments**: 15
**Labels**: release

### Description

ETA Friday -> Wednesday 07/03

* https://github.com/vllm-project/vllm/pull/4115 
* https://github.com/vllm-project/vllm/pull/4650 
* https://github.com/vllm-project/vllm/pull/6051
* https://github.com/vllm-project/vllm/pull/6033 
* https://github.com/vllm-project/vllm/pull/5987
* https://github.com/vllm-project/vllm/pull/6044
* https://github.com/vllm-project/vllm/pull/4412
* https://github.com/vllm-project/vllm/pull/6050 
* #6055 <- https://github.com/vllm-project/vllm/pull/5276
- https://github.com/vllm-project/vllm/pull/6109

---

## Issue #N/A: v0.4.3 Release Tracker

**Link**: https://github.com/vllm-project/vllm/issues/4895
**State**: closed
**Created**: 2024-05-18T01:05:16+00:00
**Closed**: 2024-06-03T17:04:07+00:00
**Comments**: 16
**Labels**: release

### Description

ETA May 30 (due to some blockers and US holiday). 

Blockers
- [ ] #4650
- [ ] #4799
- [x] #4846

Nice to have
- [ ] #4638
- [x] #4525
- [ ] #4464

---

## Issue #N/A: v0.4.2 Release Tracker

**Link**: https://github.com/vllm-project/vllm/issues/4505
**State**: closed
**Created**: 2024-04-30T17:28:48+00:00
**Closed**: 2024-05-05T07:20:30+00:00
**Comments**: 12
**Labels**: release

### Description

ETA May 3rd, Friday. 



---

## Issue #N/A: v0.4.1 Release Tracker

**Link**: https://github.com/vllm-project/vllm/issues/4181
**State**: closed
**Created**: 2024-04-18T21:37:23+00:00
**Closed**: 2024-04-24T04:43:06+00:00
**Comments**: 9
**Labels**: release

### Description

ETA Monday April 22

- [x] #4176
- [x] #4182 (addressing #4180)
- [x] #4079 
- [x] #4159 
- [x] #4138
- [x] #4209
- [x] #4210
- [x] #4271
- [x] #4304 (otherwise we cannot upload to PyPI :(

---

## Issue #N/A: [v0.4.0] Release Tracker

**Link**: https://github.com/vllm-project/vllm/issues/3155
**State**: closed
**Created**: 2024-03-02T01:41:58+00:00
**Closed**: 2024-04-03T22:42:26+00:00
**Comments**: 11
**Labels**: release

### Description

**ETA:** Before Mar 28th

##  Major changes
TBD.

## PRs to be merged before the release
- [ ] #1507 
- [x] #2762 
- [ ] ...


---

## Issue #N/A: [v0.3.3] Release Tracker

**Link**: https://github.com/vllm-project/vllm/issues/3097
**State**: closed
**Created**: 2024-02-28T22:36:29+00:00
**Closed**: 2024-03-01T20:58:07+00:00
**Comments**: 5
**Labels**: release

### Description

**ETA**: Feb 29th - Mar 1st

## Major changes

* StarCoder2 support
* Performance optimization and LoRA support for Gemma
* Performance optimization for MoE kernel
* 2/3/8-bit GPTQ support
* [Experimental] AWS Inferentia2 support

## PRs to be merged before the release

- [x] #2330 #2223
- [ ] ~~#2761~~
- [x] #2819 
- [x] #3087 #3099
- [x] #3089 

---

## Issue #N/A: [v0.3.1] Release Tracker

**Link**: https://github.com/vllm-project/vllm/issues/2859
**State**: closed
**Created**: 2024-02-13T23:36:38+00:00
**Closed**: 2024-02-16T23:05:19+00:00
**Comments**: 12
**Labels**: release

### Description

**ETA**: Feb 14-16 th

## Major changes

TBD

## PRs to be merged before the release

- [x] #2855 
- [x] #2845 
- [x] ~~#2514~~
- [x] Ensure memory release when `LLM` class is deleted. #2882 
- [x] #2875 #2880

---

## Issue #N/A: [v0.2.7] Release Tracker

**Link**: https://github.com/vllm-project/vllm/issues/2332
**State**: closed
**Created**: 2024-01-03T17:55:09+00:00
**Closed**: 2024-01-04T01:35:58+00:00
**Comments**: 3
**Labels**: release

### Description

**ETA**: Jan 3rd - 4th

## Major changes

TBD

## PRs to be merged before the release

- [x] #2221 
- [ ] ~~#2293~~ (deferred)

---

## Issue #N/A: [v0.2.3] Release Tracker

**Link**: https://github.com/vllm-project/vllm/issues/1856
**State**: closed
**Created**: 2023-11-30T08:05:43+00:00
**Closed**: 2023-12-03T20:31:00+00:00
**Comments**: 6
**Labels**: release

### Description

**ETA**: Nov 30th - Dec 2nd.

## Major changes

* Refactoring on Worker, InputMetadata, and Attention
* Fix TP support for AWQ models
* Support Prometheus metrics
* Fix Baichuan & Baichuan 2

## PRs to be merged before the release

- [x] Chat Template #1756 
- [x] ~#1707~ (We have to solve AWQ perf first, which might be possible in time).
- [x] ~#1662~ (use the new one instead)
- [x] #1890
- [x] #1852 

---

## Issue #N/A: [v0.2.2] Release Tracker

**Link**: https://github.com/vllm-project/vllm/issues/1551
**State**: closed
**Created**: 2023-11-02T17:07:19+00:00
**Closed**: 2023-11-19T05:57:08+00:00
**Comments**: 7
**Labels**: release

### Description

**ETA**: ~~Nov 3rd (Fri) - Nov 6th (Mon).~~ Nov 17th (Fri) - 19th (Sun).

## Major changes

* Extensive refactoring for better tensor parallelism & quantization support
* Changes in scheduler: from 1D flattened input tensor to 2D tensor
* Bump up to PyTorch v2.1 + CUDA 12.1
* New models: Yi, ChatGLM, Phi
* Added LogitsProcessor API
* Preliminary support for SqueezeLLM

## PRs to be merged before the release

- [x] CUDA 12 #1527
- [x] Yarn #1264, #1161
- [x] Phi model #1664 
- ~~[ ] Support embedding inputs #1265~~ 


---

## Issue #N/A: [v0.2.1] Release Tracker

**Link**: https://github.com/vllm-project/vllm/issues/1346
**State**: closed
**Created**: 2023-10-13T17:17:40+00:00
**Closed**: 2023-10-16T19:58:59+00:00
**Comments**: 0
**Labels**: release

### Description

**ETA**: ~~Oct. 15th (Sun)~~ Oct 16th (Mon).

## Major changes

TBD

## PRs to be merged before the release

- [x] PagedAttention V2 #1348 
- [x] Support `echo` #1328 #959 
- [x] Fix `TORCH_CUDA_ARCH_LIST` err msg #1239
- ~~Support YaRN #1264 #1161~~ (Deferred)
- ~~Add `repetition_penalty` sampling parameter #866~~ (Deferred)


---

## Issue #N/A: [v0.2.0] Release Tracker

**Link**: https://github.com/vllm-project/vllm/issues/1089
**State**: closed
**Created**: 2023-09-18T21:18:03+00:00
**Closed**: 2023-09-28T22:30:39+00:00
**Comments**: 0
**Labels**: release

### Description

## Major changes

* Up to 60% performance improvement by optimizing de-tokenization and sampler
* Initial support for AWQ (performance not optimized)
* Support for RoPE scaling and LongChat
* Support for Mistral-7B

## PRs to be merged before the release

- [x] Vectorized sampler: #1048, #820 
- [x] LongChat: #555 
- [x] `TORCH_CUDA_ARCH_LIST` build option: #1074 
- [x] Support for Mistral-7B: #1196 
- [x] #1198  
- ~~[ ] FP32 RoPE kernel: #1061~~ (deferred to the next PR)

---

