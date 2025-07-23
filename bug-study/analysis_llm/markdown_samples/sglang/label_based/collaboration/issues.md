# collaboration - issues

**Total Issues**: 28
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 15
- Closed Issues: 13

### Label Distribution

- collaboration: 28 issues
- high priority: 20 issues
- inactive: 6 issues
- deepseek: 5 issues
- enhancement: 4 issues
- help wanted: 4 issues
- performance: 3 issues
- feature: 3 issues
- speculative-decoding: 2 issues
- good first issue: 2 issues

---

## Issue #N/A: [RFC] DRI for every module

**Link**: https://github.com/sgl-project/sglang/issues/7851
**State**: open
**Created**: 2025-07-08T08:14:57+00:00
**Comments**: 0
**Labels**: high priority, collaboration

### Description

SGLang now has a wide range of features and modules, with many efforts happening in parallel across new features, optimizations, and bug fixes. Each module already has an internal DRI (Directly Responsible Individual), but these assignments haven’t been made public. As a result, some community pull requests have experienced delays, and contributors often don’t know who to reach out to.

We plan to make the DRI list public over the next two weeks and will actively follow up in a dedicated public channel. We're looking forward to working more closely with the community! Cheers!

---

## Issue #N/A: [Roadmap] Three-Week Optimizations Sprint

**Link**: https://github.com/sgl-project/sglang/issues/7831
**State**: open
**Created**: 2025-07-08T00:31:35+00:00
**Comments**: 0
**Labels**: high priority, collaboration

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

@kushanam 

### Related resources

_No response_

---

## Issue #N/A: Development Roadmap (2025 H2)

**Link**: https://github.com/sgl-project/sglang/issues/7736
**State**: open
**Created**: 2025-07-03T06:04:23+00:00
**Comments**: 0
**Labels**: high priority, collaboration

### Description

The SGLang team is expected to complete planning for the H2 roadmap within the next two weeks. Stay tuned—exciting things are on the way!


---

## Issue #N/A: SGLang Router Architecture Improvement Proposal

**Link**: https://github.com/sgl-project/sglang/issues/7532
**State**: open
**Created**: 2025-06-25T20:06:12+00:00
**Comments**: 1
**Labels**: high priority, collaboration, router

### Description

# SGLang Router Architecture Improvement Proposal

## Table of Contents
1. [Summary](#summary)
2. [Current Architecture Overview](#current-architecture-overview)
3. [System Components](#system-components)
4. [Request Flow Analysis](#request-flow-analysis)
5. [Identified Pain Points](#identified-pain-points)
6. [Proposed Improvements](#proposed-improvements)
7. [Long-Term Vision](#long-term-vision)
8. [Implementation Phases](#implementation-phases)
9. [Risk Analysis](#risk-analysis)
10. [Success Metrics](#success-metrics)
11. [Conclusion](#conclusion)
12. [Appendix: Architecture Diagrams](#appendix-architecture-diagrams)

## Summary

This proposal outlines a architectural improvement plan for the SGLang Router, a high-performance load balancer that supports both traditional and disaggregated (Prefill-Decode) routing modes. The improvements focus on enhancing maintainability and extensibility without disrupting existing functionality. These changes lay the foundation for a long-term tran

[... truncated for brevity ...]

---

## Issue #N/A: [RFC] Bi-weekly release

**Link**: https://github.com/sgl-project/sglang/issues/7332
**State**: open
**Created**: 2025-06-18T23:17:05+00:00
**Comments**: 0
**Labels**: high priority, collaboration

### Description

After thorough internal discussions, the SGLang team has decided to standardize the release cycle as follows:

- A new version will be released every two weeks under normal circumstances (e.g., v0.4.8, v0.4.9).

- If urgent issues or high-priority features arise between regular releases, we may publish a patch release or an additional stable version as needed.

- Bi-weekly releases will typically occur around the middle and end of each month.

- Each release will aim to include a set of planned features, usually discussed and finalized by the SGLang team in advance.



---

## Issue #N/A: [Roadmap] Blackwell Support and Optimizations

**Link**: https://github.com/sgl-project/sglang/issues/7227
**State**: open
**Created**: 2025-06-16T06:07:50+00:00
**Comments**: 45
**Labels**: high priority, collaboration, blackwell

### Description

### Roadmap

- [x] ~~Initial support and optimizations for GB200, PD disaggregation, and large-scale EP~~ -- Done in https://lmsys.org/blog/2025-06-16-gb200-part-1/
- [x] Initial optimizations for prefill for large scale EP
- [ ] Optimize kernels for the Blackwell architecture
    - [ ] Communication kernels
    - [ ] Various smaller kernels
- [ ] Optimize for latency-oriented scenarios
- [ ] Computation-communication overlap

TODO: more

### Updates after Blog

* Prefill is slightly optimized, 13149 token/s/gpu for ISL 4096 (as usual all code are open sourced)

### Blog Reproduction

<details>

To reproduce [the blog post](https://lmsys.org/blog/2025-06-16-gb200-part-1/), here are the instructions:

#### 2025.07.12

To use the latest main, the following commands can be used.

Versions that I personally use to test (other versions may work as well)
* SGLang: https://github.com/sgl-project/sglang/commit/2a2d3478afe8cdb336888f2e6faa3775ac40254e
* sgl-kernel: the one inside SGLang
* DeepG

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

## Issue #N/A: [Feature] [Roadmap] OpenAI-Compatible Server Refactor

**Link**: https://github.com/sgl-project/sglang/issues/7068
**State**: open
**Created**: 2025-06-10T22:16:39+00:00
**Comments**: 11
**Labels**: high priority, collaboration

### Description

## 1. Overview and Motivation

The current SGLang OpenAI-compatible API is integrated within the monolithic `http_server.py`. This design mixes native SGLang endpoints with OpenAI-compatible endpoints, making it difficult to maintain, extend, and debug. High request concurrency has also revealed potential latency bottlenecks within the `openai_api/adapter.py` layer.

The goal of this project is to refactor the OpenAI-compatible API into a new, self-contained, and modular server. This will improve maintainability, extensibility, and performance, drawing inspiration from the successful modular design of vLLM's OpenAI API server.

## 2. Proposed Design

We will create a new, dedicated module for the OpenAI-compatible server with a clear, extensible structure.

### 2.1. New Directory Structure

The new module will be located at `sglang/python/sglang/srt/entrypoints/openai/`:

```
sglang/
└── python/
    └── sglang/
        └── srt/
            ├── entrypoints/
            │   ├── http_serv

[... truncated for brevity ...]

---

## Issue #N/A: [PD] Support Multi-Process for TokenizerManager

**Link**: https://github.com/sgl-project/sglang/issues/6553
**State**: open
**Created**: 2025-05-23T09:19:04+00:00
**Comments**: 0
**Labels**: enhancement, collaboration, deepseek

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.
The diagram below briefly outlines the process of a request from input to output：[Detailed Documentation]

<img width="789" alt="Image" src="https://github.com/user-attachments/assets/2e95df40-c9bd-4078-80da-77098881e62e" />


The TokenizerManager is responsible for three main tasks:

1.  Receiving r

[... truncated for brevity ...]

---

## Issue #N/A: Instruction for Running DeepSeek with Large-scale PD and EP

**Link**: https://github.com/sgl-project/sglang/issues/6017
**State**: open
**Created**: 2025-05-05T04:48:15+00:00
**Comments**: 504
**Labels**: collaboration, deepseek

### Description

## Using main branch

~~NOTE: The feature is already on main, but the performance still needs some improvements on main branch.~~ will be good after a few already opened PRs - PR 6680, 6727, 6728

~~NOTE: I will try other config like 4 node for P and 9 node for D later.~~ updated

### Environment Preparation

Use SGLang and DeepEP on master is sufficient. Also remember to upgrade Mooncake.

### 4P + 9D experiments

Start server
where DeepEP config can be tuned by https://github.com/sgl-project/sglang/pull/6742

```python
# prefill nodes
MC_TE_METRIC=true SGLANG_TBO_DEBUG=1 python3 -m sglang.launch_server --model-path /dev/shm/DeepSeek-V3-0324 --disaggregation-ib-device mlx5_1 --disaggregation-mode prefill --dist-init-addr 10.5.55.3:5757 --nnodes 4 --node-rank 0 --tp-size 32 --dp-size 32 --enable-dp-attention --decode-log-interval 1 --enable-deepep-moe --page-size 1 --host 0.0.0.0 --trust-remote-code --moe-dense-tp-size 1 --enable-dp-lm-head --disable-radix-cache --watchdog-timeout 1000

[... truncated for brevity ...]

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

## Issue #N/A: [Feature] support merge_state in sgl-kernel

**Link**: https://github.com/sgl-project/sglang/issues/5361
**State**: closed
**Created**: 2025-04-14T00:56:57+00:00
**Closed**: 2025-04-15T04:32:18+00:00
**Comments**: 3
**Labels**: high priority, collaboration, speculative-decoding

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

I have talked to @deftruth, and he will support it in the sgl-kernel today

### Related resources

_No response_

---

## Issue #N/A: [Roadmap] Llama 4 Support

**Link**: https://github.com/sgl-project/sglang/issues/5118
**State**: open
**Created**: 2025-04-07T08:06:44+00:00
**Comments**: 1
**Labels**: high priority, collaboration

### Description

- [x] Initial Llama 4 Support @CatherineSue @fzyzcjy @ispobock  @ch-wan  #5092 
- [x] Llama 4 User Guide @ch-wan @ispobock #5133
- [x] Vision Backbone Support @mickqian #5144 
- [ ] Local Attention Support in Various Attention Backbones
  - [x] FlashAttention V3
  - [ ] FlashInfer
  - [ ] Triton
- [ ] Quantization 
  - [x] FP8 @HandH1998 #5194
  - [ ] INT4 @AniZpZ
- [ ] Kernel Optimization
- [ ] Memory Optimization @tarinkk @Pb314314  #6563 
- [ ] EP Optimization
- [x] Llama4 Tool Call Support @CatherineSue #5725 


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

## Issue #N/A: [Roadmap] EP Enhancement

**Link**: https://github.com/sgl-project/sglang/issues/4734
**State**: open
**Created**: 2025-03-24T18:48:57+00:00
**Comments**: 30
**Labels**: high priority, collaboration

### Description

- [x] Support normal DeepEP buffer @liz-badada  #4232 
- [x] Support DeepEP with async transfer @fzyzcjy #4610 
- [x] Support low-latency DeepEP buffer
  - [x] Single-node TP @liz-badada #4767 
    - MaskedDeepGeMM is implemented by @laixinn @sleepcoo 
    - Improved by @yuleil #5277 
  - [x] Multi-node TP @liz-badada #5068 
  - [x] Support PD disaggregation @ch-wan  #5435 
- [ ] Integrate pplx-kernels @ruizhang1230 #5010 
- [ ] Optimize permutation overhead
  - [x] Implement Titon kernels @xutizhou #4643 
  - [ ] Fuse permutation with GroupedGeMM
- [x] Extend parallelism paradigm
  - [x] Extend DeepEP to a general TP paradigm @ch-wan @tarinkk #4770 
    - Fixed by @fzyzcjy #4883 
  - [x] Support `tp_size < ep_size`
    - `tp_size=1` @fzyzcjy #4836
- [x] Overlap two batches @fzyzcjy #4068 
- [x] Integrate continuous DeepGeMM @sleepcoo @xutizhou  #5626 
- [x] Record expert distribution @yuhsuan-t #4435 
  - Improved by @fzyzcjy #4957  
- [ ] Overlap communication with shared experts’ co

[... truncated for brevity ...]

---

## Issue #N/A: [Roadmap] FlashAttention3 Support as SGLang Attention Backend

**Link**: https://github.com/sgl-project/sglang/issues/4709
**State**: closed
**Created**: 2025-03-24T06:13:12+00:00
**Closed**: 2025-04-21T06:16:51+00:00
**Comments**: 5
**Labels**: high priority, collaboration

### Description


**Functionality**
- [x] Basic FA3 support including MHA Models (Llama, QWen and etc), Cuda Graph, Sliding Window (Gemma): https://github.com/sgl-project/sglang/pull/4680 @hebiao064 @qingquansong 
- [x] Support Page Size > 1 https://github.com/sgl-project/sglang/pull/4832 @hebiao064 
- [x] Support MLA for Deepseek-like models #4831 @Fridge003  
- [x] Support Speculative Decoding [PR1](https://github.com/sgl-project/sglang/pull/4951), [PR2](https://github.com/sgl-project/sglang/pull/5050/files), [PR3](https://github.com/sgl-project/sglang/pull/5168) [PR4](https://github.com/sgl-project/sglang/pull/5318) @qingquansong @hebiao064 @zcnrex 
- [x] Figure out how to build FA3 into SGLang: https://github.com/sgl-project/sglang/pull/4902 @yinfan98 
- [x] Add E2E Test like `sglang/test/srt/test_triton_attention_backend.py`: https://github.com/sgl-project/sglang/pull/4760 @yubofredwang 
- [x] Support Multimodal  https://github.com/sgl-project/sglang/pull/5103 @zcnrex @mickqian @yizhang2077 
- [x]

[... truncated for brevity ...]

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

## Issue #N/A: Development Roadmap (2025 H1)

**Link**: https://github.com/sgl-project/sglang/issues/4042
**State**: open
**Created**: 2025-03-04T00:09:49+00:00
**Comments**: 23
**Labels**: collaboration

### Description

Here is the development roadmap for 2025 H1. Contributions and feedback are welcome ([**Join Bi-weekly Development Meeting**](https://docs.google.com/document/d/1xEow4eIM152xNcRxqZz9VEcOiTQo8-CEuuQ5qTmkt-E/edit?usp=sharing)). The previous 2024 Q4 roadmap can be found in #1487

## Focus
- Throughput-oriented large-scale deployment similar to the [deepseek inference system](https://github.com/deepseek-ai/open-infra-index?tab=readme-ov-file#day-6---one-more-thing-deepseek-v3r1-inference-system-overview)
- Long context optimizations
- Low latency speculative decoding
- Reinforcement learning training framework integration
- Kernel optimizations

## Parallelism
- [x] Support PD disaggregation @ByronHsu  #4655
- [x] Support expert parallelism and load balancer #5524
- [x] Support pipeline parallelism @Ying1123 #5724
- [x] Support data parallelism attention compatible with all other parallelism #4390 
- [x] Support overlap communication in TP/EP @tom @Zhuohao-Li #4068
- [ ] Improvements of sg

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support for Cosmos-1.0-Autoregressive (World Foundation Models)

**Link**: https://github.com/sgl-project/sglang/issues/2844
**State**: closed
**Created**: 2025-01-12T08:49:54+00:00
**Closed**: 2025-05-16T00:19:24+00:00
**Comments**: 4
**Labels**: help wanted, collaboration, inactive

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

NVIDIA released **Cosmos World Foundation Models**: A family of highly performant pre-trained world foundation models purpose-built for generating physics-aware videos and world states for physical AI development.

The Cosmos autoregressive models are a collection of pre-trained world foundation models (WFMs) that are ideal for predicting and rapidly generating video sequences from video or image inputs for physical AI. They can serve as the building block for various applications or research that are related to world generation. 

[**Hugging Face**](https://huggingface.co/collections/nvidia/cosmos-6751e884dc10e013a0a0d8e6) | [**Code**](https://github.com/NVIDIA/Cosmos) | [**Paper**](https://arxiv.org/ab

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

## Issue #N/A: [Feature] Running multi-node offline engine inference ( via SLURM)

**Link**: https://github.com/sgl-project/sglang/issues/2561
**State**: closed
**Created**: 2024-12-23T15:24:49+00:00
**Closed**: 2025-01-31T23:58:27+00:00
**Comments**: 39
**Labels**: help wanted, collaboration, feature

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

A lot of academic institutions only allow access to larger node clusters via SLURM and it is not immediately clear how would I reuse the code to run Llama 405B BF16 on 2 nodes (by starting a server) to perform offline inference

### Related resources

_No response_

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

## Issue #N/A: [Feature] Integration SGLang into OpenRLHF

**Link**: https://github.com/sgl-project/sglang/issues/2506
**State**: closed
**Created**: 2024-12-17T22:50:04+00:00
**Closed**: 2025-02-16T07:53:44+00:00
**Comments**: 1
**Labels**: high priority, collaboration, inactive

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

We've implemented the weight update API for RLHF pipeline:

api:

https://github.com/sgl-project/sglang/blob/21e9e63ad56f8bd25663fa6907ed92f47a2b2724/python/sglang/srt/server.py#L214-L239

test case / usage:

https://github.com/sgl-project/sglang/blob/main/test/srt/test_update_weights_from_distributed.py

We will integrated SGLang into OpenRLHF this week. Here is the data for our accuracy and speed test.

https://huggingface.co/datasets/OpenRLHF/prompt-collection-v0.1-dev-rand5k

https://huggingface.co/datasets/OpenRLHF/prompt-collection-v0.1-dev-100k

Typically, 50K data requires several hours.

### Related resources

See above.

---

## Issue #N/A: [Feature] Benchmarking Performance on General Devices

**Link**: https://github.com/sgl-project/sglang/issues/2488
**State**: closed
**Created**: 2024-12-16T08:01:21+00:00
**Closed**: 2025-05-11T00:20:28+00:00
**Comments**: 4
**Labels**: enhancement, collaboration, inactive

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

We need to benchmark the speed performance of SGLang on various devices, at least different types of GPUs. This could give users a standard of the engine and whether their engines are working appropriately.

### Related resources

No such.

---

## Issue #N/A: [Feature] Do we have any plan for supporting MiniCPM-V 2.6?

**Link**: https://github.com/sgl-project/sglang/issues/2461
**State**: closed
**Created**: 2024-12-12T03:25:08+00:00
**Closed**: 2025-01-18T22:17:00+00:00
**Comments**: 12
**Labels**: collaboration

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Do we have any plan for supporting MiniCPM-V 2.6?

To my experience this 8B model has better performance than other 7B vlm models

### Related resources

https://github.com/OpenBMB/MiniCPM-V
https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/minicpmv.py

---

## Issue #N/A: Colab? 

**Link**: https://github.com/sgl-project/sglang/issues/14
**State**: closed
**Created**: 2024-01-16T20:08:21+00:00
**Closed**: 2024-07-25T06:32:37+00:00
**Comments**: 11
**Labels**: collaboration, inactive

### Description

Awesome project. We have a paper https://arxiv.org/abs/2310.14034 with really complicated KV caching that I would love to go back and implement in SGLang. 

I tried to get an example working in Colab for a demo, but I got kind of stuck getting the server running. 

This runs fine: 

!nohup python -m sglang.launch_server --model-path TheBloke/Mistral-7B-v0.1-AWQ --port 30000

But then when I run the following, 

```
%%script bash
curl http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Say this is a test",
    "max_tokens": 16,
    "temperature": 0
  }'
```

I just get this. 

```
KV cache pool leak detected!
Warning: available_size=75821, max_total_num_token=75833
KV cache pool leak detected!
Warning: available_size=75821, max_total_num_token=75833
KV cache pool leak detected!
Warning: available_size=75821, max_total_num_token=75833
KV cache pool leak detected!
Warning: available_size=75821, max_total_num

[... truncated for brevity ...]

---

